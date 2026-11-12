#include "sah_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace {

constexpr int kBinCount = 12;
constexpr double kTraversalCost = 1.0;  // In units of one triangle test.

// Reserve enough levels for balanced splits even with a full uint32_t range
// of triangle counts. The binary traversal needs one stack slot per level
// plus one for the leaf.
constexpr int kSahDepth = kMaxStack - 1 - std::numeric_limits<uint32_t>::digits;

Box emptyBox() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    return {{inf, inf, inf}, {-inf, -inf, -inf}};
}

void extend(Box& box, const Box& other) {
    for (int axis = 0; axis < 3; ++axis) {
        box.lo[axis] = std::min(box.lo[axis], other.lo[axis]);
        box.hi[axis] = std::max(box.hi[axis], other.hi[axis]);
    }
}

double surfaceArea(const Box& box) {
    const double x = double(box.hi[0]) - box.lo[0];
    const double y = double(box.hi[1]) - box.lo[1];
    const double z = double(box.hi[2]) - box.lo[2];
    return 2.0 * (x * y + y * z + z * x);
}

struct Primitive {
    Box box;
    std::array<double, 3> centroid;
    uint32_t triangleIndex;
};

using PrimitiveIter = std::vector<Primitive>::iterator;

Box boundsOf(PrimitiveIter first, PrimitiveIter last) {
    Box bounds = emptyBox();
    for (; first != last; ++first) extend(bounds, first->box);
    return bounds;
}

struct Bin {
    Box bounds = emptyBox();
    size_t count = 0;

    void add(const Bin& other) {
        extend(bounds, other.bounds);
        count += other.count;
    }
};

int binOf(double centroid, double low, double scale) {
    return std::min(kBinCount - 1, static_cast<int>((centroid - low) * scale));
}

struct Split {
    int axis = -1;
    int lastLeftBin = 0;
    double low = 0.0;
    double scale = 0.0;
    double weightedArea = std::numeric_limits<double>::infinity();
};

Split findSplit(PrimitiveIter first, PrimitiveIter last) {
    auto lo = first->centroid;
    auto hi = lo;
    for (auto p = first + 1; p != last; ++p) {
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(lo[axis], p->centroid[axis]);
            hi[axis] = std::max(hi[axis], p->centroid[axis]);
        }
    }

    Split best;
    for (int axis = 0; axis < 3; ++axis) {
        if (lo[axis] == hi[axis]) continue;
        const double scale = kBinCount / (hi[axis] - lo[axis]);
        std::array<Bin, kBinCount> bins{};
        for (auto p = first; p != last; ++p) {
            Bin& bin = bins[binOf(p->centroid[axis], lo[axis], scale)];
            extend(bin.bounds, p->box);
            ++bin.count;
        }

        auto left = bins;
        for (int i = 1; i < kBinCount - 1; ++i) left[i].add(left[i - 1]);
        Bin right;
        for (int i = kBinCount - 1; i > 0; --i) {
            right.add(bins[i]);
            const Bin& below = left[i - 1];
            if (below.count == 0 || right.count == 0) continue;
            const double weightedArea = below.count * surfaceArea(below.bounds) +
                                        right.count * surfaceArea(right.bounds);
            if (weightedArea < best.weightedArea) best = {axis, i - 1, lo[axis], scale, weightedArea};
        }
    }
    return best;
}

struct Builder {
    const std::vector<Triangle>& triangles;
    double rootArea;
    std::vector<Node2> nodes;
    std::vector<unsigned char> triSection;

    uint32_t emitLeaf(PrimitiveIter first, PrimitiveIter last) {
        const size_t offset = triSection.size();
        const size_t sector = offset / kSectorBytes;
        if (sector > kMaxLeafSector) return kEmptyRef;

        const auto count = static_cast<uint32_t>(last - first);
        triSection.resize(offset + leafSectors(count) * kSectorBytes);
        auto* dst = triSection.data() + offset;
        for (auto p = first; p != last; ++p, dst += sizeof(Triangle)) {
            // The resize zeroes both the record padding and the sector padding.
            std::memcpy(dst, &triangles[p->triangleIndex], offsetof(Triangle, pad));
        }
        return makeLeafRef(static_cast<uint32_t>(sector), count);
    }

    uint32_t emit(PrimitiveIter first, PrimitiveIter last, const Box& bounds, int depth) {
        const auto count = last - first;
        if (count == 1) return emitLeaf(first, last);

        const double area = surfaceArea(bounds);
        const Split split = depth < kSahDepth && area > 0.0 ? findSplit(first, last) : Split{};
        // Compare SAH costs multiplied by the parent area to avoid division.
        if (count <= kMaxLeafTris && (split.axis < 0 || kTraversalCost * area + split.weightedArea >= count * area))
            return emitLeaf(first, last);

        // A leaf over the format limit must split, even without a usable SAH plane.
        auto middle = first + count / 2;
        if (split.axis >= 0) {
            middle = std::partition(first, last, [&](const Primitive& p) {
                return binOf(p.centroid[split.axis], split.low, split.scale) <= split.lastLeftBin;
            });
        }

        const auto index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        Node2 node{};
        const PrimitiveIter edges[] = {first, middle, last};
        for (int c = 0; c < 2; ++c) {
            node.box[c] = boundsOf(edges[c], edges[c + 1]);
            // A scene with zero box area has no area ratio. Use a finite fallback.
            node.p[c] = rootArea > 0.0 ? static_cast<float>(surfaceArea(node.box[c]) / rootArea) : 1.0f;
            node.child[c] = emit(edges[c], edges[c + 1], node.box[c], depth + 1);
            if (node.child[c] == kEmptyRef) return kEmptyRef;
        }
        nodes[index] = node;
        return index;
    }
};

}  // namespace

bool buildBinarySah(const std::vector<Triangle>& triangles, TreeFile& out, std::string& error) {
    if (triangles.empty()) {
        error = "cannot build a BVH without triangles";
        return false;
    }
    // The last leaf can end beyond the largest encodable starting sector.
    const uint64_t maxTriangleBytes = (uint64_t(kMaxLeafSector) + leafSectors(kMaxLeafTris)) * kSectorBytes;
    if (triangles.size() > maxTriangleBytes / sizeof(Triangle)) {
        error = "too many triangles for the BVH format";
        return false;
    }

    std::vector<Primitive> primitives;
    primitives.reserve(triangles.size());
    for (uint32_t i = 0; i < triangles.size(); ++i) {
        const Triangle& triangle = triangles[i];
        Primitive primitive{};
        primitive.triangleIndex = i;
        for (int axis = 0; axis < 3; ++axis) {
            const float a = triangle.v0[axis], b = triangle.v1[axis], c = triangle.v2[axis];
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
                error = "non-finite coordinates in triangle " + std::to_string(triangle.id);
                return false;
            }
            primitive.box.lo[axis] = std::min({a, b, c});
            primitive.box.hi[axis] = std::max({a, b, c});
            primitive.centroid[axis] = (double(primitive.box.lo[axis]) + primitive.box.hi[axis]) * 0.5;
        }
        primitives.push_back(primitive);
    }

    const Box sceneBox = boundsOf(primitives.begin(), primitives.end());
    Builder builder{triangles, surfaceArea(sceneBox), {}, {}};
    const uint32_t root = builder.emit(primitives.begin(), primitives.end(), sceneBox, 0);
    if (root == kEmptyRef) {
        error = "triangle section exceeds the leaf sector address range";
        return false;
    }
    if (refIsLeaf(root)) {
        // The format always stores an internal record at index zero.
        Node2 node{};
        node.box[0] = sceneBox;
        node.child[0] = root;
        node.child[1] = kEmptyRef;
        node.p[0] = 1.0f;
        builder.nodes.push_back(node);
    }

    TreeFile tree = makeTree<2>(builder.nodes, builder.triSection, static_cast<uint32_t>(triangles.size()),
                                sceneBox, TreeKind::BinarySah, PSource::SurfaceArea,
                                std::numeric_limits<float>::quiet_NaN());
    if (!validateTree(tree, error)) return false;
    out = std::move(tree);
    error.clear();
    return true;
}
