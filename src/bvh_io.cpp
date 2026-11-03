#include "bvh_io.h"

#include <cstring>
#include <fstream>
#include <iterator>

namespace {

constexpr char kTreeMagic[8] = {'C', 'A', 'B', 'V', 'H', 'T', 'R', 'E'};
constexpr char kRayMagic[8] = {'C', 'A', 'B', 'V', 'H', 'R', 'A', 'Y'};

uint64_t alignUp(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

bool readFile(const std::string& path, std::vector<unsigned char>& out, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

bool writeFile(const std::string& path, const void* data, size_t size, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) {
        error = "cannot write " + path;
        return false;
    }
    return true;
}

template <int W>
bool validateNodes(const TreeFile& tree, std::string& error) {
    const TreeHeader& h = tree.header();
    const auto* nodes = tree.nodes<W>();
    for (uint32_t i = 0; i < h.nodeCount; ++i) {
        bool sawEmpty = false;
        int used = 0;
        for (int c = 0; c < W; ++c) {
            uint32_t ref = nodes[i].child[c];
            if (refIsEmpty(ref)) {
                sawEmpty = true;
                continue;
            }
            if (sawEmpty) {
                error = "node " + std::to_string(i) + ": child after an empty slot";
                return false;
            }
            ++used;
            if (refIsLeaf(ref)) {
                uint32_t n = refLeafCount(ref);
                if (n == 0) {
                    error = "node " + std::to_string(i) + ": leaf with no triangles";
                    return false;
                }
                if (uint64_t(refLeafSector(ref)) + leafSectors(n) > h.triSectorCount) {
                    error = "node " + std::to_string(i) + ": leaf block past the triangle section";
                    return false;
                }
            } else if (refNode(ref) == 0 || refNode(ref) >= h.nodeCount) {
                error = "node " + std::to_string(i) + ": bad child index " + std::to_string(refNode(ref));
                return false;
            }
        }
        if (used == 0) {
            error = "node " + std::to_string(i) + " has no children";
            return false;
        }
    }

    // Every node reachable from the root exactly once, and shallow enough for the traversal stack.
    std::vector<uint32_t> depth(h.nodeCount, 0);
    std::vector<uint32_t> queue{0};
    depth[0] = 1;
    for (size_t q = 0; q < queue.size(); ++q) {
        uint32_t i = queue[q];
        if (depth[i] * (W - 1) + 1 > uint32_t(kMaxStack)) {
            error = "tree too deep for the traversal stack (depth " + std::to_string(depth[i]) + ")";
            return false;
        }
        for (int c = 0; c < W; ++c) {
            uint32_t ref = nodes[i].child[c];
            if (refIsEmpty(ref) || refIsLeaf(ref)) continue;
            if (depth[ref] != 0) {
                error = "node " + std::to_string(ref) + " has more than one parent";
                return false;
            }
            depth[ref] = depth[i] + 1;
            queue.push_back(ref);
        }
    }
    if (queue.size() != h.nodeCount) {
        error = std::to_string(h.nodeCount - queue.size()) + " nodes unreachable from the root";
        return false;
    }
    return true;
}

}  // namespace

bool validateTree(const TreeFile& tree, std::string& error) {
    if (tree.bytes.size() < sizeof(TreeHeader)) {
        error = "file smaller than the header";
        return false;
    }
    const TreeHeader& h = tree.header();
    if (std::memcmp(h.magic, kTreeMagic, 8) != 0) {
        error = "not a tree file (bad magic)";
        return false;
    }
    if (h.version != kFormatVersion) {
        error = "unsupported version " + std::to_string(h.version);
        return false;
    }
    if (!((h.width == 2 && h.nodeBytes == sizeof(Node2)) || (h.width == 4 && h.nodeBytes == sizeof(Node4)))) {
        error = "width and node size do not match";
        return false;
    }
    if (h.nodeCount == 0) {
        error = "tree has no nodes";
        return false;
    }
    uint64_t nodeEnd = h.nodeOffset + uint64_t(h.nodeCount) * h.nodeBytes;
    uint64_t triEnd = h.triOffset + uint64_t(h.triSectorCount) * kSectorBytes;
    if (h.nodeOffset % kSectionAlign != 0 || h.triOffset % kSectionAlign != 0 || h.nodeOffset < sizeof(TreeHeader) ||
        h.triOffset < nodeEnd || triEnd > tree.bytes.size()) {
        error = "section offsets out of range or misaligned";
        return false;
    }
    return h.width == 2 ? validateNodes<2>(tree, error) : validateNodes<4>(tree, error);
}

bool readTree(const std::string& path, TreeFile& out, std::string& error) {
    return readFile(path, out.bytes, error) && validateTree(out, error);
}

template <int W>
TreeFile makeTree(const std::vector<typename NodeOf<W>::type>& nodes, const std::vector<unsigned char>& triSection,
                  uint32_t triCount, const Box& sceneBox, TreeKind kind, PSource pSource, float tau) {
    using Node = typename NodeOf<W>::type;
    TreeHeader h{};
    std::memcpy(h.magic, kTreeMagic, 8);
    h.version = kFormatVersion;
    h.width = W;
    h.nodeBytes = sizeof(Node);
    h.nodeCount = static_cast<uint32_t>(nodes.size());
    h.triCount = triCount;
    h.triSectorCount = static_cast<uint32_t>(alignUp(triSection.size(), kSectorBytes) / kSectorBytes);
    h.nodeOffset = alignUp(sizeof(TreeHeader), kSectionAlign);
    h.triOffset = alignUp(h.nodeOffset + nodes.size() * sizeof(Node), kSectionAlign);
    h.sceneBox = sceneBox;
    h.rootP = 1.0f;
    h.pSource = static_cast<uint32_t>(pSource);
    h.treeKind = static_cast<uint32_t>(kind);
    h.tau = tau;

    TreeFile tree;
    tree.bytes.assign(h.triOffset + uint64_t(h.triSectorCount) * kSectorBytes, 0);
    std::memcpy(tree.bytes.data(), &h, sizeof(h));
    std::memcpy(tree.bytes.data() + h.nodeOffset, nodes.data(), nodes.size() * sizeof(Node));
    std::memcpy(tree.bytes.data() + h.triOffset, triSection.data(), triSection.size());
    return tree;
}

template TreeFile makeTree<2>(const std::vector<Node2>&, const std::vector<unsigned char>&, uint32_t, const Box&,
                              TreeKind, PSource, float);
template TreeFile makeTree<4>(const std::vector<Node4>&, const std::vector<unsigned char>&, uint32_t, const Box&,
                              TreeKind, PSource, float);

bool writeTree(const std::string& path, const TreeFile& tree, std::string& error) {
    return writeFile(path, tree.bytes.data(), tree.bytes.size(), error);
}

bool readRays(const std::string& path, RayHeader& header, std::vector<Ray>& rays, std::string& error) {
    std::vector<unsigned char> bytes;
    if (!readFile(path, bytes, error)) return false;
    if (bytes.size() < sizeof(RayHeader)) {
        error = "file smaller than the header";
        return false;
    }
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.magic, kRayMagic, 8) != 0 || header.version != kFormatVersion) {
        error = "not a version 1 ray file";
        return false;
    }
    if (bytes.size() != sizeof(RayHeader) + uint64_t(header.rayCount) * sizeof(Ray)) {
        error = "ray count does not match file size";
        return false;
    }
    rays.resize(header.rayCount);
    std::memcpy(rays.data(), bytes.data() + sizeof(RayHeader), rays.size() * sizeof(Ray));
    return true;
}

bool writeRays(const std::string& path, RayKind kind, uint64_t seed, const std::vector<Ray>& rays,
               std::string& error) {
    RayHeader h{};
    std::memcpy(h.magic, kRayMagic, 8);
    h.version = kFormatVersion;
    h.rayCount = static_cast<uint32_t>(rays.size());
    h.rayKind = static_cast<uint32_t>(kind);
    h.seed = seed;
    std::vector<unsigned char> bytes(sizeof(h) + rays.size() * sizeof(Ray));
    std::memcpy(bytes.data(), &h, sizeof(h));
    std::memcpy(bytes.data() + sizeof(h), rays.data(), rays.size() * sizeof(Ray));
    return writeFile(path, bytes.data(), bytes.size(), error);
}
