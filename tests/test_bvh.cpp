// CPU tests for the file formats, reader, traversal, and ray generator.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bvh_io.h"
#include "raygen.h"
#include "traverse.h"

static int failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            ++failures;                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
        }                                                                  \
    } while (0)

// ---- A small median-split builder, only for producing test trees ----------

static Box emptyBox() { return {{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}}; }

static void grow(Box& b, const float* p) {
    for (int a = 0; a < 3; ++a) {
        b.lo[a] = std::fmin(b.lo[a], p[a]);
        b.hi[a] = std::fmax(b.hi[a], p[a]);
    }
}

static float area(const Box& b) {
    float d[3] = {b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]};
    return 2.0f * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]);
}

struct BNode {
    Box box;
    int left = -1, right = -1;  // -1: leaf
    uint32_t first = 0, count = 0;
};

struct TestBuilder {
    const std::vector<Triangle>& tris;
    std::vector<uint32_t> order;
    std::vector<BNode> bnodes;

    explicit TestBuilder(const std::vector<Triangle>& t) : tris(t), order(t.size()) {
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        build(0, static_cast<uint32_t>(t.size()));
    }

    int build(uint32_t first, uint32_t count) {
        BNode n;
        n.box = emptyBox();
        for (uint32_t i = first; i < first + count; ++i) {
            grow(n.box, tris[order[i]].v0);
            grow(n.box, tris[order[i]].v1);
            grow(n.box, tris[order[i]].v2);
        }
        n.first = first;
        n.count = count;
        int idx = static_cast<int>(bnodes.size());
        bnodes.push_back(n);
        if (count <= 4) return idx;

        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (n.box.hi[a] - n.box.lo[a] > n.box.hi[axis] - n.box.lo[axis]) axis = a;
        auto centroid = [&](uint32_t t) { return tris[t].v0[axis] + tris[t].v1[axis] + tris[t].v2[axis]; };
        uint32_t half = count / 2;
        std::nth_element(order.begin() + first, order.begin() + first + half, order.begin() + first + count,
                         [&](uint32_t x, uint32_t y) { return centroid(x) < centroid(y); });
        int l = build(first, half);
        int r = build(first + half, count - half);
        bnodes[idx].left = l;
        bnodes[idx].right = r;
        return idx;
    }

    // Writes one leaf block, starting on a sector, and returns its reference.
    uint32_t emitLeaf(const BNode& b, std::vector<unsigned char>& triSection) {
        triSection.resize((triSection.size() + kSectorBytes - 1) / kSectorBytes * kSectorBytes);
        uint32_t sector = static_cast<uint32_t>(triSection.size() / kSectorBytes);
        for (uint32_t i = b.first; i < b.first + b.count; ++i) {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(&tris[order[i]]);
            triSection.insert(triSection.end(), src, src + sizeof(Triangle));
        }
        return makeLeafRef(sector, b.count);
    }

    // Collapses the binary tree to width W by repeatedly opening the internal
    // child with the largest surface area. Not the Ylitie DP; good enough to
    // exercise the reader and traversal.
    template <int W>
    uint32_t emit(int bi, std::vector<typename NodeOf<W>::type>& nodes, std::vector<unsigned char>& triSection,
                  float rootArea) {
        uint32_t idx = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        std::vector<int> kids;
        if (bnodes[bi].left < 0) {
            kids.push_back(bi);  // tiny scene: the root is one leaf
        } else {
            kids = {bnodes[bi].left, bnodes[bi].right};
        }
        while (int(kids.size()) < W) {
            int best = -1;
            for (int k = 0; k < int(kids.size()); ++k)
                if (bnodes[kids[k]].left >= 0 && (best < 0 || area(bnodes[kids[k]].box) > area(bnodes[kids[best]].box)))
                    best = k;
            if (best < 0) break;
            int open = kids[best];
            kids[best] = bnodes[open].left;
            kids.push_back(bnodes[open].right);
        }
        typename NodeOf<W>::type node{};
        for (int c = 0; c < W; ++c) node.child[c] = kEmptyRef;
        for (int c = 0; c < int(kids.size()); ++c) {
            const BNode& b = bnodes[kids[c]];
            node.box[c] = b.box;
            node.p[c] = area(b.box) / rootArea;
            node.child[c] = b.left < 0 ? emitLeaf(b, triSection) : emit<W>(kids[c], nodes, triSection, rootArea);
        }
        nodes[idx] = node;
        return idx;
    }

    template <int W>
    TreeFile tree() {
        std::vector<typename NodeOf<W>::type> nodes;
        std::vector<unsigned char> triSection;
        emit<W>(0, nodes, triSection, area(bnodes[0].box));
        return makeTree<W>(nodes, triSection, static_cast<uint32_t>(tris.size()), bnodes[0].box, TreeKind::Test,
                           PSource::SurfaceArea, NAN);
    }
};

static std::vector<Triangle> randomTriangles(uint32_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> pos(-10.0f, 10.0f), off(-0.5f, 0.5f);
    std::vector<Triangle> tris(n);
    for (uint32_t i = 0; i < n; ++i) {
        Triangle& t = tris[i];
        std::memset(&t, 0, sizeof(t));
        for (int a = 0; a < 3; ++a) {
            t.v0[a] = pos(rng);
            t.v1[a] = t.v0[a] + off(rng);
            t.v2[a] = t.v0[a] + off(rng);
        }
        t.id = i;
    }
    return tris;
}

static Hit bruteForce(const std::vector<Triangle>& tris, const Ray& ray) {
    Hit hit{ray.tmax, UINT32_MAX};
    for (const Triangle& t : tris) {
        float th;
        if (intersectTriangle(t, ray, ray.tmin, hit.t, th)) {
            hit.t = th;
            hit.triId = t.id;
        }
    }
    return hit;
}

// ---- Tests ---------------------------------------------------------------

template <int W>
static void testTraversalMatchesBruteForce(const std::vector<Triangle>& tris, const std::vector<Ray>& rays) {
    TestBuilder builder(tris);
    TreeFile tree = builder.tree<W>();
    std::string error;
    CHECK(validateTree(tree, error), "W=%d: %s", W, error.c_str());

    int hits = 0, mismatches = 0;
    for (const Ray& r : rays) {
        Hit a = traceClosest<W>(tree.nodes<W>(), tree.triSection(), tree.header().sceneBox, r);
        Hit b = bruteForce(tris, r);
        if (b.triId != UINT32_MAX) ++hits;
        // Ties at equal t may pick different triangles; the distance must match.
        if (a.t != b.t || (a.triId == UINT32_MAX) != (b.triId == UINT32_MAX)) ++mismatches;
    }
    CHECK(mismatches == 0, "W=%d: %d of %zu rays differ from brute force", W, mismatches, rays.size());
    CHECK(hits > int(rays.size()) / 20, "W=%d: only %d hits, test is too weak", W, hits);
    std::printf("  W=%d: %zu rays, %d hits, %u nodes\n", W, rays.size(), hits, tree.header().nodeCount);
}

static void testRoundTrip(const std::vector<Triangle>& tris, const std::string& dir) {
    TestBuilder builder(tris);
    TreeFile tree = builder.tree<4>();
    std::string path = dir + "/test_tree.bvh", error;
    CHECK(writeTree(path, tree, error), "%s", error.c_str());
    TreeFile back;
    CHECK(readTree(path, back, error), "%s", error.c_str());
    CHECK(back.bytes == tree.bytes, "tree bytes changed on round trip");
    const TreeHeader& h = back.header();
    CHECK(h.width == 4 && h.nodeBytes == 128 && h.nodeOffset == 128 && h.triOffset % 128 == 0, "header fields");
    CHECK(h.triCount == tris.size() && h.rootP == 1.0f && std::isnan(h.tau), "header fields");

    // Every triangle appears exactly once across the leaf blocks.
    std::vector<int> seen(tris.size(), 0);
    const Node4* nodes = back.nodes<4>();
    for (uint32_t i = 0; i < h.nodeCount; ++i)
        for (uint32_t ref : nodes[i].child)
            if (!refIsEmpty(ref) && refIsLeaf(ref)) {
                const Triangle* t = leafTriangles(back.triSection(), ref);
                for (uint32_t k = 0; k < refLeafCount(ref); ++k) ++seen[t[k].id];
            }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }), "triangle missing or duplicated");

    std::vector<Ray> rays = isotropicLines(h.sceneBox, 1000, 7), raysBack;
    RayHeader rh;
    std::string rpath = dir + "/test.rays";
    CHECK(writeRays(rpath, RayKind::IsotropicLines, 7, rays, error), "%s", error.c_str());
    CHECK(readRays(rpath, rh, raysBack, error), "%s", error.c_str());
    CHECK(rh.rayCount == 1000 && rh.seed == 7 && std::memcmp(rays.data(), raysBack.data(), 1000 * sizeof(Ray)) == 0,
          "ray file round trip");
}

static void testValidatorRejects(const std::vector<Triangle>& tris) {
    TestBuilder builder(tris);
    std::string error;

    TreeFile bad = builder.tree<4>();
    reinterpret_cast<Node4*>(bad.bytes.data() + bad.header().nodeOffset)[0].child[0] = 999999;
    CHECK(!validateTree(bad, error), "accepted an out-of-range child index");

    bad = builder.tree<4>();
    reinterpret_cast<Node4*>(bad.bytes.data() + bad.header().nodeOffset)[0].child[0] = makeLeafRef(kMaxLeafSector, 4);
    CHECK(!validateTree(bad, error), "accepted a leaf block past the triangle section");

    bad = builder.tree<4>();
    bad.bytes[0] = 'X';
    CHECK(!validateTree(bad, error), "accepted a bad magic");
}

// Crofton check: among lines that hit box A, the fraction that also hit a box
// B inside A should be SA(B) / SA(A).
static void testIsotropicLines() {
    Box outer = {{-1, -2, -3}, {3, 2, 1}};
    Box inner = {{0, -1, -1}, {1, 1, 0}};
    std::vector<Ray> rays = isotropicLines(outer, 400000, 42);
    int hitOuter = 0, hitInner = 0, badOrigin = 0;
    float r2 = 4 + 4 + 4;  // half extents (2, 2, 2)
    for (const Ray& r : rays) {
        float dx = r.org[0] - 1, dy = r.org[1], dz = r.org[2] + 1;
        if (std::fabs(dx * dx + dy * dy + dz * dz - r2) > 1e-3f * r2) ++badOrigin;
        RayInv ri = makeRayInv(r);
        float t;
        if (intersectBox(outer, ri, r.tmin, r.tmax, t)) {
            ++hitOuter;
            if (intersectBox(inner, ri, r.tmin, r.tmax, t)) ++hitInner;
        }
    }
    double measured = double(hitInner) / hitOuter;
    double expected = area(inner) / area(outer);
    CHECK(badOrigin == 0, "%d rays start off the bounding sphere", badOrigin);
    CHECK(std::fabs(measured - expected) < 0.005, "P(inner | outer) = %.4f, expected %.4f", measured, expected);
    std::printf("  P(inner | outer) = %.4f, SA ratio = %.4f\n", measured, expected);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    std::vector<Triangle> tris = randomTriangles(3000, 1);
    Box scene = emptyBox();
    for (const Triangle& t : tris) grow(scene, t.v0), grow(scene, t.v1), grow(scene, t.v2);
    std::vector<Ray> rays = isotropicLines(scene, 20000, 2);

    std::printf("traversal vs brute force\n");
    testTraversalMatchesBruteForce<2>(tris, rays);
    testTraversalMatchesBruteForce<4>(tris, rays);
    std::printf("tiny scene (root is one leaf)\n");
    std::vector<Triangle> few(3, tris[0]);  // three overlapping copies, shifted
    for (uint32_t k = 1; k < 3; ++k) {
        for (float* v : {few[k].v0, few[k].v1, few[k].v2}) v[0] += 0.1f * k;
        few[k].id = k;
    }
    few[0].id = 0;
    Box fewBox = emptyBox();
    for (const Triangle& t : few) grow(fewBox, t.v0), grow(fewBox, t.v1), grow(fewBox, t.v2);
    testTraversalMatchesBruteForce<4>(few, isotropicLines(fewBox, 20000, 3));
    std::printf("file round trip\n");
    testRoundTrip(tris, dir);
    std::printf("validator\n");
    testValidatorRejects(tris);
    std::printf("isotropic lines\n");
    testIsotropicLines();

    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
