#pragma once

// On-disk layouts from docs/FORMATS.md. All files are little-endian.

#include <cstddef>
#include <cstdint>

#include "hd.h"

constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kSectorBytes = 32;
constexpr uint32_t kSectionAlign = 128;

// Traversal stack entries. validateTree rejects trees deeper than this allows:
// a root-to-leaf path of d internal nodes needs at most d * (W - 1) + 1 entries.
constexpr int kMaxStack = 64;

// ---- Child references -----------------------------------------------------
// bit 31 = 0: internal child, bits 0..30 = node index
// bit 31 = 1: leaf, bits 27..30 = triangle count (1..15), bits 0..26 = first
//             sector of the triangle block, relative to the triangle section
// 0x80000000 (leaf with 0 triangles) = empty slot

constexpr uint32_t kLeafBit = 0x80000000u;
constexpr uint32_t kEmptyRef = kLeafBit;
constexpr uint32_t kMaxLeafTris = 15;
constexpr uint32_t kMaxLeafSector = (1u << 27) - 1;

HD inline bool refIsEmpty(uint32_t ref) { return ref == kEmptyRef; }
HD inline bool refIsLeaf(uint32_t ref) { return (ref & kLeafBit) != 0; }
HD inline uint32_t refNode(uint32_t ref) { return ref; }
HD inline uint32_t refLeafCount(uint32_t ref) { return (ref >> 27) & 0xFu; }
HD inline uint32_t refLeafSector(uint32_t ref) { return ref & kMaxLeafSector; }
HD inline uint32_t makeLeafRef(uint32_t firstSector, uint32_t count) {
    return kLeafBit | (count << 27) | firstSector;
}

// Sectors a leaf block of `count` triangles occupies (blocks start on a sector).
HD inline uint32_t leafSectors(uint32_t count) { return (count * 48u + kSectorBytes - 1) / kSectorBytes; }

// ---- Records --------------------------------------------------------------

struct Box {
    float lo[3];
    float hi[3];
};

// Binary node: 56 B of traversal data plus the children's P values = 64 B (2 sectors).
struct Node2 {
    Box box[2];
    uint32_t child[2];
    float p[2];
};

// 4-wide node: 112 B of traversal data plus the children's P values = 128 B (4 sectors).
struct Node4 {
    Box box[4];
    uint32_t child[4];
    float p[4];
};

template <int W> struct NodeOf;
template <> struct NodeOf<2> { using type = Node2; };
template <> struct NodeOf<4> { using type = Node4; };

struct Triangle {
    float v0[3];
    float v1[3];
    float v2[3];
    uint32_t id;  // index in the source mesh
    uint32_t pad[2];
};

enum class PSource : uint32_t { SurfaceArea = 0, Measured = 1 };
enum class TreeKind : uint32_t { BinarySah = 0, SahCollapse = 1, MsahCostCollapse = 2, MemoryAware = 3, Test = 255 };

struct TreeHeader {
    char magic[8];  // "CABVHTRE"
    uint32_t version;
    uint32_t width;      // 2 or 4
    uint32_t nodeBytes;  // 64 or 128
    uint32_t nodeCount;
    uint32_t triCount;
    uint32_t triSectorCount;
    uint64_t nodeOffset;  // bytes from file start, multiple of 128
    uint64_t triOffset;   // bytes from file start, multiple of 128
    Box sceneBox;         // root node's box
    float rootP;
    uint32_t pSource;   // PSource
    uint32_t treeKind;  // TreeKind
    float tau;          // memory-aware threshold; NaN for other kinds
    uint8_t reserved[40];
};

// ---- Ray file -------------------------------------------------------------

enum class RayKind : uint32_t { IsotropicLines = 0, Camera = 1, Bounce = 2 };

struct RayHeader {
    char magic[8];  // "CABVHRAY"
    uint32_t version;
    uint32_t rayCount;
    uint32_t rayKind;  // RayKind
    uint32_t reserved0;
    uint64_t seed;
};

struct Ray {
    float org[3];
    float tmin;
    float dir[3];
    float tmax;
};

// ---- Node-access trace ----------------------------------------------------

enum class TraceMode : uint32_t { ClosestHit = 0, AllHit = 1 };
enum class AccessKind : uint8_t { Node = 0, TriBlock = 1 };

struct TraceHeader {
    char magic[8];  // "CABVHTRC"
    uint32_t version;
    uint32_t mode;  // TraceMode
    uint32_t rayCount;
    uint32_t width;
    uint64_t eventCount;
};

struct TraceEvent {
    uint32_t ray;
    uint32_t firstSector;  // absolute: byte offset in the tree file / 32
    uint16_t sectorCount;
    uint8_t kind;  // AccessKind
    uint8_t reserved;
};

static_assert(sizeof(Box) == 24, "Box layout");
static_assert(sizeof(Node2) == 64, "Node2 must be 2 sectors");
static_assert(sizeof(Node4) == 128, "Node4 must be 4 sectors");
static_assert(sizeof(Triangle) == 48, "Triangle layout");
static_assert(sizeof(TreeHeader) == 128, "TreeHeader layout");
static_assert(offsetof(TreeHeader, nodeOffset) == 32, "TreeHeader layout");
static_assert(offsetof(TreeHeader, sceneBox) == 48, "TreeHeader layout");
static_assert(offsetof(TreeHeader, reserved) == 88, "TreeHeader layout");
static_assert(sizeof(RayHeader) == 32, "RayHeader layout");
static_assert(sizeof(Ray) == 32, "Ray layout");
static_assert(sizeof(TraceHeader) == 32, "TraceHeader layout");
static_assert(sizeof(TraceEvent) == 12, "TraceEvent layout");
