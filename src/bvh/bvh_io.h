#pragma once

// Host-side readers and writers for the tree, ray, and trace files (docs/FORMATS.md).

#include <string>
#include <vector>

#include "bvh_format.h"

// A whole tree file kept as one byte buffer, so the GPU can upload it with one
// copy and trace sectors (file offset / 32) match device addresses.
struct TreeFile {
    std::vector<unsigned char> bytes;

    const TreeHeader& header() const { return *reinterpret_cast<const TreeHeader*>(bytes.data()); }
    template <int W> const typename NodeOf<W>::type* nodes() const {
        return reinterpret_cast<const typename NodeOf<W>::type*>(bytes.data() + header().nodeOffset);
    }
    const unsigned char* triSection() const { return bytes.data() + header().triOffset; }
};

// Reads and validates a tree file. On failure returns false and sets `error`.
bool readTree(const std::string& path, TreeFile& out, std::string& error);

// Checks header fields, section bounds, and every child reference.
bool validateTree(const TreeFile& tree, std::string& error);

// Builds a tree file from nodes (node 0 is the root) and a triangle section.
// `triSection` holds 32-B sectors, with each leaf block starting on a sector.
template <int W>
TreeFile makeTree(const std::vector<typename NodeOf<W>::type>& nodes, const std::vector<unsigned char>& triSection,
                  uint32_t triCount, const Box& sceneBox, TreeKind kind, PSource pSource, float tau);

bool writeTree(const std::string& path, const TreeFile& tree, std::string& error);

// Every triangle in the tree's leaf blocks, in node order. Used by the brute-force tracer.
std::vector<Triangle> treeTriangles(const TreeFile& tree);

bool readRays(const std::string& path, RayHeader& header, std::vector<Ray>& rays, std::string& error);
bool writeRays(const std::string& path, RayKind kind, uint64_t seed, const std::vector<Ray>& rays, std::string& error);

bool readTrace(const std::string& path, TraceHeader& header, std::vector<TraceEvent>& events, std::string& error);
bool writeTrace(const std::string& path, TraceMode mode, uint32_t rayCount, uint32_t width,
                const std::vector<TraceEvent>& events, std::string& error);
