// tracedump: CPU node-access trace for a tree and a ray file.
// Usage: tracedump <tree.bvh> <in.rays> <out.trace> [closest|all]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bvh_io.h"
#include "trace_dump.h"

int main(int argc, char** argv) {
    if (argc < 4 || (argc > 4 && std::strcmp(argv[4], "closest") != 0 && std::strcmp(argv[4], "all") != 0)) {
        std::fprintf(stderr, "usage: %s <tree.bvh> <in.rays> <out.trace> [closest|all]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const TraceMode mode = argc > 4 && std::strcmp(argv[4], "all") == 0 ? TraceMode::AllHit : TraceMode::ClosestHit;

    std::string error;
    TreeFile tree;
    if (!readTree(argv[1], tree, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], error.c_str());
        return EXIT_FAILURE;
    }
    RayHeader rh;
    std::vector<Ray> rays;
    if (!readRays(argv[2], rh, rays, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[2], error.c_str());
        return EXIT_FAILURE;
    }

    std::vector<RayStats> stats;
    std::vector<TraceEvent> events = dumpTrace(tree, rays, mode, &stats);
    if (!writeTrace(argv[3], mode, rh.rayCount, tree.header().width, events, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    double boxes = 0, tris = 0;
    for (const RayStats& s : stats) boxes += s.boxesVisited, tris += s.trianglesTested;
    const double n = rays.empty() ? 1.0 : double(rays.size());
    std::printf("Wrote %zu events for %u rays to %s (%s)\n", events.size(), rh.rayCount, argv[3],
                mode == TraceMode::AllHit ? "all-hit" : "closest-hit");
    std::printf("Per ray: %.2f boxes visited, %.2f triangles tested\n", boxes / n, tris / n);
    return 0;
}
