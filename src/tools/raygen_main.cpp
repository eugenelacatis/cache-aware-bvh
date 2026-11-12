// raygen: writes isotropic random lines for a tree's scene box.
// Usage: raygen <tree.bvh> <out.rays> <count> [seed]

#include <cstdio>
#include <cstdlib>
#include <string>

#include "bvh_io.h"
#include "raygen.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <tree.bvh> <out.rays> <count> [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t count = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;

    std::string error;
    TreeFile tree;
    if (!readTree(argv[1], tree, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], error.c_str());
        return EXIT_FAILURE;
    }
    std::vector<Ray> rays = isotropicLines(tree.header().sceneBox, count, seed);
    if (!writeRays(argv[2], RayKind::IsotropicLines, seed, rays, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }
    std::printf("Wrote %u isotropic lines to %s (seed %llu)\n", count, argv[2], (unsigned long long)seed);
    return 0;
}
