#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "obj_loader.h"
#include "sah_builder.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <mesh.obj> <out.bvh>\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<Triangle> triangles;
    std::string error;
    if (!loadObj(argv[1], triangles, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], error.c_str());
        return EXIT_FAILURE;
    }

    TreeFile tree;
    if (!buildBinarySah(triangles, tree, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], error.c_str());
        return EXIT_FAILURE;
    }
    if (!writeTree(argv[2], tree, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
    }

    const TreeHeader& header = tree.header();
    std::printf("Wrote %s: %u triangles, %u binary nodes, %zu bytes\n", argv[2], header.triCount,
                header.nodeCount, tree.bytes.size());
    return EXIT_SUCCESS;
}
