#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "obj_loader.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <mesh.obj>\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<Triangle> triangles;
    std::string error;
    if (!loadObj(argv[1], triangles, error)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], error.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Loaded %zu triangles from %s\n", triangles.size(), argv[1]);
    return EXIT_SUCCESS;
}
