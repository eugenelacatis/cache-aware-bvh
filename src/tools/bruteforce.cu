// bruteforce: GPU reference tracer. Each thread tests one ray against every
// triangle, then the host checks the hits against CPU closest-hit traversal.
// Usage: bruteforce <tree.bvh> <in.rays>
//
// Not yet compiled or run on a GPU. Only the host logic has been checked, by
// building this file as plain C++ against a stubbed CUDA runtime.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bvh_io.h"
#include "cuda_check.h"
#include "traverse.h"

__global__ void bruteForceKernel(const Triangle* tris, uint32_t triCount, const Ray* rays, uint32_t rayCount,
                                 Hit* hits) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rayCount) return;
    const Ray ray = rays[i];
    Hit hit{ray.tmax, UINT32_MAX};
    for (uint32_t k = 0; k < triCount; ++k) {
        float t;
        if (intersectTriangle(tris[k], ray, ray.tmin, hit.t, t)) {
            hit.t = t;
            hit.triId = tris[k].id;
        }
    }
    hits[i] = hit;
}

template <int W>
static Hit cpuClosest(const TreeFile& tree, const Ray& ray) {
    return traceClosest<W>(tree.nodes<W>(), tree.triSection(), tree.header().sceneBox, ray);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tree.bvh> <in.rays>\n", argv[0]);
        return EXIT_FAILURE;
    }
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
    std::vector<Triangle> tris = treeTriangles(tree);
    const uint32_t triCount = static_cast<uint32_t>(tris.size());
    const uint32_t rayCount = static_cast<uint32_t>(rays.size());

    Triangle* dTris = nullptr;
    Ray* dRays = nullptr;
    Hit* dHits = nullptr;
    CUDA_CHECK(cudaMalloc(&dTris, tris.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&dRays, rays.size() * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&dHits, rays.size() * sizeof(Hit)));
    CUDA_CHECK(cudaMemcpy(dTris, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRays, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const uint32_t block = 128;
    const uint32_t grid = (rayCount + block - 1) / block;
    CUDA_CHECK(cudaEventRecord(start));
    bruteForceKernel<<<grid, block>>>(dTris, triCount, dRays, rayCount, dHits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<Hit> hits(rays.size());
    CUDA_CHECK(cudaMemcpy(hits.data(), dHits, hits.size() * sizeof(Hit), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dTris));
    CUDA_CHECK(cudaFree(dRays));
    CUDA_CHECK(cudaFree(dHits));

    // nvcc fuses multiply-adds, so t can differ from the CPU in the last bits.
    // A mismatch is a hit/miss disagreement or a relative t error above 1e-4.
    uint32_t hitCount = 0, exact = 0, mismatches = 0;
    for (uint32_t i = 0; i < rayCount; ++i) {
        Hit c = tree.header().width == 2 ? cpuClosest<2>(tree, rays[i]) : cpuClosest<4>(tree, rays[i]);
        const Hit& g = hits[i];
        if (g.triId != UINT32_MAX) ++hitCount;
        if (g.t == c.t && g.triId == c.triId) ++exact;
        bool sameHit = (g.triId == UINT32_MAX) == (c.triId == UINT32_MAX);
        if (!sameHit || std::fabs(g.t - c.t) > 1e-4f * std::fmax(1.0f, std::fabs(c.t))) {
            if (mismatches < 10)
                std::fprintf(stderr, "ray %u: GPU t=%.9g tri=%u, CPU t=%.9g tri=%u\n", i, g.t, g.triId, c.t, c.triId);
            ++mismatches;
        }
    }

    std::printf("%u rays x %u triangles in %.3f ms (%.2f Mrays/s)\n", rayCount, triCount, ms,
                rayCount / (ms * 1e3));
    std::printf("%u hits; %u of %u rays match CPU traversal exactly; %u mismatches\n", hitCount, exact, rayCount,
                mismatches);
    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
