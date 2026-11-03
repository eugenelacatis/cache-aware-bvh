#include <cstdio>
#include <fstream>
#include <vector>

#include "cuda_check.h"

__global__ void render(float3* fb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    fb[y * width + x] = make_float3(float(x) / (width - 1), float(height - 1 - y) / (height - 1), 0.25f);
}

static void writePpm(const char* path, const std::vector<float3>& fb, int width, int height) {
    std::ofstream out(path);
    out << "P3\n" << width << ' ' << height << "\n255\n";
    for (const float3& c : fb) {
        out << int(255.99f * c.x) << ' ' << int(255.99f * c.y) << ' ' << int(255.99f * c.z) << '\n';
    }
}

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "out.ppm";
    const int width = 1280;
    const int height = 720;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("L2 cache: %d bytes (%.1f MB)\n", prop.l2CacheSize, prop.l2CacheSize / (1024.0 * 1024.0));

    size_t l2FetchGranularity = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&l2FetchGranularity, cudaLimitMaxL2FetchGranularity));
    std::printf("Max L2 fetch granularity: %zu bytes\n", l2FetchGranularity);

    float3* dFb = nullptr;
    CUDA_CHECK(cudaMalloc(&dFb, width * height * sizeof(float3)));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    render<<<grid, block>>>(dFb, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float3> fb(width * height);
    CUDA_CHECK(cudaMemcpy(fb.data(), dFb, fb.size() * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dFb));

    writePpm(outPath, fb, width, height);
    std::printf("Wrote %s\n", outPath);
    return 0;
}
