#pragma once

// Marks functions that run on both CPU and GPU. Empty outside nvcc, so the
// host compiler can build the CPU tests and tools from the same code.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
