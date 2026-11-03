# cache-aware-bvh

CMPE214 project by Eugene Lacatis, Harsha Krishnaswamy, and Diego Mateos. We test whether building a 4-wide BVH to minimize memory fetches, with a term for nodes that are likely already cached, reduces the data a GPU fetches per ray compared with the standard SAH collapse. Every tree runs through the same CUDA traversal kernel and is profiled with Nsight. We prove that our collapse is optimal inside a stated cache model, then measure how closely real GPUs follow that model.

The proposal is in `docs/`, and the roles, schedule, and proof outline are in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/main.cu` | Week 1 sanity check: prints the GPU name and compute capability, runs one kernel, and writes a test image |
| `src/bruteforce.cu` | `bruteforce` tool: GPU reference tracer that tests every ray against every triangle and checks the hits against CPU traversal. Not yet tested on a GPU |
| `src/cuda_check.h` | `CUDA_CHECK` macro that exits with the CUDA error name, file, and line |
| `src/bvh_format.h` | Structs for the tree, ray, and trace files in `docs/FORMATS.md` |
| `src/bvh_io.cpp` | Reads, writes, and validates tree and ray files, and reads and writes trace files |
| `src/traverse.h` | Ray-box and ray-triangle tests, closest-hit and all-hit traversal for 2-wide and 4-wide trees, and per-ray counters (boxes visited, triangles tested), shared by CPU and GPU |
| `src/raygen.cpp`, `src/raygen_main.cpp` | `raygen` tool that writes isotropic random lines for a tree |
| `src/trace_dump.cpp`, `src/tracedump_main.cpp` | `tracedump` tool that runs the traversal on the CPU and writes a node-access trace |
| `tests/test_bvh.cpp` | CPU tests: closest-hit, all-hit, and counters against brute force, the trace dump, file round trips, validator, and the isotropy of the random lines |
| `CMakeLists.txt` | Builds `pathtracer` for sm_75 and sm_120 in Release with `-lineinfo`, plus `bruteforce`, `raygen`, `tracedump`, and `bvh_tests` |
| `docs/FORMATS.md` | Byte layouts of the tree, ray, and node-access trace files |
| `docs/PROJECT_PLAN.md` | Roles, schedule, cache model, theorems, and benchmark matrix |
| `docs/LEARNING.md` | Notes from the earlier path tracer idea, not part of the current plan |

## Build and run

Needs CMake 3.24 or newer and the CUDA toolkit listed below.

```sh
cmake -S . -B build
cmake --build build -j
./build/pathtracer out.ppm
```

Run the CPU tests with `ctest --test-dir build`.

To check the GPU against the CPU on a tree, generate rays and run the brute-force tracer. It exits nonzero if any ray's hit differs:

```sh
./build/raygen scene.bvh scene.rays 100000
./build/bruteforce scene.bvh scene.rays
```

The build defaults to Release, because only Release numbers count as benchmarks. To build for one GPU only, pass `-DCMAKE_CUDA_ARCHITECTURES=75` (2070 Super) or `-DCMAKE_CUDA_ARCHITECTURES=120` (5070) to the first command.

## Profiling

Collect DRAM traffic with Nsight Compute and turn off its default cache flush, so the numbers reflect a warm cache:

```sh
ncu --cache-control none --metrics dram__bytes_read.sum -o report ./build/pathtracer
```

Nsight Compute locks clocks to base by default (`--clock-control base`); keep that setting for every benchmark run. Reports (`*.ncu-rep`, `*.nsys-rep`) and output images (`*.ppm`) are git-ignored.

## Benchmark platform

All official benchmark numbers are collected on a native (non-WSL) **Ubuntu 26.04.1 LTS** install, so differences come from the GPU and not the OS.

| Owner  | Hardware                     | Architecture               | Compute capability |
| ------ | ---------------------------- | -------------------------- | ------------------ |
| Diego  | CPU - Intel Core Ultra 7 355 | N/A                        | N/A                |
| Eugene | RTX 2070 Super               | Turing (RT cores gen 1)    | sm_75              |
| Harsha | RTX 5070                     | Blackwell (RT cores gen 4) | sm_120             |

Traversal runs as CUDA code on the shader cores. The RT cores are not used; their generation is listed only to identify the architecture.

Requirements on both machines:
- CUDA Toolkit 13.4 (`cuda-toolkit-13-4` from NVIDIA's `ubuntu2604` repo)
- NVIDIA driver R615 or newer; the RTX 5070 requires the open kernel modules (`-open` driver packages)
