# cache-aware-bvh

CMPE214 project by Eugene Lacatis, Harsha Krishnaswamy, and Diego Mateos. We test whether building a 4-wide BVH to minimize memory fetches, with a term for nodes that are likely already cached, reduces the data a GPU fetches per ray compared with the standard SAH collapse. Every tree runs through the same CUDA traversal kernel and is profiled with Nsight. A math proof backs the memory-traffic claim.

The proposal is in `docs/`, and the roles, schedule, and proof outline are in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## Benchmark platform

All official benchmark numbers are collected on a native (non-WSL) **Ubuntu 26.04.1 LTS** install, so differences come from the GPU and not the OS.

| Owner  | GPU                          | Architecture               | Compute capability |
| ------ | ---------------------------- | -------------------------- | ------------------ |
| Diego  | CPU - Intel Core Ultra 7 355 | N/A                        | N/A                |
| Eugene | RTX 2070 Super               | Turing (RT cores gen 1)    | sm_75              |
| Harsha | RTX 5070                     | Blackwell (RT cores gen 4) | sm_120             |

Requirements on both machines:
- CUDA Toolkit 13.4 (`cuda-toolkit-13-4` from NVIDIA's `ubuntu2604` repo)
- NVIDIA driver R615 or newer; the RTX 5070 requires the open kernel modules (`-open` driver packages)
