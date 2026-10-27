# Project plan

Six weeks, three people, Oct 27 to Dec 6, 2026. This plan carries out the proposal in `Project_Proposal_BVH_Memory_Aware_Collapse.pdf`: a 2-wide SAH BVH, a 4-wide tree from the standard SAH collapse, and a 4-wide tree from our memory-aware collapse, all traced by the same CUDA kernel. It adds the math proof the professor asked for, so the memory-efficiency claim rests on theorems and not only on our measurements.

`LEARNING.md` is from our earlier path tracer idea and is not part of this plan.

## What checking the citations changed

Read this section first. All of it came from reading the cited papers, and some of it changes the project.

1. Reading Haydel et al. in full confirms the proposal's positioning. Their paper (Proc. ACM CGIT 9(4), Article 48, HPG 2026) defines a memory-based SAH, MSAH = Σ A_n · c_n, where c_n is the size in bytes of the node block or primitive block. They build MSAH-optimal wide BVHs with the Ylitie dynamic program. MSAH has no cache-residency term, and their results come from the Arches cycle-level simulator configured like an RTX 2080, not from a real GPU. Section 3 of the proposal already names our differences: a fixed 4-wide layout, the cache term, and real-GPU measurements. The proof is a fourth. One small fix: the proposal's intro presents "least expected memory fetching" as new, and it should credit MSAH for that part.
2. Haydel's MSAH is the special case of our cost where nothing is cached (τ above every P(n)). So we get their method as a fourth tree for free, and comparing it with ours isolates what the cache term adds. That tree is now in the benchmark matrix.
3. T5 as first written is wrong under sector counting. Haydel et al. found that access granularity matters (their 64 B node blocks beat 128 B ones). With fp32 boxes, a binary node is about 64 B (2 sectors) and a 4-wide node about 112 B (4 sectors). Collapsing then lowers expected sectors only when SA(c1) + SA(c2) > SA(p), so it is a condition, not a guarantee. That makes a sharper statement of hypothesis 2, and we can measure it per node.
4. Sleator and Tarjan's LRU bound has an additive term. Theorem 6 of their paper reads F_LRU(s) ≤ n_LRU / (n_LRU − n_MIN + 1) · F_MIN(s) + n_MIN. T7 now includes the + n_MIN term, which only matters for cold misses.
5. Everett's theorem covers only the capacities that the τ sweep actually reaches. It says nothing about the capacities in between. Our gap bound for those (T6b) is our own work, and the report has to present it that way.
6. The Crofton constant depends on how the measure of lines is normalized (sources give π/4 or π/2 times surface area). Only ratios SA(B)/SA(A) appear in our proofs, so the constant cancels. T2 should say so explicitly.
7. Corrections to the proposal's references: Haydel et al. now has a venue (above). Barbier and Paulin is Computer Graphics Forum 2025, presented at HPG 2025. Haydel's 15.8% and 31.7% figures come from one scene (San Miguel), measured in a simulator against the Vaidyanathan et al. compressed 6-wide BVH. Chou et al. (both papers) are hardware proposals evaluated in Vulkan-Sim.

## Roles

Eugene (RTX 2070 Super) owns the GPU side: the tree and ray file reader, the 2-wide and 4-wide traversal kernels, per-ray counters (boxes visited, triangles tested), the node-access trace dump, and profiling on the 2070 Super. Reviews T5.

Harsha (RTX 5070) owns the CPU builder: scene loading, the binary SAH builder, the standard SAH collapse, the memory-aware collapse with leaf packing (the MSAH tree is the same code with τ set high), and a cost evaluator that prints SAH, MSAH, and E_mem for every tree. Also profiles on the 5070. Co-owns T6 and T6b, since the DP proof and the DP code have to match, and builds the scene generator for T8. Reviews T1.

Diego (no GPU needed) leads the proof (T1 to T8, written as a LaTeX appendix) and owns the tree validators (each triangle appears once, every box contains its children, CPU and GPU return the same triangle IDs), the LRU trace simulator, the model-fidelity analysis, and assembling the report. Reviews the cost evaluator against the T6 formula.

In week 1 we fix two interfaces that everyone codes against: the binary tree/ray file format and the node-access trace format.

## Schedule

| Week | Eugene | Harsha | Diego |
|---|---|---|---|
| 1 (Oct 27) | CUDA and Nsight sanity check on the 2070S; file format spec with node byte sizes | Same check on the 5070; OBJ loader (Dragon) | Write the formal cache model; update the proposal's related work for Haydel et al. |
| 2 (Nov 3) | Brute-force GPU tracer, 2-wide traversal | CPU reference tracer, binary SAH builder | Validators; draft T1 to T3 |
| 3 (Nov 10) | 4-wide traversal, counters | Standard SAH collapse; log SA(c1) + SA(c2) vs SA(p) per node for T5 | T4 and T5; trace format and LRU simulator. Checkpoint: 2-wide vs 4-wide result |
| 4 (Nov 17) | Trace dump; Sponza and Bistro loading | Memory-aware DP (and MSAH as τ high), leaf packing, cost evaluator | T6, T6b, T7 written; confirm simulated LRU misses stay under the T7 ceiling. Cross-review: each person checks one theorem |
| 5 (Nov 24) | Benchmark matrix on the 2070S | Benchmark matrix on the 5070 | Simulator runs, fidelity correlation, charts |
| 6 (Dec 1) | Report: method and GPU results | Report: builder and results | Final proof appendix, report integration, slides |

The week 3 checkpoint matters because the 2-wide vs 4-wide comparison is already a result if the memory-aware work runs late.

## The proof

Nothing about real GPU caches can be proven without a formal model of the cache. So the proof has three layers. We prove theorems inside a standard, explicit model. We show that the proposal's two heuristics (surface area predicts visits, nodes near the top stay cached) follow from that model as theorems. Then we test by experiment how well the model predicts the hardware.

### Formal cache model

Diego writes these definitions in week 1.

```
Items:      32 B sectors holding BVH nodes and triangles, s = 1..M
Capacity:   C sectors = cudaDeviceProp.l2CacheSize / 32 on each GPU
            (L1 per SM handled separately)
References: Independent Reference Model (IRM). Each access is independent;
            sector s is hit with probability p_s ∝ Σ P(n) over nodes n in s,
            where P(n) = SA(n) / SA(root)
Policy:     A0 (keep the C most probable sectors) for the theorems;
            LRU for T7 and the simulator
Cost:       expected DRAM sectors fetched per ray
```

Assumptions we state up front: rays are isotropic uniform random lines (the SAH assumption); a ray visits every node whose box it crosses, which gives an upper bound for closest-hit traversal; references follow IRM; the cache holds C sectors.

If the A0 predictions turn out too optimistic compared with the simulator, we use the Che approximation (Che, Tung, Wang 2002) for closed-form LRU hit rates under IRM.

### Theorems

| # | Statement | Type | Basis |
|---|---|---|---|
| T1 | A 4-wide tree over n leaves has at most ⌈(n−1)/3⌉ internal nodes, vs n−1 for a binary tree. Bound on bytes per triangle, including leaf packing | Counting, no model needed | Tree combinatorics |
| T1b | The memory-aware 4-wide tree is never larger than the SAH 4-wide tree at equal E_mem: leaf-packing bytes bound plus a DP tie-break toward fewer sectors | Lemma | Counting |
| T2 | P(ray hits B given it hits A) = SA(B)/SA(A) for convex B ⊆ A. The Crofton normalization constant cancels in the ratio | Theorem | Cauchy–Crofton formula (Santaló); MacDonald and Booth 1990 |
| T3 | If a child box lies inside its parent, P(child) ≤ P(parent) | Lemma | T2, and surface area is monotone for convex sets |
| T4 | Under IRM the miss-minimizing cache holds the C most probable sectors (policy A0). By T3 that set is a subtree containing the root, {n : P(n) ≥ τ}. This proves "nodes near the top stay cached" | Theorem | Aho, Denning, Ullman 1971 |
| T5 | With a 2-sector binary node and a 4-sector 4-wide node, collapsing a parent with both internal children lowers expected sectors fetched if and only if SA(c1) + SA(c2) > SA(p). The general form for other node sizes is s4 · P(p) < s2 · (P(p) + P(c1) + P(c2)) | Lemma | T2, node byte sizes from the file format |
| T6 | Main theorem. For any threshold τ, the bottom-up DP over the binary tree returns the 4-wide collapse that exactly minimizes E_mem = Σ P(n) · sectors(n) · [P(n) < τ]. Corollary: E_mem(memory-aware) ≤ E_mem(SAH collapse) and ≤ E_mem(MSAH collapse) on the same binary tree, because both are candidates. With τ above every P(n), E_mem equals Haydel's MSAH up to units | Theorem (optimal substructure) | DP from Ylitie et al. 2017, as also used by Haydel et al. 2026 |
| T6b | A fixed cache size C is not the same as a fixed τ. Choosing τ is a Lagrangian relaxation of the capacity constraint. It is exact at every capacity the τ sweep reaches (Everett's theorem). Between those capacities we prove our own gap bound | Theorem | Everett 1963, plus our bound |
| T7 | For any access trace, with no IRM assumption: F_LRU ≤ k/(k−h+1) · F_MIN + h, where LRU has k sectors and Belady's MIN has h. MIN is optimal on every trace, so F_MIN ≤ F_A0. With k = 2h, LRU misses on the memory-aware tree are below 2 × its A0 prediction at C/2, plus C/2 cold misses | Theorem (competitive analysis) | Sleator and Tarjan 1985, Theorem 6; Belady 1966 |
| T8 | A constructed family of scenes where E_mem(SAH collapse) / E_mem(memory-aware) is at least a constant above 1, so the T6 inequality is strict | Construction | T2, T6 |

T7 bounds the memory-aware tree's misses under real LRU from above. It can't show that the memory-aware tree beats the SAH tree under LRU, because the SAH tree's LRU misses have no general lower bound. That comparison stays empirical.

### Limits we state in the report

- The T6 corollary compares against the SAH and MSAH collapses of the same binary tree, not against every possible tree.
- The theorems compare upper bounds with no early ray termination. Closest-hit cost is measured, not proven.
- Uniform random lines fit scattered bounce rays. Coherent camera rays break the IRM independence assumption.
- The model has one cache level. The GPU has L1 and L2 with 32 B sectors, so unit costs count sectors and the simulator checks L1 and L2 separately.
- Aila, Karras, and Laine (HPG 2013) showed that SAH only partly predicts real performance. Haydel et al. showed that MSAH tracks performance in a simulator. Our hardware test extends that check to real GPUs.

A0 is optimal among policies that know the reference probabilities but not the future access order. Belady's offline MIN is optimal on every trace, so MIN ≤ A0, which is the direction T7 uses.

### Testing the model against hardware

1. Trace simulator: replay the GPU node-access traces through LRU at the L1 and L2 sizes of both GPUs, and compare misses with the model's prediction.
2. Nsight Compute: measure `dram__bytes` per ray.

We report the rank correlation between predicted, simulated, and measured bytes per ray across all trees, scenes, and ray types. T6 holds either way. The correlation shows how much of it carries over to the GPU.

## Benchmark matrix

- Scenes: Stanford Dragon (debugging), Sponza, Bistro, plus scaled copies for size
- Rays: camera rays and bounce rays
- Trees: 2-wide SAH, 4-wide SAH collapse, 4-wide MSAH collapse (Haydel's cost, τ high), 4-wide memory-aware
- GPUs: RTX 2070 Super, RTX 5070
- Measurements: rays per second (CUDA events), DRAM bytes per ray, cache hit rate, warp efficiency (Nsight Compute), boxes visited and triangles tested (counters), tree bytes per triangle (builder), predicted SAH, MSAH, and E_mem (cost evaluator)

## How the hypotheses map to the plan

| Slide 8 hypothesis | Covered by |
|---|---|
| 1. A box tree beats brute force on large scenes | Week 2 brute-force vs 2-wide timing |
| 2. 4-wide cuts visits more than it cuts data fetched | T1, T5 condition; week 3 per-node SA log and counters |
| 3. Memory-aware fetches less, mostly for bounce rays on large scenes | T6, T8; week 5 matrix |
| 4. Memory-aware may visit more boxes yet fetch less | Week 5 counters vs DRAM bytes |
| New: the cache term helps beyond MSAH | Memory-aware vs MSAH tree in the week 5 matrix |

## References

Checked against the papers or their publisher pages on Oct 26, 2026.

1. Aho, Denning, Ullman. Principles of Optimal Page Replacement. JACM 18(1):80–93, 1971. doi:10.1145/321623.321632
2. Sleator, Tarjan. Amortized Efficiency of List Update and Paging Rules. CACM 28(2):202–208, 1985. Theorem 6 is the LRU vs MIN bound.
3. Belady. A Study of Replacement Algorithms for Virtual Storage Computers. IBM Systems Journal 5:78–101, 1966.
4. Everett. Generalized Lagrange Multiplier Method for Solving Problems of Optimum Allocation of Resources. Operations Research 11(3):399–417, 1963. doi:10.1287/opre.11.3.399
5. Che, Tung, Wang. Hierarchical Web Caching Systems: Modeling, Design and Experimental Results. IEEE JSAC 20(7):1305–1314, 2002.
6. Santaló. Integral Geometry and Geometric Probability. Cambridge University Press (2nd ed. 2004).
7. MacDonald, Booth. Heuristics for Ray Tracing Using Space Subdivision. The Visual Computer 6(3):153–166, 1990. doi:10.1007/BF01911006
8. Ylitie, Karras, Laine. Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs. HPG 2017. doi:10.1145/3105762.3105773
9. Haydel, Kensler, Brunvand, Yuksel. Memory-Efficient Bounding Volume Hierarchies with Merged Nodes for Hardware Ray Tracing. Proc. ACM CGIT 9(4), Article 48, HPG 2026. doi:10.1145/3820018
10. Aila, Karras, Laine. On Quality Metrics of Bounding Volume Hierarchies. HPG 2013. doi:10.1145/2492045.2492056
11. Yoon, Manocha. Cache-Efficient Layouts of Bounding Volume Hierarchies. Computer Graphics Forum 25(3):507–516, 2006.
12. Vaidyanathan, Woop, Benthin. Wide BVH Traversal with a Short Stack. HPG 2019. doi:10.2312/hpg.20191190
13. Barbier, Paulin. Fused Collapsing for Wide BVH Construction. Computer Graphics Forum, 2025. doi:10.1111/cgf.70213
14. Chou, Nowicki, Aamodt. Treelet Prefetching for Ray Tracing. MICRO 2023, 742–755.
15. Chou, Aamodt. Treelet Accelerated Ray Tracing on GPUs. ASPLOS 2025, 1334–1347.
