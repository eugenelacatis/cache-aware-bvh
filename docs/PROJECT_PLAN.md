# Project plan

Six weeks, three people, Oct 27 to Dec 6, 2026. This plan carries out the proposal in `Project_Proposal_BVH_Memory_Aware_Collapse.pdf`: a 2-wide SAH BVH, a 4-wide tree from the standard SAH collapse, and a 4-wide tree from our memory-aware collapse, all traced by the same CUDA kernel. It adds the math proof the professor asked for. The theorems hold inside a stated cache model, and our measurements test how much of that model carries over to real GPUs.

`LEARNING.md` is from our earlier path tracer idea and is not part of this plan.

## What checking the citations changed

Read this section first. All of it came from reading the cited papers, and some of it changes the project.

1. Reading Haydel et al. in full confirms the proposal's positioning. Their paper (Proc. ACM CGIT 9(4), Article 48, HPG 2026) defines a memory-based SAH, MSAH = Σ A_n · c_n, where c_n is the size in bytes of the node block or primitive block. They build MSAH-optimal wide BVHs with the Ylitie dynamic program. MSAH has no cache-residency term, and their results come from the Arches cycle-level simulator configured like an RTX 2080, not from a real GPU. Section 3 of the proposal already names our differences: a fixed 4-wide layout, the cache term, and real-GPU measurements. The proof is a fourth. One small fix: the proposal's intro presents "least expected memory fetching" as new, and it should credit MSAH for that part.
2. Haydel's MSAH is the special case of our cost where nothing is cached (τ above every P(n)). So the same DP builds an MSAH-cost tree, and comparing it with ours isolates what the cache term adds. That tree is now in the benchmark matrix. It is not Haydel's method. They merge nodes into blocks of different sizes, while every internal node in our layout is a fixed 4-wide node of 4 sectors. In that layout MSAH reduces to SAH with a different ratio of node cost to triangle cost. The report calls it the MSAH-cost collapse.
3. T5 as first written is wrong under sector counting. Haydel et al. found that access granularity matters (their 64 B node blocks beat 128 B ones). With fp32 boxes, a binary node is about 64 B (2 sectors) and a 4-wide node about 112 B (4 sectors). Collapsing then lowers expected sectors only when SA(c1) + SA(c2) > SA(p), so it is a condition, not a guarantee. That makes a sharper statement of hypothesis 2, and we can measure it per node.
4. Sleator and Tarjan's LRU bound has an additive term. Theorem 6 of their paper reads F_LRU(s) ≤ n_LRU / (n_LRU − n_MIN + 1) · F_MIN(s) + n_MIN. T7 now includes the + n_MIN term, which only matters for cold misses.
5. Everett's theorem covers only the capacities that the τ sweep actually reaches. It says nothing about the capacities in between. We report those from the sweep's Pareto curve and don't claim a bound for them.
6. The Crofton constant depends on how the measure of lines is normalized (sources give π/4 or π/2 times surface area). Only ratios SA(B)/SA(A) appear in our proofs, so the constant cancels. T2 should say so explicitly.
7. Corrections to the proposal's references: Haydel et al. now has a venue (above). Barbier and Paulin is Computer Graphics Forum 2025, presented at HPG 2025. Haydel's 15.8% and 31.7% figures come from one scene (San Miguel), measured in a simulator against the Vaidyanathan et al. compressed 6-wide BVH. Chou et al. (both papers) are hardware proposals evaluated in Vulkan-Sim.
8. The proposal's related work has to separate our cache term from treelets and cache-friendly layouts, since both also aim to keep hot nodes on chip. Aila and Karras (2010) group nodes into treelets sized to fit in cache, Chou et al. group rays by treelet and prefetch treelets in hardware, and Yoon and Manocha (2006) reorder nodes in memory for locality. All three change where or when nodes are stored and leave the tree's shape alone. Our term changes the shape: the collapse is told which nodes are likely cached and builds the tree around them. The two can be combined, because a memory-aware tree can still be laid out or prefetched by treelet.

## What the plan review changed

A review of the first version of this plan (PR #1) found errors in the proof outline and gaps in the experiment design. The sections below already include these changes.

1. T1's bound pointed the wrong way. Every tree with n leaves satisfies n − 1 = Σ (k_i − 1) over its internal nodes, where k_i is the node's number of children. With k_i ≤ 4 that gives at least ⌈(n−1)/3⌉ internal nodes, and exactly that many only when every node is full. The Ylitie DP makes 2- and 3-child nodes, and a half-empty 4-slot node spends 112 B on 2 boxes where a binary node spends 56 B. T1 now states the lower bound and writes bytes per triangle in terms of the histogram of node child counts, which the builder logs.
2. The old T6 objective, Σ P(n)·s(n)·[P(n) < τ], gave cached nodes zero cost. The DP could then add cached nodes without counting the cache space they use, so Everett's theorem didn't apply to it. The new objective is Σ s(n)·min(P(n), τ). It is the Lagrangian relaxation of the capacity constraint, with τ as the multiplier, so every cached sector costs τ. The DP then prefers full nodes above τ, which fits more probability mass into the cache.
3. The Independent Reference Model fails even for isotropic random rays, because a ray visits a child only after its parent. Aho, Denning, and Ullman's result needs IRM. The model is now an independent-ray model, under which T4' has a short proof of our own.
4. Surface-area probabilities (Crofton, T2) describe lines that come from outside the scene. Bounce rays start on surfaces inside it, and camera rays all start at one point. T6 is now stated for any visit probabilities, and the builder runs the DP with surface-area P and with P measured from a low-resolution trace pass.
5. The SAH baseline uses the SAH-optimal Ylitie DP, not a greedy collapse. Otherwise part of any gain would come from DP vs. greedy and not from the cache term.
6. T6's corollary holds by construction, so the claim that the GPU fetches less rests on the experiments. The experiment design changed to test that link: a τ sweep per scene, comparisons within each scene, a test ladder that removes one model assumption per step, working-set size over L2 size as the scene axis, and a week 3 estimate of the predicted gain.
7. T2, T6b, and T7 become cited remarks so Diego has time for the proofs that carry the argument: T1, T1b, T3, T4', T5, T6, and T8.

## Roles

Eugene (RTX 2070 Super) owns the GPU side: the tree and ray file reader, the 2-wide and 4-wide traversal kernels (closest-hit, plus an all-hit mode that visits every node whose box the ray crosses, for the model test), per-ray counters (boxes visited, triangles tested), the node-access trace dump, and profiling on the 2070 Super. Reviews T5.

Harsha (RTX 5070) owns the CPU builder: scene loading, the binary SAH builder, the SAH collapse (Ylitie DP), the memory-aware collapse with leaf packing (the MSAH-cost tree is the same code with τ set high), and a cost evaluator that prints SAH, MSAH, and E_mem for every tree. The DP reads P(n) per node, so it runs with surface-area P or measured P. Also profiles on the 5070. Co-owns T6, since the DP proof and the DP code have to match, and builds the scene generator for T8. Reviews T1.

Diego (no GPU needed) leads the proof, written as a LaTeX appendix: full proofs of T1, T1b, T3, T4', T5, T6, and T8, and cited remarks for T2, T6b, and T7. Also owns the tree validators (each triangle appears once, every box contains its children, CPU and GPU return the same triangle IDs), the LRU trace simulator, the model-fidelity analysis, and assembling the report. Reviews the cost evaluator against the T6 formula.

In week 1 we fix two interfaces that everyone codes against: the binary tree/ray file format (with a P value per node) and the node-access trace format.

## Schedule

| Week | Eugene | Harsha | Diego |
|---|---|---|---|
| 1 (Oct 27) | CUDA and Nsight sanity check on the 2070S; file format spec with node byte sizes, nodes padded to 128 B | Same check on the 5070, including `l2CacheSize` and the L2 fetch granularity; OBJ loader (Dragon) | Write the formal cache model (independent-ray model); update the proposal's related work for Haydel et al. and for treelets and layouts (item 8) |
| 2 (Nov 3) | Brute-force GPU tracer, 2-wide traversal | CPU reference tracer, binary SAH builder | Validators; draft T1, T3, T4' |
| 3 (Nov 10) | 4-wide traversal, counters, all-hit mode | SAH collapse (Ylitie DP); log SA(c1) + SA(c2) vs SA(p) per node for T5; CPU prototype of the memory-aware DP that prints the predicted E_mem gap per scene | T1b and T5; trace format and LRU simulator. Checkpoint: 2-wide vs 4-wide result and predicted E_mem gap |
| 4 (Nov 17) | Trace dump; Sponza and Bistro loading; scaled scene copies as distinct geometry | Memory-aware DP in the builder (and MSAH-cost as τ high), leaf packing, cost evaluator, measured P from a low-resolution trace | T6 and T8 written; T6b and T7 remarks; confirm simulated LRU misses stay under the T7 ceiling. Cross-review: each person checks one theorem |
| 5 (Nov 24) | Benchmark matrix and τ sweep on the 2070S | Benchmark matrix and τ sweep on the 5070 | Simulator runs, within-scene fidelity analysis, charts |
| 6 (Dec 1) | Report: method and GPU results | Report: builder and results | Final proof appendix, report integration, slides |

The week 3 checkpoint matters for two reasons. The 2-wide vs 4-wide comparison is already a result if the memory-aware work runs late. And the cache term only changes the tree near the τ boundary, so its predicted gain on real scenes could be within measurement noise. If the prototype shows that, we learn it in week 3 and shift weight to the 2-wide vs 4-wide result and T5.

Stretch goal, only if the week 5 matrix finishes early: add spatial splits to the binary builder, with split leaves pointing at a shared primitive block so no triangle is duplicated, then rerun all collapses on the new tree. Haydel et al. list this as future work. It needs one new lemma, because T1's byte counting assumes each triangle sits in exactly one leaf.

## The proof

Nothing about real GPU caches can be proven without a formal model of the cache. So the proof has three layers. We prove theorems inside a standard, explicit model. We show that the proposal's two heuristics (surface area predicts visits, nodes near the top stay cached) follow from that model. Then we test by experiment how well the model predicts the hardware.

The main theorem (T6) says our DP exactly minimizes the model's cost. So E_mem(ours) ≤ E_mem(other trees) holds by construction, and it can't show by itself that a GPU fetches less. That claim depends on E_mem predicting DRAM traffic, which only the experiments can test.

### Formal cache model

Diego writes these definitions in week 1.

```
Items:      32 B sectors holding BVH nodes and triangles, s = 1..M
Capacity:   C sectors = cudaDeviceProp.l2CacheSize / 32 on each GPU
            (L1 per SM handled separately)
Visits:     P(n) = probability that a ray visits node n. T6 holds for any P.
            Default P(n) = SA(n) / SA(root) (T2); also P measured from a trace.
            P(s) = probability that a ray visits some node stored in sector s
Rays:       Independent-ray model. Each ray draws its set V of visited
            sectors independently from the same distribution and touches
            each sector at most once
Policy:     A0 (keep the C sectors with the largest P(s)) for the theorems;
            LRU for T7 and the simulator
Cost:       expected DRAM sectors fetched per ray
```

Assumptions we state up front: rays are independent of each other; a ray visits every node whose box it crosses (the all-hit kernel), which gives an upper bound for closest-hit traversal; the cache holds C sectors. The default surface-area P also assumes rays are isotropic uniform random lines (the SAH assumption).

The memory-aware collapse minimizes

```
E_mem(T, τ) = Σ_n s(n) · min(P(n), τ)
            = Σ_{P(n) < τ} P(n) · s(n)  +  τ · Σ_{P(n) ≥ τ} s(n)
```

over the 4-wide collapses T of one binary tree, where s(n) is the number of sectors of node n. The first sum is the expected number of sectors fetched from DRAM per ray. The second is τ times the cache space the cached nodes use. This is the Lagrangian relaxation of "minimize expected misses subject to Σ cached s(n) ≤ C", with τ as the multiplier: for a fixed tree, the inner choice of which nodes to cache picks exactly the nodes with P(n) > τ. With τ above every P(n), E_mem is MSAH counted in sectors.

If the A0 predictions turn out too optimistic compared with the simulator, we use the Che approximation (Che, Tung, Wang 2002) for closed-form LRU hit rates under IRM.

### Theorems

| # | Statement | Type | Basis |
|---|---|---|---|
| T1 | A 4-wide tree over n leaves has at least ⌈(n−1)/3⌉ internal nodes, with equality only when every internal node has 4 children, since n − 1 = Σ (k_i − 1). Bytes per triangle, written in terms of the histogram of child counts, including leaf packing and 128 B padding | Counting, no model needed | Tree combinatorics |
| T1b | The memory-aware 4-wide tree is never larger than the SAH 4-wide tree at equal E_mem: leaf-packing bytes bound plus a DP tie-break toward fewer sectors. The lexicographic cost (E_mem, sectors) still decomposes over nodes | Lemma | Counting |
| T2 | P(line hits B given it hits A) = SA(B)/SA(A) for convex B ⊆ A and isotropic uniform lines from outside A. The Crofton normalization constant cancels in the ratio. This justifies the default P for the random-lines test workload only: bounce rays start inside the scene and camera rays start at one point | Cited remark | Cauchy–Crofton formula (Santaló); MacDonald and Booth 1990 |
| T3 | A ray visits a child only after visiting its parent, so P(child) ≤ P(parent) for any visit probabilities, measured ones included | Lemma | Traversal order |
| T4' | Under the independent-ray model, no demand-fetch cache policy gets more expected hits per ray than A0, which keeps the C sectors with the largest P(s). Proof: the cache contents S when a ray starts depend only on earlier rays, so S is independent of the ray's visit set V. A sector loaded during the ray is not reused by that ray, and evicting during the ray only removes hits. So E[hits] ≤ E\|V ∩ S\| = Σ_{s∈S} P(s), which is at most the sum of the C largest P(s), and A0 reaches it. By T3 the top-C set is closed under taking parents (up to ties), so it is a subtree containing the root. This proves "nodes near the top stay cached" | Theorem (our proof) | Independence and linearity of expectation. Aho, Denning, Ullman 1971 is the IRM version |
| T5 | With a 2-sector binary node and a 4-sector 4-wide node, collapsing a parent with both internal children lowers expected sectors fetched if and only if SA(c1) + SA(c2) > SA(p). The general form for other node sizes is s4 · P(p) < s2 · (P(p) + P(c1) + P(c2)). On SAH trees we expect the condition to hold at most nodes (halving a cube gives children whose areas sum to 4/3 of the parent's), so the week 3 log looks for the nodes where it fails | Lemma | T2, node byte sizes from the file format |
| T6 | Main theorem. For any visit probabilities P and any threshold τ, the bottom-up DP over the binary tree returns the 4-wide collapse that exactly minimizes E_mem(T, τ) = Σ s(n) · min(P(n), τ) over all 4-wide collapses of that binary tree. Corollary: E_mem(memory-aware) ≤ E_mem(SAH collapse) and ≤ E_mem(MSAH-cost collapse) on the same binary tree at the same τ, because both are candidates. The corollary holds by construction; whether E_mem predicts DRAM traffic is tested by experiment. With τ above every P(n), E_mem equals Haydel's MSAH counted in sectors | Theorem (optimal substructure) | DP from Ylitie et al. 2017, as also used by Haydel et al. 2026 |
| T6b | A fixed cache size C is not the same as a fixed τ. A tree that minimizes E_mem(T, τ), and whose cached nodes use C(τ) sectors, is optimal for capacity C(τ) (Everett's theorem). Capacities between those the τ sweep reaches are reported from the sweep's Pareto curve, with no bound claimed | Cited remark | Everett 1963 |
| T7 | For any access trace, with no model assumption: F_LRU ≤ k/(k−h+1) · F_MIN + h, where LRU has k sectors and Belady's MIN has h. MIN is optimal on every trace, so F_MIN ≤ F_A0 on the same trace, where F_A0 counts A0's misses replayed on that trace (not the model's prediction). With k = 2h, LRU misses on the memory-aware tree are below 2 × F_A0 at C/2, plus C/2 cold misses. Assumes a fully associative cache, so it applies to the simulator, not to the GPU's set-associative L2 | Cited remark | Sleator and Tarjan 1985, Theorem 6; Belady 1966 |
| T8 | A constructed family of scenes where E_mem(SAH collapse) / E_mem(memory-aware) is at least a constant above 1, so the T6 inequality is strict | Construction | T2, T6 |

T7 bounds the memory-aware tree's misses under LRU from above. It can't show that the memory-aware tree beats the SAH tree under LRU, because the SAH tree's LRU misses have no general lower bound. That comparison stays empirical.

### Limits we state in the report

- The T6 corollary compares against the SAH and MSAH-cost collapses of the same binary tree, not against every possible tree, and it holds by construction.
- The theorems compare upper bounds with no early ray termination (the all-hit kernel). Closest-hit cost is measured, not proven.
- Surface-area P fits isotropic lines from outside the scene. Bounce rays start inside the scene and camera rays start at one point, so for them we also run the DP with measured P. Rays in the same warp that visit the same node share one fetch, which the per-ray model ignores; this matters most for coherent camera rays.
- The model has one cache level. The GPU has L1 and L2 with 32 B sectors, so unit costs count sectors and the simulator checks L1 and L2 separately. GPU L2 is set-associative with an undocumented replacement policy, so T7 applies to the simulator only.
- Fewer bytes per ray need not mean more rays per second. Software traversal is often limited by load latency rather than bandwidth, so we check Nsight stall reasons before linking the two. We trace in CUDA software and don't use the RT cores.
- Aila, Karras, and Laine (HPG 2013) showed that SAH only partly predicts real performance. Haydel et al. showed that MSAH tracks performance in a simulator. Our hardware test extends that check to real GPUs.

A0 is optimal in expectation under the independent-ray model (T4'). Belady's offline MIN is optimal on every trace, so MIN ≤ A0 on any trace, which is the direction T7 uses.

### Testing the model against hardware

Each step below drops one model assumption, so each loss of fidelity has a single cause:

1. Isotropic random lines with the all-hit kernel. This workload matches the model, so predicted, simulated, and measured bytes should agree closely.
2. The same rays replayed through the LRU trace simulator at the L1 and L2 sizes of both GPUs. This measures the error from assuming A0 instead of LRU.
3. Bounce rays with closest-hit traversal and early termination.
4. Camera rays, which are coherent and share fetches within a warp.

Measurements come from Nsight Compute (`dram__bytes` per ray) with `--cache-control none`, so caches aren't flushed before each profiled kernel, plus locked clocks and repeated runs reported with confidence intervals. In week 1 we check the effective fetch granularity on each GPU (`dram__sectors` against bytes requested, and `cudaLimitMaxL2FetchGranularity`), because the 32 B unit cost is an assumption until measured.

Pooled across scenes, scene size would dominate any correlation, and 4 trees per scene are too few points. So we sweep τ per scene, plot predicted against measured bytes per ray, and report the rank correlation of the differences between trees within each scene. T6 holds either way. The correlation shows how much of it carries over to the GPU.

## Benchmark matrix

- Scenes: Stanford Dragon (debugging), Sponza, Bistro, plus scaled copies made of distinct geometry, not instances. Results are plotted against working-set size divided by L2 size. The 2070 Super has 4 MB of L2 and the 5070 about 48 MB (confirmed in week 1), so a scene that stresses one GPU's L2 may fit entirely in the other's
- Rays: isotropic random lines (model test), camera rays, and bounce rays
- Trees: 2-wide SAH, 4-wide SAH collapse (Ylitie DP), 4-wide MSAH-cost collapse (τ high), 4-wide memory-aware over a sweep of τ values, each with surface-area P and with measured P
- GPUs: RTX 2070 Super, RTX 5070
- Measurements: rays per second (CUDA events), DRAM bytes per ray, cache hit rate, warp efficiency and stall reasons (Nsight Compute), boxes visited and triangles tested (counters), tree bytes per triangle and the histogram of node child counts (builder), predicted SAH, MSAH, and E_mem (cost evaluator)

## How the hypotheses map to the plan

| Slide 8 hypothesis | Covered by |
|---|---|
| 1. A box tree beats brute force on large scenes | Week 2 brute-force vs 2-wide timing |
| 2. 4-wide cuts visits more than it cuts data fetched | T1, T5 condition; week 3 per-node SA log and counters |
| 3. Memory-aware fetches less, mostly for bounce rays on large scenes | T6, T8; week 3 predicted gap; week 5 matrix and τ sweep |
| 4. Memory-aware may visit more boxes yet fetch less | Week 5 counters vs DRAM bytes |
| New: the cache term helps beyond MSAH | Memory-aware vs MSAH-cost tree in the week 5 matrix |

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
16. Aila, Karras. Architecture Considerations for Tracing Incoherent Rays. HPG 2010, 113–122.

### To check before citing

Suggested in the plan review and not yet checked against the papers. Diego verifies these in week 1 along with the related-work update.

- Wodniok, Schulz, Widmer, Goesele. Analysis of Cache Behavior and Performance of Different BVH Memory Layouts for Tracing Incoherent Rays. EGPGV 2013.
- Liktor, Vaidyanathan. Bandwidth-Efficient BVH Layout for Incremental Hardware Traversal. HPG 2016.
- Bittner, Havran. RDH: Ray Distribution Heuristics for Construction of Spatial Data Structures. SCCG 2009. The idea behind measured P.
- Fabianowski, Fowler, Dingliana. A Cost Metric for Scene-Interior Ray Origins. Eurographics 2009 short paper.
