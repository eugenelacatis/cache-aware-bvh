# File formats

Version 1. The builder writes tree files, `raygen` writes ray files, the GPU kernels read both, and the trace dump writes node-access traces for the LRU simulator. The C++ structs in `src/bvh/bvh_format.h` match these layouts byte for byte, and `static_assert`s check the sizes. `src/bvh/bvh_io.cpp` reads, writes, and validates tree and ray files.

All files are little-endian. Floats are IEEE 754 binary32. Offsets are in bytes from the start of the file.

## Tree file (`.bvh`)

```
offset 0     header                128 B
offset 128   node section          nodeCount × nodeBytes
             padding to a multiple of 128
triOffset    triangle section      triSectorCount × 32 B
```

The GPU loads the whole file with one `cudaMemcpy`. `cudaMalloc` returns 256 B aligned memory, so a sector in the file (byte offset / 32) is also a sector on the device. Both sections start on 128 B boundaries, so every 4-wide node fills exactly 4 sectors and every binary node exactly 2.

### Header (128 B)

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `magic` | ASCII `CABVHTRE` |
| 8 | 4 | `version` | 1 |
| 12 | 4 | `width` | 2 or 4 |
| 16 | 4 | `nodeBytes` | 64 for width 2, 128 for width 4 |
| 20 | 4 | `nodeCount` | at least 1; node 0 is the root |
| 24 | 4 | `triCount` | triangles in the scene |
| 28 | 4 | `triSectorCount` | length of the triangle section in 32 B sectors |
| 32 | 8 | `nodeOffset` | 128 in version 1 |
| 40 | 8 | `triOffset` | multiple of 128, after the last node |
| 48 | 24 | `sceneBox` | root's box: `lo[3]`, `hi[3]` |
| 72 | 4 | `rootP` | P of the root; 1.0 for surface-area P |
| 76 | 4 | `pSource` | 0 surface area (SA(n)/SA(root)), 1 measured from a trace |
| 80 | 4 | `treeKind` | 0 binary SAH, 1 SAH collapse, 2 MSAH-cost collapse, 3 memory-aware, 255 test |
| 84 | 4 | `tau` | τ used by the memory-aware DP; NaN for other kinds |
| 88 | 40 | `reserved` | zero |

### Nodes

A node stores the boxes, child references, and P values of its children. The root's own box is `sceneBox` and its P is `rootP`, so every node and every leaf has a P stored somewhere.

Binary node, 64 B (2 sectors):

| Offset | Size | Field |
|---|---|---|
| 0 | 48 | `box[2]`, each `lo[3]`, `hi[3]` |
| 48 | 8 | `child[2]`, u32 references |
| 56 | 8 | `p[2]`, P of each child |

4-wide node, 128 B (4 sectors):

| Offset | Size | Field |
|---|---|---|
| 0 | 96 | `box[4]` |
| 96 | 16 | `child[4]` |
| 112 | 16 | `p[4]` |

Traversal reads the first 56 B or 112 B. The P values sit in what would otherwise be padding, so they don't add sectors. These sizes match the 2-sector and 4-sector nodes that T5 assumes.

Used child slots come first. Empty slots follow and hold `0x80000000`. Every node has at least one child.

### Child references (u32)

```
bit 31 = 0   internal child   bits 0..30  node index (never 0, the root)
bit 31 = 1   leaf             bits 27..30 triangle count, 1..15
                              bits 0..26  first sector of the leaf block,
                                          relative to triOffset
0x80000000   empty slot (a leaf with 0 triangles)
```

Each node has exactly one parent, and every node is reachable from the root. A scene small enough for one leaf still has a root node, with that leaf as its only child.

### Triangle section

Each triangle is 48 B:

| Offset | Size | Field |
|---|---|---|
| 0 | 36 | `v0[3]`, `v1[3]`, `v2[3]` |
| 36 | 4 | `id`, index of the triangle in the source mesh |
| 40 | 8 | padding, zero |

A leaf's triangles are stored back to back in a leaf block, and each block starts on a 32 B sector. A leaf with k triangles therefore occupies ⌈48k / 32⌉ = ⌈3k / 2⌉ sectors no matter where the builder puts it, so the DP can use s(leaf) without knowing the layout. The bytes between the end of one block and the next sector are zero.

### What `validateTree` checks

Magic, version, width and node size agree, sections are aligned and inside the file, child references point inside the node array or the triangle section, empty slots come last, every node has one parent and is reachable from the root, and the tree is shallow enough for the traversal stack. The stack holds 64 entries, so a root-to-leaf path may have at most 63 binary nodes or 21 4-wide nodes. Checking that each triangle appears once and that boxes contain their children is left to Diego's validators.

## Ray file (`.rays`)

A 32 B header followed by `rayCount` rays of 32 B each.

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `magic` | ASCII `CABVHRAY` |
| 8 | 4 | `version` | 1 |
| 12 | 4 | `rayCount` | |
| 16 | 4 | `rayKind` | 0 isotropic lines, 1 camera, 2 bounce |
| 20 | 4 | reserved | zero |
| 24 | 8 | `seed` | RNG seed used to generate the rays |

Ray, 32 B: `org[3]`, `tmin`, `dir[3]`, `tmax`. Directions are unit length. A ray hits a triangle at distance t only when tmin < t < tmax.

`raygen <tree.bvh> <out.rays> <count> [seed]` writes isotropic lines for the tree's scene box. It picks a direction uniform on the sphere and an offset uniform on the disk of radius R perpendicular to it, where R is the radius of the scene box's bounding sphere. That gives lines that are uniform in the Crofton measure among those crossing the sphere. Each ray starts where its line enters the sphere and has tmin = 0, tmax = 2R. Lines that miss the scene box are kept. Among the lines that hit a box A, the fraction that also hit a box B inside A is SA(B)/SA(A). The test suite checks that ratio to within 0.005.

## Node-access trace (`.trace`)

The GPU trace dump writes this file, and the LRU simulator reads it. Until the GPU dump exists, `tracedump <tree.bvh> <in.rays> <out.trace> [closest|all]` writes the same file from the CPU traversal. A 32 B header is followed by `eventCount` events of 12 B each.

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `magic` | ASCII `CABVHTRC` |
| 8 | 4 | `version` | 1 |
| 12 | 4 | `mode` | 0 closest-hit, 1 all-hit |
| 16 | 4 | `rayCount` | |
| 20 | 4 | `width` | width of the traced tree |
| 24 | 8 | `eventCount` | |

Event, 12 B:

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 4 | `ray` | index into the ray file |
| 4 | 4 | `firstSector` | absolute sector in the tree file (byte offset / 32) |
| 8 | 2 | `sectorCount` | 2 or 4 for a node, ⌈3k/2⌉ for a leaf block |
| 10 | 1 | `kind` | 0 node, 1 triangle block |
| 11 | 1 | reserved | zero |

One event is one fetch of a whole node or a whole leaf block. Events are sorted by ray and, within a ray, listed in visit order. The GPU does not record the interleaving between rays, so the simulator replays rays in index order, which matches the independent-ray model. Warp w holds rays 32w to 32w + 31, so a simulator that models shared fetches inside a warp can group events by `ray / 32`.

A node's index is `(firstSector × 32 − nodeOffset) / nodeBytes`. Use it to look up P(n) when comparing simulated and predicted misses.

## Open questions

- The leaf limit is 15 triangles because the count has 4 bits. Tell Eugene if the builder needs larger leaves.
- Leaf blocks are padded to start on a sector, so 1-triangle leaves waste 16 B each. Packing leaves together would save that space but would make s(leaf) depend on the layout. That needs Harsha's input before we change it.
