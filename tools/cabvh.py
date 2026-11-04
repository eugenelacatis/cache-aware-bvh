"""Read .bvh, .rays and .trace files (docs/FORMATS.md) into numpy arrays.

    import cabvh
    tree = cabvh.read_tree("scene.bvh")
    rays = cabvh.read_rays("scene.rays")
    trace = cabvh.read_trace("scene.trace")
    nodes = trace.node_indices(tree)   # node index per event, -1 for leaf blocks
"""

from dataclasses import dataclass

import numpy as np

SECTOR = 32
LEAF_BIT = 0x80000000
EMPTY_REF = LEAF_BIT
MAX_LEAF_SECTOR = (1 << 27) - 1

NODE = 0
TRI_BLOCK = 1

TREE_HEADER = np.dtype([
    ("magic", "S8"), ("version", "<u4"), ("width", "<u4"), ("nodeBytes", "<u4"),
    ("nodeCount", "<u4"), ("triCount", "<u4"), ("triSectorCount", "<u4"),
    ("nodeOffset", "<u8"), ("triOffset", "<u8"), ("sceneBox", "<f4", (2, 3)),
    ("rootP", "<f4"), ("pSource", "<u4"), ("treeKind", "<u4"), ("tau", "<f4"),
    ("reserved", "u1", (40,)),
])

# box[i] is [lo, hi] of child i.
NODE2 = np.dtype([("box", "<f4", (2, 2, 3)), ("child", "<u4", (2,)), ("p", "<f4", (2,))])
NODE4 = np.dtype([("box", "<f4", (4, 2, 3)), ("child", "<u4", (4,)), ("p", "<f4", (4,))])

TRIANGLE = np.dtype([("v", "<f4", (3, 3)), ("id", "<u4"), ("pad", "<u4", (2,))])

RAY_HEADER = np.dtype([
    ("magic", "S8"), ("version", "<u4"), ("rayCount", "<u4"), ("rayKind", "<u4"),
    ("reserved0", "<u4"), ("seed", "<u8"),
])
RAY = np.dtype([("org", "<f4", (3,)), ("tmin", "<f4"), ("dir", "<f4", (3,)), ("tmax", "<f4")])

TRACE_HEADER = np.dtype([
    ("magic", "S8"), ("version", "<u4"), ("mode", "<u4"), ("rayCount", "<u4"),
    ("width", "<u4"), ("eventCount", "<u8"),
])
EVENT = np.dtype([
    ("ray", "<u4"), ("firstSector", "<u4"), ("sectorCount", "<u2"), ("kind", "u1"), ("reserved", "u1"),
])

assert TREE_HEADER.itemsize == 128 and NODE2.itemsize == 64 and NODE4.itemsize == 128
assert TRIANGLE.itemsize == 48 and RAY_HEADER.itemsize == 32 and RAY.itemsize == 32
assert TRACE_HEADER.itemsize == 32 and EVENT.itemsize == 12


# ---- Child references (scalar or array) -------------------------------------

def is_empty(ref):
    return np.asarray(ref, dtype=np.uint32) == EMPTY_REF


def is_leaf(ref):
    return (np.asarray(ref, dtype=np.uint32) & LEAF_BIT) != 0


def leaf_count(ref):
    return (np.asarray(ref, dtype=np.uint32) >> 27) & 0xF


def leaf_sector(ref):
    """First sector of the leaf block, relative to triOffset."""
    return np.asarray(ref, dtype=np.uint32) & MAX_LEAF_SECTOR


def leaf_sectors(count):
    """Sectors a leaf with `count` triangles occupies: ceil(3k / 2)."""
    return (3 * np.asarray(count) + 1) // 2


# ---- Files ------------------------------------------------------------------

def _header(buf, dtype, magic, path):
    if len(buf) < dtype.itemsize:
        raise ValueError(f"{path}: file is shorter than its {dtype.itemsize} B header")
    h = np.frombuffer(buf, dtype, count=1)[0]
    if h["magic"] != magic:
        raise ValueError(f"{path}: magic is {h['magic']!r}, expected {magic!r}")
    if h["version"] != 1:
        raise ValueError(f"{path}: version {h['version']}, expected 1")
    return h


@dataclass
class Tree:
    header: np.void
    nodes: np.ndarray        # NODE2 or NODE4 records, node 0 is the root
    tri_section: np.ndarray  # raw bytes of the triangle section

    @property
    def width(self):
        return int(self.header["width"])

    def leaf_triangles(self, ref):
        """TRIANGLE records of one leaf block."""
        start = int(leaf_sector(ref)) * SECTOR
        return np.frombuffer(self.tri_section, TRIANGLE, count=int(leaf_count(ref)), offset=start)

    def leaves(self):
        """(node, slot, ref) arrays for every leaf child in the tree."""
        child = self.nodes["child"]
        node, slot = np.nonzero(is_leaf(child) & ~is_empty(child))
        return node, slot, child[node, slot]

    def triangles(self):
        """All triangles, leaf by leaf in node order."""
        _, _, refs = self.leaves()
        if len(refs) == 0:
            return np.empty(0, TRIANGLE)
        return np.concatenate([self.leaf_triangles(r) for r in refs])


def read_tree(path):
    buf = open(path, "rb").read()
    h = _header(buf, TREE_HEADER, b"CABVHTRE", path)
    width, node_bytes = int(h["width"]), int(h["nodeBytes"])
    if (width, node_bytes) not in ((2, 64), (4, 128)):
        raise ValueError(f"{path}: width {width} with {node_bytes} B nodes")
    node_end = int(h["nodeOffset"]) + int(h["nodeCount"]) * node_bytes
    tri_end = int(h["triOffset"]) + int(h["triSectorCount"]) * SECTOR
    if node_end > int(h["triOffset"]) or tri_end > len(buf):
        raise ValueError(f"{path}: sections run past the end of the file")
    nodes = np.frombuffer(buf, NODE2 if width == 2 else NODE4, count=int(h["nodeCount"]),
                          offset=int(h["nodeOffset"]))
    tri_section = np.frombuffer(buf, np.uint8, count=tri_end - int(h["triOffset"]), offset=int(h["triOffset"]))
    return Tree(h, nodes, tri_section)


@dataclass
class Rays:
    header: np.void
    rays: np.ndarray  # RAY records


def read_rays(path):
    buf = open(path, "rb").read()
    h = _header(buf, RAY_HEADER, b"CABVHRAY", path)
    n = int(h["rayCount"])
    if len(buf) != RAY_HEADER.itemsize + n * RAY.itemsize:
        raise ValueError(f"{path}: size does not match rayCount {n}")
    return Rays(h, np.frombuffer(buf, RAY, count=n, offset=RAY_HEADER.itemsize))


@dataclass
class Trace:
    header: np.void
    events: np.ndarray  # EVENT records, sorted by ray, visit order within a ray

    def node_indices(self, tree):
        """Node index of each event, or -1 for triangle-block events."""
        h = tree.header
        sector = self.events["firstSector"].astype(np.int64)
        idx = (sector * SECTOR - int(h["nodeOffset"])) // int(h["nodeBytes"])
        return np.where(self.events["kind"] == NODE, idx, -1)

    def leaf_refs(self, tree):
        """Child reference of each triangle-block event, or 0 for node events."""
        ev = self.events
        rel = ev["firstSector"].astype(np.int64) - int(tree.header["triOffset"]) // SECTOR
        count = (ev["sectorCount"].astype(np.int64) * 2) // 3  # inverse of ceil(3k/2) for k in 1..15
        ref = (LEAF_BIT | (count << 27) | rel).astype(np.uint32)
        return np.where(ev["kind"] == TRI_BLOCK, ref, np.uint32(0))


def read_trace(path):
    buf = open(path, "rb").read()
    h = _header(buf, TRACE_HEADER, b"CABVHTRC", path)
    n = int(h["eventCount"])
    if len(buf) != TRACE_HEADER.itemsize + n * EVENT.itemsize:
        raise ValueError(f"{path}: size does not match eventCount {n}")
    return Trace(h, np.frombuffer(buf, EVENT, count=n, offset=TRACE_HEADER.itemsize))
