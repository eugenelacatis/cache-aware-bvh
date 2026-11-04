"""Checks tools/cabvh.py against files written by the C++ side.

Needs bvh_tests and tracedump from the CMake build. Set CABVH_BIN to their
directory if it is not build/. Run with: pytest tests/test_cabvh.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import cabvh  # noqa: E402

BIN = Path(os.environ.get("CABVH_BIN", ROOT / "build"))
if not (BIN / "bvh_tests").exists() or not (BIN / "tracedump").exists():
    pytest.skip(f"bvh_tests and tracedump not found in {BIN}", allow_module_level=True)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cabvh")
    # bvh_tests writes test_tree.bvh (4-wide, 3000 triangles) and test.rays (1000 rays, seed 7).
    subprocess.run([BIN / "bvh_tests", d], check=True, capture_output=True)
    out = {}
    for mode in ("closest", "all"):
        out[mode] = d / f"{mode}.trace"
        subprocess.run([BIN / "tracedump", d / "test_tree.bvh", d / "test.rays", out[mode], mode],
                       check=True, capture_output=True)
    return d / "test_tree.bvh", d / "test.rays", out


def test_tree(files):
    tree = cabvh.read_tree(files[0])
    h = tree.header
    assert tree.width == 4 and h["nodeBytes"] == 128 and h["nodeOffset"] == 128
    assert h["triCount"] == 3000 and h["rootP"] == 1.0 and np.isnan(h["tau"])
    assert len(tree.nodes) == h["nodeCount"]

    child = tree.nodes["child"]
    internal = child[~cabvh.is_leaf(child)]
    assert np.all((internal > 0) & (internal < h["nodeCount"]))
    # Every node except the root has exactly one parent.
    assert np.array_equal(np.sort(internal), np.arange(1, h["nodeCount"]))

    tris = tree.triangles()
    assert np.array_equal(np.sort(tris["id"]), np.arange(3000))

    # A child's P is at most 1, and leaf triangles sit inside their box.
    used = ~cabvh.is_empty(child)
    assert np.all((tree.nodes["p"][used] > 0) & (tree.nodes["p"][used] <= 1))
    node, slot, refs = tree.leaves()
    for n, s, r in zip(node, slot, refs):
        lo, hi = tree.nodes["box"][n, s]
        v = tree.leaf_triangles(r)["v"].reshape(-1, 3)
        assert np.all(v >= lo - 1e-5) and np.all(v <= hi + 1e-5)


def test_rays(files):
    rays = cabvh.read_rays(files[1])
    assert rays.header["rayCount"] == 1000 and rays.header["seed"] == 7 and rays.header["rayKind"] == 0
    r = rays.rays
    assert np.allclose(np.linalg.norm(r["dir"], axis=1), 1, atol=1e-5)
    assert np.all(r["tmin"] == 0) and np.all(r["tmax"] > 0)


@pytest.mark.parametrize("mode,code", [("closest", 0), ("all", 1)])
def test_trace(files, mode, code):
    tree = cabvh.read_tree(files[0])
    trace = cabvh.read_trace(files[2][mode])
    h, ev = trace.header, trace.events
    assert h["mode"] == code and h["rayCount"] == 1000 and h["width"] == 4
    assert len(ev) == h["eventCount"] > 0
    assert np.all(np.diff(ev["ray"].astype(np.int64)) >= 0)

    # Node events decode to real nodes, and each ray's first event is the root.
    nodes = trace.node_indices(tree)
    is_node = ev["kind"] == cabvh.NODE
    assert np.all(ev["sectorCount"][is_node] == 4)
    assert np.all((nodes[is_node] >= 0) & (nodes[is_node] < tree.header["nodeCount"]))
    first = np.r_[True, ev["ray"][1:] != ev["ray"][:-1]]
    assert np.all(nodes[first] == 0)

    # Triangle-block events rebuild child references that exist in the tree.
    _, _, refs = tree.leaves()
    blocks = trace.leaf_refs(tree)[ev["kind"] == cabvh.TRI_BLOCK]
    assert len(blocks) > 0 and np.all(np.isin(blocks, refs))
