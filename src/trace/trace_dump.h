#pragma once

// CPU version of the node-access trace dump (docs/FORMATS.md). It runs the same
// traversal as the GPU kernels and records one event per node or leaf fetch.

#include <vector>

#include "bvh_io.h"
#include "traverse.h"

// Traces every ray in index order. Events come out sorted by ray and, within a
// ray, in visit order. If `stats` is not null it gets one entry per ray.
std::vector<TraceEvent> dumpTrace(const TreeFile& tree, const std::vector<Ray>& rays, TraceMode mode,
                                  std::vector<RayStats>* stats = nullptr);
