#include "trace_dump.h"

namespace {

// Records fetches as trace events and counts work for RayStats.
struct TraceProbe {
    const TreeHeader& h;
    std::vector<TraceEvent>& out;
    uint32_t ray = 0;
    CountProbe count;

    void node(uint32_t index, int children) {
        uint64_t byte = h.nodeOffset + uint64_t(index) * h.nodeBytes;
        out.push_back({ray, uint32_t(byte / kSectorBytes), uint16_t(h.nodeBytes / kSectorBytes),
                       uint8_t(AccessKind::Node), 0});
        count.node(index, children);
    }
    void leaf(uint32_t ref) {
        uint32_t first = uint32_t(h.triOffset / kSectorBytes) + refLeafSector(ref);
        out.push_back({ray, first, uint16_t(leafSectors(refLeafCount(ref))), uint8_t(AccessKind::TriBlock), 0});
        count.leaf(ref);
    }
};

template <int W>
void dumpAll(const TreeFile& tree, const std::vector<Ray>& rays, TraceMode mode, std::vector<RayStats>* stats,
             std::vector<TraceEvent>& out) {
    const TreeHeader& h = tree.header();
    TraceProbe probe{h, out, 0, {}};
    for (uint32_t i = 0; i < rays.size(); ++i) {
        probe.ray = i;
        probe.count = CountProbe{};
        if (mode == TraceMode::AllHit)
            traceAll<W>(tree.nodes<W>(), tree.triSection(), h.sceneBox, rays[i], probe);
        else
            traceClosest<W>(tree.nodes<W>(), tree.triSection(), h.sceneBox, rays[i], probe);
        if (stats) stats->push_back(probe.count.stats);
    }
}

}  // namespace

std::vector<TraceEvent> dumpTrace(const TreeFile& tree, const std::vector<Ray>& rays, TraceMode mode,
                                  std::vector<RayStats>* stats) {
    std::vector<TraceEvent> events;
    if (stats) stats->clear();
    if (tree.header().width == 2)
        dumpAll<2>(tree, rays, mode, stats, events);
    else
        dumpAll<4>(tree, rays, mode, stats, events);
    return events;
}
