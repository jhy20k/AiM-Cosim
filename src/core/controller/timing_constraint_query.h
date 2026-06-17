#pragma once
// =============================================================================
// TimingConstraintQuery — Read-only query wrapper for IDRAM::m_timing_cons
//
// Extracts inter-command minimum latency from the existing DRAM spec data
// without modifying aim_simulator internals. Works with any DRAM spec
// (GDDR6, LPDDR5, HBM-PIM) that uses populate_timingcons(). HBM-PIM remains a
// future-work scaffold in this repository.
// =============================================================================

#include "dram/dram.h"

namespace aim_cosim {

class TimingConstraintQuery {
    Ramulator::IDRAM* m_dram;

public:
    explicit TimingConstraintQuery(Ramulator::IDRAM* dram) : m_dram(dram) {}

    /// Returns minimum latency (in cycles) from preceding_cmd to following_cmd
    /// at the specified hierarchy level. Returns 0 if no constraint exists.
    ///
    /// Example: min_latency("MAC16", "RDMAC16", "channel") returns nCCDL value.
    int min_latency(const char* preceding, const char* following,
                    const char* level = "channel") const {
        if (!m_dram) return 0;
        if (!m_dram->m_commands.contains(preceding)) return 0;
        if (!m_dram->m_commands.contains(following)) return 0;
        if (!m_dram->m_levels.contains(level)) return 0;

        int lvl = m_dram->m_levels(level);
        int p_cmd = m_dram->m_commands(preceding);
        int f_cmd = m_dram->m_commands(following);

        if (lvl < 0 || lvl >= static_cast<int>(m_dram->m_timing_cons.size()))
            return 0;
        if (p_cmd < 0 || p_cmd >= static_cast<int>(m_dram->m_timing_cons[lvl].size()))
            return 0;

        for (const auto& t : m_dram->m_timing_cons[lvl][p_cmd]) {
            if (t.cmd == f_cmd) return t.val;
        }
        return 0;
    }
};

} // namespace aim_cosim
