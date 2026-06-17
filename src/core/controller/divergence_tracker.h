#pragma once
// =============================================================================
// DivergenceTracker — Stage 1 timing semantics + Stage 4 gating
//
// Owns:
//   - 2-phase timing (Phase A: record at producer issue, Phase B: finalize at consumer)
//   - Static pair classification (NEVER_DIVERGE / MAYBE_DIVERGE) from RTL upper bound
//     vs DRAM min_constraint, computed once in init()
//   - Ready-cycle scoreboard for Stage 4 conditional consumer gating
//   - m_divergence_count stat (rtl_cycles > dram_elapsed events)
// =============================================================================

#include "architecture/architecture_types.h"
#include "controller/timing_constraint_query.h"

#include <cstdint>
#include <vector>

namespace aim_cosim {

enum class DivergenceTier : uint8_t {
    NEVER_DIVERGE,   // rtl_cycles <= min_constraint → runtime check skip
    MAYBE_DIVERGE    // rtl_cycles > min_constraint → runtime dram_elapsed needed
};

struct PendingComputeTiming {
    bool valid = false;
    int64_t sim_issue_cycle = 0;
    const char* command_name = "";
    uint64_t rtl_cycles = 0;
};

// Result of Phase B finalization
struct DivergenceResult {
    bool logged = false;         // true if a timing row was produced
    int64_t sim_issue_cycle = 0;
    const char* command_name = "";
    uint64_t rtl_cycles = 0;
    int min_constraint = 0;
    int64_t dram_elapsed = 0;
    bool diverged = false;
    uint64_t deferred_cycles = 0;
    bool fallback_flush = false;
};

struct PairTimingProfile {
    ComputeConsumerPair pair;
    int min_constraint = 0;
    DivergenceTier tier = DivergenceTier::MAYBE_DIVERGE;
};

struct ReadyCycleState {
    uint64_t rdmac16_ready_cycle = 0;
    uint64_t rdaf16_ready_cycle = 0;
};

class DivergenceTracker {
public:
    // Consumer type for finalize_mac — selects which min_constraint to compare against
    enum class MacConsumer : uint8_t { AF16, RDMAC16 };

    // Initialize with the backend's compute→consumer pair table.
    // For each pair, looks up min_constraint from the DRAM spec and pre-classifies tier.
    void init(Ramulator::IDRAM* dram, const std::vector<ComputeConsumerPair>& pairs);
    void reset();

    // Phase A: record compute timing at producer issue.
    // If a previous pending entry was never finalized (consumer never arrived),
    // returns its DivergenceResult so the caller can flush it as a fallback row.
    DivergenceResult record_mac(const char* cmd_name, uint64_t rtl_cycles, int64_t sim_cycle);
    DivergenceResult record_af16(uint64_t rtl_cycles, int64_t sim_cycle);

    // Phase B: consumer arrived — compute dram_elapsed and produce a finalized row.
    DivergenceResult finalize_mac(int64_t consumer_cycle,
                                  MacConsumer consumer = MacConsumer::AF16);
    DivergenceResult finalize_af16(int64_t consumer_cycle);

    // Trace teardown: emit fallback rows for any still-pending entries.
    std::vector<DivergenceResult> flush_all_pending();

    // Stage 4 gating hooks (called from controller via backend forward).
    bool should_defer_consumer(int command, uint64_t now_cycle) const;
    void note_consumer_deferred(int command, uint64_t cycles = 1);

    size_t divergence_count() const { return m_divergence_count; }
    size_t& divergence_count_ref() { return m_divergence_count; }
    const std::vector<PairTimingProfile>& pair_profiles() const { return m_profiles; }

private:
    DivergenceResult do_finalize(PendingComputeTiming& pending, int min_constraint,
                                 int64_t consumer_cycle, uint64_t& deferred_cycles);
    DivergenceResult do_flush(PendingComputeTiming& pending, uint64_t& deferred_cycles);

    int  get_min_mac_to_af16(const char* mac_cmd) const;
    int  get_min_mac_to_rdmac16(const char* mac_cmd) const;
    int  lookup_min_constraint(const char* producer_cmd, const char* consumer_cmd) const;
    const PairTimingProfile* lookup_profile(const char* producer_cmd,
                                            const char* consumer_cmd) const;

    void arm_rdmac16_ready_cycle(const char* producer_cmd, uint64_t rtl_cycles,
                                 int64_t sim_cycle);
    void arm_rdaf16_ready_cycle(uint64_t rtl_cycles, int64_t sim_cycle);
    void reset_runtime_state();

    PendingComputeTiming m_pending_mac;
    PendingComputeTiming m_pending_af;
    std::vector<PairTimingProfile> m_profiles;
    ReadyCycleState m_ready_cycles;
    uint64_t m_pending_rdmac16_defer_cycles = 0;
    uint64_t m_pending_rdaf16_defer_cycles = 0;

    // Cached min_constraints
    int m_min_mac_to_rdmac16 = 0;
    int m_min_mac16_to_rdmac16 = 0;
    int m_min_mac_to_af16 = 0;
    int m_min_mac16_to_af16 = 0;
    int m_min_af16_to_rdaf16 = 0;
    int m_cmd_rdmac16 = -1;
    int m_cmd_rdaf16 = -1;

    size_t m_divergence_count = 0;
};

} // namespace aim_cosim
