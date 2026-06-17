#include "controller/divergence_tracker.h"

#include <cstring>

namespace aim_cosim {

void DivergenceTracker::init(Ramulator::IDRAM* dram,
                             const std::vector<ComputeConsumerPair>& pairs) {
    TimingConstraintQuery tcq(dram);

    // 1. Build per-pair profiles with static tier classification.
    m_profiles.clear();
    m_profiles.reserve(pairs.size());
    for (const auto& pair : pairs) {
        int min_constraint = tcq.min_latency(pair.producer_cmd, pair.consumer_cmd, pair.level);
        DivergenceTier tier =
            (pair.rtl_cycle_upper_bound <= static_cast<uint32_t>(min_constraint))
                ? DivergenceTier::NEVER_DIVERGE
                : DivergenceTier::MAYBE_DIVERGE;
        m_profiles.push_back({pair, min_constraint, tier});
    }

    // 2. Cache the five hot-path min_constraints used by finalize_*.
    //    Returns 0 when a pair was not declared by the backend.
    m_min_mac_to_rdmac16   = lookup_min_constraint("MAC",   "RDMAC16");
    m_min_mac16_to_rdmac16 = lookup_min_constraint("MAC16", "RDMAC16");
    m_min_mac_to_af16      = lookup_min_constraint("MAC",   "AF16");
    m_min_mac16_to_af16    = lookup_min_constraint("MAC16", "AF16");
    m_min_af16_to_rdaf16   = lookup_min_constraint("AF16",  "RDAF16");

    // 3. Cache numeric command IDs for the consumer-side gating fast path.
    m_cmd_rdmac16 = (dram && dram->m_commands.contains("RDMAC16"))
                  ? dram->m_commands("RDMAC16") : -1;
    m_cmd_rdaf16  = (dram && dram->m_commands.contains("RDAF16"))
                  ? dram->m_commands("RDAF16")  : -1;

    reset_runtime_state();
}

void DivergenceTracker::reset() {
    reset_runtime_state();
}

int DivergenceTracker::get_min_mac_to_af16(const char* mac_cmd) const {
    return (std::strcmp(mac_cmd, "MAC") == 0) ? m_min_mac_to_af16 : m_min_mac16_to_af16;
}

int DivergenceTracker::get_min_mac_to_rdmac16(const char* mac_cmd) const {
    return (std::strcmp(mac_cmd, "MAC") == 0) ? m_min_mac_to_rdmac16 : m_min_mac16_to_rdmac16;
}

DivergenceResult DivergenceTracker::record_mac(const char* cmd_name, uint64_t rtl_cycles,
                                                int64_t sim_cycle) {
    DivergenceResult flush_result{};
    if (m_pending_mac.valid) {
        flush_result = do_flush(m_pending_mac, m_pending_rdmac16_defer_cycles);
    }
    m_pending_mac = {true, sim_cycle, cmd_name, rtl_cycles};
    m_pending_rdmac16_defer_cycles = 0;
    arm_rdmac16_ready_cycle(cmd_name, rtl_cycles, sim_cycle);
    return flush_result;
}

DivergenceResult DivergenceTracker::record_af16(uint64_t rtl_cycles, int64_t sim_cycle) {
    DivergenceResult flush_result{};
    if (m_pending_af.valid) {
        flush_result = do_flush(m_pending_af, m_pending_rdaf16_defer_cycles);
    }
    m_pending_af = {true, sim_cycle, "AF16", rtl_cycles};
    m_pending_rdaf16_defer_cycles = 0;
    arm_rdaf16_ready_cycle(rtl_cycles, sim_cycle);
    return flush_result;
}

DivergenceResult DivergenceTracker::finalize_mac(int64_t consumer_cycle, MacConsumer consumer) {
    if (!m_pending_mac.valid) return {};
    int mc = (consumer == MacConsumer::RDMAC16)
           ? get_min_mac_to_rdmac16(m_pending_mac.command_name)
           : get_min_mac_to_af16(m_pending_mac.command_name);
    auto result = do_finalize(m_pending_mac, mc, consumer_cycle,
                              m_pending_rdmac16_defer_cycles);
    if (consumer == MacConsumer::RDMAC16) {
        m_ready_cycles.rdmac16_ready_cycle = 0;
    }
    return result;
}

DivergenceResult DivergenceTracker::finalize_af16(int64_t consumer_cycle) {
    if (!m_pending_af.valid) return {};
    auto result = do_finalize(m_pending_af, m_min_af16_to_rdaf16, consumer_cycle,
                              m_pending_rdaf16_defer_cycles);
    m_ready_cycles.rdaf16_ready_cycle = 0;
    return result;
}

DivergenceResult DivergenceTracker::do_finalize(PendingComputeTiming& pending,
                                                 int min_constraint,
                                                 int64_t consumer_cycle,
                                                 uint64_t& deferred_cycles) {
    int64_t dram_elapsed = consumer_cycle - pending.sim_issue_cycle;
    bool diverged = false;

    // Fast path: if rtl_cycles <= min_constraint, structurally impossible
    if (static_cast<int64_t>(pending.rtl_cycles) > min_constraint) {
        diverged = (static_cast<int64_t>(pending.rtl_cycles) > dram_elapsed);
    }

    if (diverged) m_divergence_count++;

    DivergenceResult r{};
    r.logged = true;
    r.sim_issue_cycle = pending.sim_issue_cycle;
    r.command_name = pending.command_name;
    r.rtl_cycles = pending.rtl_cycles;
    r.min_constraint = min_constraint;
    r.dram_elapsed = dram_elapsed;
    r.diverged = diverged;
    r.deferred_cycles = deferred_cycles;
    r.fallback_flush = false;

    pending.valid = false;
    deferred_cycles = 0;
    return r;
}

DivergenceResult DivergenceTracker::do_flush(PendingComputeTiming& pending,
                                            uint64_t& deferred_cycles) {
    DivergenceResult r{};
    r.logged = true;
    r.sim_issue_cycle = pending.sim_issue_cycle;
    r.command_name = pending.command_name;
    r.rtl_cycles = pending.rtl_cycles;
    r.min_constraint = 0;
    r.dram_elapsed = -1;
    r.diverged = false;
    r.deferred_cycles = deferred_cycles;
    r.fallback_flush = true;

    pending.valid = false;
    deferred_cycles = 0;
    return r;
}

std::vector<DivergenceResult> DivergenceTracker::flush_all_pending() {
    std::vector<DivergenceResult> out;
    if (m_pending_mac.valid) {
        out.push_back(do_flush(m_pending_mac, m_pending_rdmac16_defer_cycles));
    }
    if (m_pending_af.valid) {
        out.push_back(do_flush(m_pending_af, m_pending_rdaf16_defer_cycles));
    }
    m_ready_cycles = {};
    return out;
}

int DivergenceTracker::lookup_min_constraint(const char* producer_cmd,
                                             const char* consumer_cmd) const {
    if (const auto* profile = lookup_profile(producer_cmd, consumer_cmd)) {
        return profile->min_constraint;
    }
    return 0;
}

const PairTimingProfile* DivergenceTracker::lookup_profile(const char* producer_cmd,
                                                           const char* consumer_cmd) const {
    for (const auto& profile : m_profiles) {
        if (std::strcmp(profile.pair.producer_cmd, producer_cmd) == 0 &&
            std::strcmp(profile.pair.consumer_cmd, consumer_cmd) == 0) {
            return &profile;
        }
    }
    return nullptr;
}

void DivergenceTracker::arm_rdmac16_ready_cycle(const char* producer_cmd,
                                                uint64_t rtl_cycles,
                                                int64_t sim_cycle) {
    const auto* profile = lookup_profile(producer_cmd, "RDMAC16");
    if (!profile || !profile->pair.supports_gating ||
        profile->tier != DivergenceTier::MAYBE_DIVERGE) {
        m_ready_cycles.rdmac16_ready_cycle = 0;
        return;
    }

    m_ready_cycles.rdmac16_ready_cycle =
        static_cast<uint64_t>(sim_cycle) + rtl_cycles;
}

void DivergenceTracker::arm_rdaf16_ready_cycle(uint64_t rtl_cycles,
                                               int64_t sim_cycle) {
    const auto* profile = lookup_profile("AF16", "RDAF16");
    if (!profile || !profile->pair.supports_gating ||
        profile->tier != DivergenceTier::MAYBE_DIVERGE) {
        m_ready_cycles.rdaf16_ready_cycle = 0;
        return;
    }

    m_ready_cycles.rdaf16_ready_cycle =
        static_cast<uint64_t>(sim_cycle) + rtl_cycles;
}

bool DivergenceTracker::should_defer_consumer(int command, uint64_t now_cycle) const {
    if (command == m_cmd_rdmac16) {
        return m_ready_cycles.rdmac16_ready_cycle != 0 &&
               now_cycle < m_ready_cycles.rdmac16_ready_cycle;
    }
    if (command == m_cmd_rdaf16) {
        return m_ready_cycles.rdaf16_ready_cycle != 0 &&
               now_cycle < m_ready_cycles.rdaf16_ready_cycle;
    }
    return false;
}

void DivergenceTracker::note_consumer_deferred(int command, uint64_t cycles) {
    if (cycles == 0) {
        return;
    }
    if (command == m_cmd_rdmac16 && m_pending_mac.valid) {
        m_pending_rdmac16_defer_cycles += cycles;
        return;
    }
    if (command == m_cmd_rdaf16 && m_pending_af.valid) {
        m_pending_rdaf16_defer_cycles += cycles;
    }
}

void DivergenceTracker::reset_runtime_state() {
    m_pending_mac = {};
    m_pending_af = {};
    m_ready_cycles = {};
    m_pending_rdmac16_defer_cycles = 0;
    m_pending_rdaf16_defer_cycles = 0;
    m_divergence_count = 0;
}

} // namespace aim_cosim
