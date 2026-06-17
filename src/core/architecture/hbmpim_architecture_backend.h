#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include "architecture/architecture_backend.h"
#include "compute_engine/hbmpim_compute_engine.h"
#include "controller/divergence_tracker.h"
#include "controller/result_logger.h"
#include "memory_manager/hbmpim_memory_manager.h"

#include <memory>
#include <vector>

namespace aim_cosim {

class HBMPIMArchitectureBackend final : public IArchitectureBackend {
public:
    HBMPIMArchitectureBackend() = default;
    ~HBMPIMArchitectureBackend() override;

    void init(const ArchitectureInitContext& ctx) override;
    void reset() override;

    void on_command_issued(const Ramulator::Request& req,
                           uint64_t sim_issue_cycle) override;

    void flush_pending_timing() override;
    bool can_defer_consumer(int command, uint64_t now_cycle) const override;
    void note_consumer_deferred(int command, uint64_t cycles = 1) override;

    std::vector<ComputeConsumerPair> get_compute_consumer_pairs() const override;
    size_t& divergence_count_ref() override {
        return m_divergence_tracker.divergence_count_ref();
    }

private:
    void handle_dram_command(const Ramulator::Request& req);
    void handle_hbmpim_command(const Ramulator::Request& req, uint64_t sim_issue_cycle);
    void emit_readback_if_enabled(const Ramulator::Request& req,
                                  uint64_t sim_issue_cycle);
    int infer_target_word(const Ramulator::HBMPIMRequestInfo& info) const;
    void for_each_target_block(uint64_t pim_block_mask,
                               const std::function<void(int block_id)>& fn);
    int bank_id_from_addr_vec(const std::vector<int>& addr_vec) const;
    int row_from_addr_vec(const std::vector<int>& addr_vec) const;

    std::unique_ptr<HBMPIMMemoryManager> m_memory_manager;
    std::unique_ptr<HBMPIMComputeEngine> m_compute_engine;
    ResultLogger m_result_logger;
    DivergenceTracker m_divergence_tracker;
    std::vector<ComputeConsumerPair> m_pairs;

    Ramulator::IDRAM* m_dram = nullptr;
    int m_logical_pseudochannel_id = -1;
    int m_cmd_act = -1;
    int m_cmd_pre = -1;
    int m_cmd_prea = -1;
    int m_bankgroup_level = -1;
    int m_bank_level = -1;
    int m_row_level = -1;
    bool m_initialized = false;
};

} // namespace aim_cosim
