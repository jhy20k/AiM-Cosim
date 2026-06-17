#pragma once
// =============================================================================
// FunctionalDispatcher — Command dispatch + operand gathering + writeback
//
// Owns:
//   - All handle_* methods (previously inlined in Controller)
//   - MAC16 reusable staging buffer (avoids per-call stack allocation)
//   - Compute invocation via IComputeEngine
//   - Writeback orchestration via AiMMemoryManager
//   - Delegates functional logging to ResultLogger
// =============================================================================

#include "compute_engine/compute_engine.h"
#include "memory_manager/memory_manager.h"
#include "controller/result_logger.h"
#include "base/request.h"

#include <array>
#include <cstdint>

namespace aim_cosim {

class FunctionalDispatcher {
public:
    void init(AiMMemoryManager* mm, IComputeEngine* ce,
              ResultLogger* logger);

    // Main dispatch entry point — called from Controller::tick().
    // Returns measured rtl_cycles for compute-producing commands, else 0.
    uint64_t dispatch(const Ramulator::Request& req, int64_t sim_cycle);

private:
    // GDDR6 command indices (must match GDDR6.cpp m_commands)
    enum Cmd {
        CMD_ACT = 0, CMD_PREA = 1, CMD_PRE = 2, CMD_RD = 3, CMD_WR = 4,
        CMD_RDA = 5, CMD_WRA = 6, CMD_REFab = 7, CMD_REFpb = 8,
        CMD_ACT4 = 9, CMD_ACT16 = 10, CMD_PRE4 = 11,
        CMD_MAC = 12, CMD_MAC16 = 13, CMD_AF16 = 14, CMD_EWMUL16 = 15,
        CMD_RDCP = 16, CMD_WRCP = 17, CMD_WRGB = 18,
        CMD_RDMAC16 = 19, CMD_RDAF16 = 20, CMD_WRMAC16 = 21, CMD_WRA16 = 22,
        CMD_TMOD = 23, CMD_SYNC = 24, CMD_EOC = 25
    };

    // Row management
    void handle_act(const Ramulator::Request& req);
    void handle_pre(const Ramulator::Request& req);

    // Data writes
    void handle_wr(const Ramulator::Request& req);
    void handle_wrgb(const Ramulator::Request& req);
    void handle_wrmac16(const Ramulator::Request& req);
    void handle_wra16(const Ramulator::Request& req);

    // Data reads
    void handle_rd(const Ramulator::Request& req, int64_t sim_cycle);
    void handle_rdcp(const Ramulator::Request& req);
    void handle_wrcp(const Ramulator::Request& req);

    // Compute
    uint64_t handle_mac(const Ramulator::Request& req, int64_t sim_cycle);
    uint64_t handle_mac16(const Ramulator::Request& req, int64_t sim_cycle);
    uint64_t handle_af16(const Ramulator::Request& req, int64_t sim_cycle);
    uint64_t handle_ewmul16(const Ramulator::Request& req, int64_t sim_cycle);

    // Result reads
    void handle_rdmac16(const Ramulator::Request& req, int64_t sim_cycle);
    void handle_rdaf16(const Ramulator::Request& req, int64_t sim_cycle);

    // Helpers
    void flush_pending_mac16();
    int addr_to_bank_id(const std::vector<int>& addr_vec) const;
    void emit_data_packet(int command, int bank_id, int row, int col,
                          const uint16_t* data, int64_t sim_cycle);

    // Dependencies (non-owning)
    AiMMemoryManager* m_mm = nullptr;
    IComputeEngine* m_ce = nullptr;
    ResultLogger* m_logger = nullptr;

    // MAC16 reusable staging buffer — allocated once, reused every call
    std::array<std::array<uint16_t, 16>, 16> m_mac16_weights;
    std::array<const uint16_t*, 16> m_mac16_weight_ptrs{};
    std::array<uint16_t, 16> m_mac16_biases;
};

} // namespace aim_cosim
