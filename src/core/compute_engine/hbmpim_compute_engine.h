#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include "hbmpim/pim_block_model.h"
#include "hbmpim/pim_cmd.h"

#include "base/request.h"

#include <cstdint>

namespace aim_cosim {

class HBMPIMMemoryManager;

class HBMPIMComputeEngine {
public:
    enum class Mode : uint8_t {
        SOFTWARE,
        RTL,
    };

    struct TriggerResult {
        uint32_t previous_pc = 0;
        uint32_t next_pc = 0;
        uint32_t raw_instruction = 0;
        uint64_t rtl_cycles = 0;
        bool exit_flag = false;
        bool success = true;
    };

    explicit HBMPIMComputeEngine(Mode mode = Mode::SOFTWARE);

    void reset();
    Mode mode() const { return m_mode; }

    TriggerResult execute_trigger(HBMPIMMemoryManager& memory_manager,
                                  const Ramulator::HBMPIMRequestInfo& request_info);

private:
    static constexpr uint64_t DEFAULT_BLOCK_MASK = 0x1ULL;

    using Burst = PIMBlockModel::Burst;

    Mode m_mode = Mode::SOFTWARE;
    int32_t m_last_jump_pc = -1;
    int32_t m_jump_remaining = -1;

    static uint64_t normalize_mask(int64_t pim_block_mask);
    static int select_grf_index(bool use_high_bits, int row_addr, int col_addr, int fallback_index);
    static Burst read_operand(HBMPIMMemoryManager& memory_manager,
                              int block_id,
                              const Ramulator::HBMPIMRequestInfo& request_info,
                              PIMOpdType operand,
                              int operand_index,
                              bool is_auto,
                              bool is_mac);
    static void write_operand(HBMPIMMemoryManager& memory_manager,
                              int block_id,
                              const Ramulator::HBMPIMRequestInfo& request_info,
                              PIMOpdType operand,
                              int operand_index,
                              bool is_auto,
                              bool is_mac,
                              const Burst& value);
    TriggerResult execute_software_trigger(HBMPIMMemoryManager& memory_manager,
                                          const Ramulator::HBMPIMRequestInfo& request_info);
};

} // namespace aim_cosim
