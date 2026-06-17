// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include "compute_engine/hbmpim_compute_engine.h"

#include "memory_manager/hbmpim_memory_manager.h"

#include <algorithm>
#include <stdexcept>

namespace aim_cosim {

HBMPIMComputeEngine::HBMPIMComputeEngine(Mode mode) : m_mode(mode) {}

void HBMPIMComputeEngine::reset() {
    m_last_jump_pc = -1;
    m_jump_remaining = -1;
}

HBMPIMComputeEngine::TriggerResult
HBMPIMComputeEngine::execute_trigger(HBMPIMMemoryManager& memory_manager,
                                     const Ramulator::HBMPIMRequestInfo& request_info) {
    switch (m_mode) {
        case Mode::SOFTWARE:
            return execute_software_trigger(memory_manager, request_info);
        case Mode::RTL:
            throw std::runtime_error("HBMPIMComputeEngine RTL mode is not implemented");
    }
    throw std::runtime_error("HBMPIMComputeEngine unknown mode");
}

uint64_t HBMPIMComputeEngine::normalize_mask(int64_t pim_block_mask) {
    return (pim_block_mask <= 0) ? DEFAULT_BLOCK_MASK : static_cast<uint64_t>(pim_block_mask);
}

int HBMPIMComputeEngine::select_grf_index(bool use_high_bits,
                                          int row_addr,
                                          int col_addr,
                                          int fallback_index) {
    if (!use_high_bits) {
        return (col_addr >= 0) ? (col_addr & 0x7) : (fallback_index & 0x7);
    }
    if (row_addr < 0 || col_addr < 0) {
        return fallback_index & 0x7;
    }
    return (((row_addr & 0x1) << 2) | ((col_addr >> 3) & 0x3)) & 0x7;
}

HBMPIMComputeEngine::Burst HBMPIMComputeEngine::read_operand(
    HBMPIMMemoryManager& memory_manager,
    int block_id,
    const Ramulator::HBMPIMRequestInfo& request_info,
    PIMOpdType operand,
    int operand_index,
    bool is_auto,
    bool is_mac) {
    switch (operand) {
        case PIMOpdType::A_OUT:
            return memory_manager.read_a_out_fp16_lanes(block_id);
        case PIMOpdType::M_OUT:
            return memory_manager.read_m_out_fp16_lanes(block_id);
        case PIMOpdType::EVEN_BANK:
            return memory_manager.read_bank_fp16_lanes(block_id * 2);
        case PIMOpdType::ODD_BANK:
            return memory_manager.read_bank_fp16_lanes(block_id * 2 + 1);
        case PIMOpdType::GRF_A: {
            const int index = is_auto
                ? select_grf_index(false, request_info.row_addr, request_info.col_addr, operand_index)
                : (operand_index & 0x7);
            return memory_manager.read_grf_a_fp16_lanes(block_id, index);
        }
        case PIMOpdType::GRF_B: {
            const int index = is_auto
                ? select_grf_index(is_mac, request_info.row_addr, request_info.col_addr, operand_index)
                : (operand_index & 0x7);
            return memory_manager.read_grf_b_fp16_lanes(block_id, index);
        }
        case PIMOpdType::SRF_M: {
            const auto srf = memory_manager.read_srf_fp16_lanes(block_id);
            return PIMBlockModel::broadcast(srf[operand_index & 0x7]);
        }
        case PIMOpdType::SRF_A: {
            const auto srf = memory_manager.read_srf_fp16_lanes(block_id);
            return PIMBlockModel::broadcast(srf[(operand_index & 0x7) + 8]);
        }
    }
    return {};
}

void HBMPIMComputeEngine::write_operand(HBMPIMMemoryManager& memory_manager,
                                        int block_id,
                                        const Ramulator::HBMPIMRequestInfo& request_info,
                                        PIMOpdType operand,
                                        int operand_index,
                                        bool is_auto,
                                        bool is_mac,
                                        const Burst& value) {
    switch (operand) {
        case PIMOpdType::A_OUT:
            memory_manager.write_a_out_fp16_lanes(block_id, value);
            return;
        case PIMOpdType::M_OUT:
            memory_manager.write_m_out_fp16_lanes(block_id, value);
            return;
        case PIMOpdType::EVEN_BANK:
            memory_manager.write_bank_fp16_lanes(block_id * 2, value);
            return;
        case PIMOpdType::ODD_BANK:
            memory_manager.write_bank_fp16_lanes(block_id * 2 + 1, value);
            return;
        case PIMOpdType::GRF_A: {
            const int index = is_auto
                ? select_grf_index(false, request_info.row_addr, request_info.col_addr, operand_index)
                : (operand_index & 0x7);
            memory_manager.write_grf_a_fp16_lanes(block_id, index, value);
            return;
        }
        case PIMOpdType::GRF_B: {
            const int index = is_auto
                ? select_grf_index(is_mac, request_info.row_addr, request_info.col_addr, operand_index)
                : (operand_index & 0x7);
            memory_manager.write_grf_b_fp16_lanes(block_id, index, value);
            return;
        }
        case PIMOpdType::SRF_M:
        case PIMOpdType::SRF_A:
            memory_manager.write_srf_fp16_lanes(block_id, value);
            return;
    }
}

HBMPIMComputeEngine::TriggerResult HBMPIMComputeEngine::execute_software_trigger(
    HBMPIMMemoryManager& memory_manager,
    const Ramulator::HBMPIMRequestInfo& request_info) {
    TriggerResult result{};
    result.previous_pc = memory_manager.pc();
    result.next_pc = memory_manager.pc();
    result.exit_flag = memory_manager.exit_flag();
    result.success = true;

    if (memory_manager.exit_flag()) {
        return result;
    }

    const auto raw_instruction = memory_manager.read_crf(static_cast<int>(memory_manager.pc()));
    const auto cmd = PIMCmd::from_int(raw_instruction);
    result.raw_instruction = raw_instruction;

    if (cmd.type == PIMCmdType::EXIT) {
        memory_manager.set_exit_flag(true);
        result.exit_flag = true;
        return result;
    }

    if (cmd.type == PIMCmdType::JUMP) {
        const auto current_pc = memory_manager.pc();
        if (m_last_jump_pc != static_cast<int32_t>(current_pc)) {
            if (cmd.loop_counter > 0) {
                m_last_jump_pc = static_cast<int32_t>(current_pc);
                m_jump_remaining = cmd.loop_counter;
            }
        }

        if (m_jump_remaining > 0) {
            if (cmd.loop_offset > static_cast<int>(current_pc)) {
                throw std::runtime_error("HBMPIMComputeEngine JUMP underflow");
            }
            memory_manager.set_pc(current_pc - static_cast<uint32_t>(cmd.loop_offset) + 1u);
            --m_jump_remaining;
        } else {
            m_last_jump_pc = -1;
            memory_manager.advance_pc();
        }

        result.next_pc = memory_manager.pc();
        return result;
    }

    if (cmd.type == PIMCmdType::NOP) {
        memory_manager.advance_pc();
        result.next_pc = memory_manager.pc();
        return result;
    }

    const bool is_auto = (cmd.type == PIMCmdType::FILL) ? true : (cmd.is_auto != 0);
    const bool is_mac = (cmd.type == PIMCmdType::MAC);
    const auto mask = normalize_mask(request_info.pim_block_mask);

    for (int block_id = 0; block_id < HBMPIMMemoryManager::NUM_PIM_BLOCKS; ++block_id) {
        if ((mask & (1ULL << block_id)) == 0) {
            continue;
        }

        Burst dst{};
        Burst src0 = read_operand(memory_manager, block_id, request_info,
                                  cmd.src0, cmd.src0_idx, is_auto, false);

        switch (cmd.type) {
            case PIMCmdType::MOV:
            case PIMCmdType::FILL:
                dst = (cmd.type == PIMCmdType::FILL)
                    ? PIMBlockModel::fill(src0, cmd.is_relu != 0)
                    : PIMBlockModel::mov(src0, cmd.is_relu != 0);
                break;
            case PIMCmdType::ADD: {
                auto src1 = read_operand(memory_manager, block_id, request_info,
                                         cmd.src1, cmd.src1_idx, is_auto, false);
                dst = PIMBlockModel::add(src0, src1);
                break;
            }
            case PIMCmdType::MUL: {
                auto src1 = read_operand(memory_manager, block_id, request_info,
                                         cmd.src1, cmd.src1_idx, is_auto, false);
                dst = PIMBlockModel::mul(src0, src1);
                break;
            }
            case PIMCmdType::MAC: {
                auto src1 = read_operand(memory_manager, block_id, request_info,
                                         cmd.src1, cmd.src1_idx, is_auto, true);
                auto acc = read_operand(memory_manager, block_id, request_info,
                                        cmd.dst, cmd.dst_idx, is_auto, true);
                dst = PIMBlockModel::mac(acc, src0, src1);
                break;
            }
            case PIMCmdType::MAD: {
                auto src1 = read_operand(memory_manager, block_id, request_info,
                                         cmd.src1, cmd.src1_idx, is_auto, false);
                auto src2 = read_operand(memory_manager, block_id, request_info,
                                         cmd.src2, cmd.src1_idx, is_auto, false);
                dst = PIMBlockModel::mad(src0, src1, src2);
                break;
            }
            default:
                break;
        }

        write_operand(memory_manager, block_id, request_info,
                      cmd.dst, cmd.dst_idx, is_auto, is_mac, dst);
    }

    memory_manager.advance_pc();
    result.next_pc = memory_manager.pc();
    result.exit_flag = memory_manager.exit_flag();
    return result;
}

} // namespace aim_cosim
