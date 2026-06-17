// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include "hbmpim/pim_cmd.h"

#include <sstream>
#include <stdexcept>

namespace aim_cosim {

PIMCmd::PIMCmd(PIMCmdType type_value, int loop_counter_value)
    : type(type_value), loop_counter(loop_counter_value) {}

PIMCmd::PIMCmd(PIMCmdType type_value, int loop_counter_value, int loop_offset_value)
    : type(type_value), loop_counter(loop_counter_value), loop_offset(loop_offset_value) {}

PIMCmd::PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
               int is_auto_value, int dst_idx_value, int src0_idx_value,
               int src1_idx_value, int is_relu_value)
    : type(type_value), dst(dst_value), src0(src0_value), is_auto(is_auto_value),
      dst_idx(dst_idx_value), src0_idx(src0_idx_value), src1_idx(src1_idx_value),
      is_relu(is_relu_value) {}

PIMCmd::PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
               PIMOpdType src1_value, int is_auto_value, int dst_idx_value,
               int src0_idx_value, int src1_idx_value)
    : type(type_value), dst(dst_value), src0(src0_value), src1(src1_value),
      is_auto(is_auto_value), dst_idx(dst_idx_value), src0_idx(src0_idx_value),
      src1_idx(src1_idx_value) {}

PIMCmd::PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
               PIMOpdType src1_value, PIMOpdType src2_value, int is_auto_value,
               int dst_idx_value, int src0_idx_value, int src1_idx_value)
    : type(type_value), dst(dst_value), src0(src0_value), src1(src1_value),
      src2(src2_value), is_auto(is_auto_value), dst_idx(dst_idx_value),
      src0_idx(src0_idx_value), src1_idx(src1_idx_value) {}

PIMCmd PIMCmd::from_int(uint32_t value) {
    PIMCmd cmd;
    cmd.type = static_cast<PIMCmdType>(from_bits(value, 4, 28));
    switch (cmd.type) {
        case PIMCmdType::EXIT:
            break;
        case PIMCmdType::NOP:
            cmd.loop_counter = static_cast<int>(from_bits(value, 11, 0));
            break;
        case PIMCmdType::JUMP:
            cmd.loop_counter = static_cast<int>(from_bits(value, 17, 11));
            cmd.loop_offset = static_cast<int>(from_bits(value, 11, 0));
            break;
        case PIMCmdType::FILL:
        case PIMCmdType::MOV:
            cmd.dst = static_cast<PIMOpdType>(from_bits(value, 3, 25));
            cmd.src0 = static_cast<PIMOpdType>(from_bits(value, 3, 22));
            cmd.is_relu = static_cast<int>(from_bits(value, 1, 12));
            cmd.dst_idx = static_cast<int>(from_bits(value, 4, 8));
            cmd.src0_idx = static_cast<int>(from_bits(value, 4, 4));
            cmd.src1_idx = static_cast<int>(from_bits(value, 4, 0));
            break;
        case PIMCmdType::MAD:
            cmd.src2 = static_cast<PIMOpdType>(from_bits(value, 3, 16));
            [[fallthrough]];
        case PIMCmdType::ADD:
        case PIMCmdType::MUL:
        case PIMCmdType::MAC:
            cmd.dst = static_cast<PIMOpdType>(from_bits(value, 3, 25));
            cmd.src0 = static_cast<PIMOpdType>(from_bits(value, 3, 22));
            cmd.src1 = static_cast<PIMOpdType>(from_bits(value, 3, 19));
            cmd.is_auto = static_cast<int>(from_bits(value, 1, 15));
            cmd.dst_idx = static_cast<int>(from_bits(value, 4, 8));
            cmd.src0_idx = static_cast<int>(from_bits(value, 4, 4));
            cmd.src1_idx = static_cast<int>(from_bits(value, 4, 0));
            break;
        default:
            break;
    }
    return cmd;
}

uint32_t PIMCmd::to_int() const {
    validation_check();
    uint32_t value = to_bits(static_cast<uint32_t>(type), 4, 28);
    switch (type) {
        case PIMCmdType::EXIT:
            break;
        case PIMCmdType::NOP:
            value |= to_bits(static_cast<uint32_t>(loop_counter), 11, 0);
            break;
        case PIMCmdType::JUMP:
            value |= to_bits(static_cast<uint32_t>(loop_counter), 17, 11);
            value |= to_bits(static_cast<uint32_t>(loop_offset), 11, 0);
            break;
        case PIMCmdType::FILL:
        case PIMCmdType::MOV:
            value |= to_bits(static_cast<uint32_t>(dst), 3, 25);
            value |= to_bits(static_cast<uint32_t>(src0), 3, 22);
            value |= to_bits(static_cast<uint32_t>(dst_idx), 4, 8);
            value |= to_bits(static_cast<uint32_t>(src0_idx), 4, 4);
            value |= to_bits(static_cast<uint32_t>(src1_idx), 4, 0);
            value |= to_bits(static_cast<uint32_t>(is_relu), 1, 12);
            break;
        case PIMCmdType::MAD:
            value |= to_bits(static_cast<uint32_t>(src2), 3, 16);
            [[fallthrough]];
        case PIMCmdType::ADD:
        case PIMCmdType::MUL:
        case PIMCmdType::MAC:
            value |= to_bits(static_cast<uint32_t>(dst), 3, 25);
            value |= to_bits(static_cast<uint32_t>(src0), 3, 22);
            value |= to_bits(static_cast<uint32_t>(src1), 3, 19);
            value |= to_bits(static_cast<uint32_t>(is_auto), 1, 15);
            value |= to_bits(static_cast<uint32_t>(dst_idx), 4, 8);
            value |= to_bits(static_cast<uint32_t>(src0_idx), 4, 4);
            value |= to_bits(static_cast<uint32_t>(src1_idx), 4, 0);
            break;
        default:
            break;
    }
    return value;
}

void PIMCmd::validation_check() const {
    if (type == PIMCmdType::MOV || type == PIMCmdType::FILL) {
        const bool dst_is_bank = (dst == PIMOpdType::EVEN_BANK || dst == PIMOpdType::ODD_BANK);
        const bool src_reads_grf =
            (src0 == PIMOpdType::GRF_A || src0 == PIMOpdType::GRF_B ||
             src1 == PIMOpdType::GRF_A || src1 == PIMOpdType::GRF_B ||
             src2 == PIMOpdType::GRF_A || src2 == PIMOpdType::GRF_B);
        if (dst_is_bank && src_reads_grf) {
            throw std::runtime_error("Invalid PIM ISA 1.0 MOV/FILL operand combination");
        }
    }
}

std::string PIMCmd::to_string() const {
    std::ostringstream oss;
    oss << type_to_string(type) << ' ';
    switch (type) {
        case PIMCmdType::EXIT:
            break;
        case PIMCmdType::NOP:
            oss << (loop_counter + 1) << "x";
            break;
        case PIMCmdType::JUMP:
            oss << loop_counter << "x [PC - " << loop_offset << "]";
            break;
        case PIMCmdType::FILL:
        case PIMCmdType::MOV:
            oss << operand_to_string(dst, dst_idx) << ", "
                << operand_to_string(src0, src0_idx);
            if (is_relu) {
                oss << ", relu";
            }
            break;
        case PIMCmdType::ADD:
        case PIMCmdType::MUL:
        case PIMCmdType::MAC:
            oss << operand_to_string(dst, dst_idx) << ", "
                << operand_to_string(src0, src0_idx) << ", "
                << operand_to_string(src1, src1_idx);
            break;
        case PIMCmdType::MAD:
            oss << operand_to_string(dst, dst_idx) << ", "
                << operand_to_string(src0, src0_idx) << ", "
                << operand_to_string(src1, src1_idx) << ", "
                << operand_to_string(src2, src1_idx);
            break;
        default:
            break;
    }
    if (is_auto) {
        oss << ", auto";
    }
    return oss.str();
}

std::string PIMCmd::operand_to_string(PIMOpdType operand, int idx) {
    switch (operand) {
        case PIMOpdType::A_OUT:
            return "A_OUT";
        case PIMOpdType::M_OUT:
            return "M_OUT";
        case PIMOpdType::EVEN_BANK:
            return "EVEN_BANK";
        case PIMOpdType::ODD_BANK:
            return "ODD_BANK";
        case PIMOpdType::GRF_A:
            return "GRF_A[" + std::to_string(idx) + "]";
        case PIMOpdType::GRF_B:
            return "GRF_B[" + std::to_string(idx) + "]";
        case PIMOpdType::SRF_M:
            return "SRF_M[" + std::to_string(idx) + "]";
        case PIMOpdType::SRF_A:
            return "SRF_A[" + std::to_string(idx) + "]";
    }
    return "UNKNOWN";
}

std::string PIMCmd::type_to_string(PIMCmdType type) {
    switch (type) {
        case PIMCmdType::EXIT:
            return "EXIT";
        case PIMCmdType::NOP:
            return "NOP";
        case PIMCmdType::JUMP:
            return "JUMP";
        case PIMCmdType::FILL:
            return "FILL";
        case PIMCmdType::MOV:
            return "MOV";
        case PIMCmdType::ADD:
            return "ADD";
        case PIMCmdType::MUL:
            return "MUL";
        case PIMCmdType::MAC:
            return "MAC";
        case PIMCmdType::MAD:
            return "MAD";
        default:
            return "NOT_DEFINED";
    }
}

uint32_t PIMCmd::bitmask(int bit) {
    return (1u << bit) - 1u;
}

uint32_t PIMCmd::to_bits(uint32_t value, int bit_len, int bit_pos) {
    return ((value & bitmask(bit_len)) << bit_pos);
}

uint32_t PIMCmd::from_bits(uint32_t value, int bit_len, int bit_pos) {
    return ((value >> bit_pos) & bitmask(bit_len));
}

bool operator==(const PIMCmd& lhs, const PIMCmd& rhs) {
    return lhs.to_int() == rhs.to_int();
}

bool operator!=(const PIMCmd& lhs, const PIMCmd& rhs) {
    return !(lhs == rhs);
}

} // namespace aim_cosim
