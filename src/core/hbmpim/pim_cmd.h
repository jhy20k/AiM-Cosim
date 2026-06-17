#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include <cstdint>
#include <string>

namespace aim_cosim {

enum class PIMCmdType : uint8_t {
    NOP = 0,
    ADD = 1,
    MUL = 2,
    MAC = 3,
    MAD = 4,
    REV0 = 5,
    REV1 = 6,
    REV2 = 7,
    MOV = 8,
    FILL = 9,
    REV3 = 10,
    REV4 = 11,
    REV5 = 12,
    REV6 = 13,
    JUMP = 14,
    EXIT = 15,
};

enum class PIMOpdType : uint8_t {
    A_OUT = 0,
    M_OUT = 1,
    EVEN_BANK = 2,
    ODD_BANK = 3,
    GRF_A = 4,
    GRF_B = 5,
    SRF_M = 6,
    SRF_A = 7,
};

class PIMCmd {
public:
    PIMCmdType type = PIMCmdType::NOP;
    PIMOpdType dst = PIMOpdType::A_OUT;
    PIMOpdType src0 = PIMOpdType::A_OUT;
    PIMOpdType src1 = PIMOpdType::A_OUT;
    PIMOpdType src2 = PIMOpdType::A_OUT;
    int loop_counter = 0;
    int loop_offset = 0;
    int is_auto = 0;
    int dst_idx = 0;
    int src0_idx = 0;
    int src1_idx = 0;
    int is_relu = 0;

    PIMCmd() = default;
    explicit PIMCmd(PIMCmdType type_value) : type(type_value) {}
    PIMCmd(PIMCmdType type_value, int loop_counter_value);
    PIMCmd(PIMCmdType type_value, int loop_counter_value, int loop_offset_value);
    PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
           int is_auto_value = 0, int dst_idx_value = 0, int src0_idx_value = 0,
           int src1_idx_value = 0, int is_relu_value = 0);
    PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
           PIMOpdType src1_value, int is_auto_value = 0, int dst_idx_value = 0,
           int src0_idx_value = 0, int src1_idx_value = 0);
    PIMCmd(PIMCmdType type_value, PIMOpdType dst_value, PIMOpdType src0_value,
           PIMOpdType src1_value, PIMOpdType src2_value, int is_auto_value = 0,
           int dst_idx_value = 0, int src0_idx_value = 0, int src1_idx_value = 0);

    static PIMCmd from_int(uint32_t value);
    uint32_t to_int() const;

    void validation_check() const;
    std::string to_string() const;

    static std::string operand_to_string(PIMOpdType operand, int idx = 0);
    static std::string type_to_string(PIMCmdType type);

private:
    static uint32_t bitmask(int bit);
    static uint32_t to_bits(uint32_t value, int bit_len, int bit_pos);
    static uint32_t from_bits(uint32_t value, int bit_len, int bit_pos);
};

bool operator==(const PIMCmd& lhs, const PIMCmd& rhs);
bool operator!=(const PIMCmd& lhs, const PIMCmd& rhs);

} // namespace aim_cosim
