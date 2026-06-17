// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include "hbmpim/pim_block_model.h"

#include <bit>
#include <cmath>
#include <cstdint>

namespace aim_cosim {

namespace {

uint32_t float_to_bits(float value) {
    return std::bit_cast<uint32_t>(value);
}

float bits_to_float(uint32_t bits) {
    return std::bit_cast<float>(bits);
}

} // namespace

PIMBlockModel::Burst PIMBlockModel::add(const Burst& src0, const Burst& src1) {
    Burst out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float_to_fp16(fp16_to_float(src0[i]) + fp16_to_float(src1[i]));
    }
    return out;
}

PIMBlockModel::Burst PIMBlockModel::mul(const Burst& src0, const Burst& src1) {
    Burst out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float_to_fp16(fp16_to_float(src0[i]) * fp16_to_float(src1[i]));
    }
    return out;
}

PIMBlockModel::Burst PIMBlockModel::mac(const Burst& dst,
                                        const Burst& src0,
                                        const Burst& src1) {
    Burst out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float_to_fp16(fp16_to_float(src0[i]) * fp16_to_float(src1[i]) +
                               fp16_to_float(dst[i]));
    }
    return out;
}

PIMBlockModel::Burst PIMBlockModel::mad(const Burst& src0,
                                        const Burst& src1,
                                        const Burst& src2) {
    Burst out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float_to_fp16(fp16_to_float(src0[i]) * fp16_to_float(src1[i]) +
                               fp16_to_float(src2[i]));
    }
    return out;
}

PIMBlockModel::Burst PIMBlockModel::mov(const Burst& src0, bool relu) {
    return relu ? apply_relu(src0) : src0;
}

PIMBlockModel::Burst PIMBlockModel::fill(const Burst& src0, bool relu) {
    return mov(src0, relu);
}

PIMBlockModel::Burst PIMBlockModel::broadcast(uint16_t value) {
    Burst out{};
    out.fill(value);
    return out;
}

float PIMBlockModel::fp16_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    const uint32_t exp = (value >> 10) & 0x1Fu;
    const uint32_t frac = value & 0x03FFu;

    if (exp == 0) {
        if (frac == 0) {
            return bits_to_float(sign);
        }
        float mant = static_cast<float>(frac) / 1024.0f;
        float result = std::ldexp(mant, -14);
        return (sign != 0) ? -result : result;
    }

    if (exp == 0x1F) {
        const uint32_t bits = sign | 0x7F800000u | (frac << 13);
        return bits_to_float(bits);
    }

    const uint32_t bits = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    return bits_to_float(bits);
}

uint16_t PIMBlockModel::float_to_fp16(float value) {
    const uint32_t bits = float_to_bits(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | 0x7E00u);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = 1u << (shift - 1u);
        if ((mant & round_bit) && ((mant & (round_bit - 1u)) || (half_mant & 1u))) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    uint16_t half_exp = static_cast<uint16_t>(exp << 10);
    uint16_t half_mant = static_cast<uint16_t>(mant >> 13);
    if (mant & 0x00001000u) {
        half_mant = static_cast<uint16_t>(half_mant + 1u);
        if (half_mant == 0x0400u) {
            half_mant = 0;
            half_exp = static_cast<uint16_t>(half_exp + 0x0400u);
            if (half_exp >= 0x7C00u) {
                half_exp = 0x7C00u;
            }
        }
    }

    return static_cast<uint16_t>(sign | half_exp | (half_mant & 0x03FFu));
}

PIMBlockModel::Burst PIMBlockModel::apply_relu(const Burst& input) {
    Burst out = input;
    for (auto& lane : out) {
        if ((lane & 0x8000u) != 0) {
            lane = 0;
        }
    }
    return out;
}

} // namespace aim_cosim
