#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include <array>
#include <cstdint>

namespace aim_cosim {

class PIMBlockModel {
public:
    using Burst = std::array<uint16_t, 16>;

    static Burst add(const Burst& src0, const Burst& src1);
    static Burst mul(const Burst& src0, const Burst& src1);
    static Burst mac(const Burst& dst, const Burst& src0, const Burst& src1);
    static Burst mad(const Burst& src0, const Burst& src1, const Burst& src2);
    static Burst mov(const Burst& src0, bool relu = false);
    static Burst fill(const Burst& src0, bool relu = false);
    static Burst broadcast(uint16_t value);

private:
    static float fp16_to_float(uint16_t value);
    static uint16_t float_to_fp16(float value);
    static Burst apply_relu(const Burst& input);
};

} // namespace aim_cosim
