#pragma once

// IComputeEngine — Abstract interface for PIM compute backends
// All implementations use Verilator RTL DUT for bit-exact verification.
// See doc9 Phase 5 for design rationale.

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace aim_cosim {

class IComputeEngine {
public:
    virtual ~IComputeEngine() = default;

    struct ComputeResult {
        std::array<uint16_t, 16> data;
        uint64_t rtl_cycles;  // RTL pipeline latency (measured per command)
        bool success;
    };

    // 16-bank parallel MAC
    virtual ComputeResult execute_mac16(
        const std::array<std::array<uint16_t, 16>, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]) = 0;

    virtual ComputeResult execute_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]) {
        std::array<std::array<uint16_t, 16>, 16> staged{};
        for (int b = 0; b < 16; ++b) {
            std::memcpy(staged[b].data(), bank_weights[b], 16 * sizeof(uint16_t));
        }
        return execute_mac16(staged, vector, acc_en, bias_en, bias);
    }

    virtual bool supports_deferred_mac16() const { return false; }
    virtual bool has_pending_mac16() const { return false; }
    virtual ComputeResult enqueue_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]) {
        return execute_mac16(bank_weights, vector, acc_en, bias_en, bias);
    }
    virtual ComputeResult flush_mac16_pipeline() { return ComputeResult{}; }

    // Activation function on MAC results
    virtual ComputeResult execute_af(uint8_t af_type) = 0;

    // Element-wise multiply
    virtual ComputeResult execute_ewmul(
        const uint16_t a[16],
        const uint16_t b[16]) = 0;

    // Single-bank MAC
    virtual ComputeResult execute_mac(
        const uint16_t weight[16],
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        uint16_t bias) = 0;

    virtual void reset() = 0;

    // VCD trace control (optional — default no-op)
    virtual void enable_trace(const std::string& /*path*/) {}
    virtual void disable_trace() {}
    virtual bool is_tracing() const { return false; }

    // Simulation time (optional — default 0)
    virtual uint64_t get_time() const { return 0; }
};

} // namespace aim_cosim
