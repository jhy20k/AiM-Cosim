#pragma once

// AiMComputeEngine — GDDR6-AiM RTL compute backend (IComputeEngine implementation)
// Wraps VerilatorEngine (pim_mac_tree DUT) with rtl_cycles measurement.
// See doc9 Phase 5 for design rationale.

#include "compute_engine/compute_engine.h"
#include "verilator_engine/verilator_engine.h"

#include <cstdint>
#include <memory>
#include <string>

namespace aim_cosim {

class AiMComputeEngine final : public IComputeEngine {
public:
    AiMComputeEngine();
    ~AiMComputeEngine() override;

    // IComputeEngine interface
    ComputeResult execute_mac16(
        const std::array<std::array<uint16_t, 16>, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en, bool bias_en,
        const uint16_t bias[16]) override;

    ComputeResult execute_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en, bool bias_en,
        const uint16_t bias[16]) override;

    bool supports_deferred_mac16() const override { return true; }
    bool has_pending_mac16() const override;
    ComputeResult enqueue_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en, bool bias_en,
        const uint16_t bias[16]) override;
    ComputeResult flush_mac16_pipeline() override;

    ComputeResult execute_af(uint8_t af_type) override;

    ComputeResult execute_ewmul(
        const uint16_t a[16],
        const uint16_t b[16]) override;

    ComputeResult execute_mac(
        const uint16_t weight[16],
        const uint16_t vector[16],
        bool acc_en, bool bias_en,
        uint16_t bias) override;

    void reset() override;

    // VCD trace control
    void enable_trace(const std::string& path) override;
    void disable_trace() override;
    bool is_tracing() const override;

    // Simulation time
    uint64_t get_time() const override;

    // Optional debug/validation override. When non-zero, the reported
    // rtl_cycles is clamped to at least this value.
    void set_mac_cycle_override(uint64_t cycles);
    void set_mac16_cycle_override(uint64_t cycles);
    void set_af16_cycle_override(uint64_t cycles);
    void set_ewmul16_cycle_override(uint64_t cycles);

private:
    uint64_t apply_override(uint64_t measured_cycles, uint64_t override_cycles) const;

    VerilatorEngine engine_;
    uint64_t m_mac_cycle_override = 0;
    uint64_t m_mac16_cycle_override = 0;
    uint64_t m_af16_cycle_override = 0;
    uint64_t m_ewmul16_cycle_override = 0;
};

} // namespace aim_cosim
