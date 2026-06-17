#pragma once

// VerilatorEngine — Verilator DUT wrapper (ported from PIMPyVerilog SimEngine)
// Simplified for AiM-Cosim: no Python binding, no backend pull pattern,
// synchronous calls from VerilatorBridge

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#ifndef AIM_COSIM_ENABLE_VCD_TRACE
#define AIM_COSIM_ENABLE_VCD_TRACE 0
#endif

// Forward declarations for Verilator
#if ACC_MAG_WIDTH == 32
class Vpim_mac_tree_mag32;
using VpimMacTree = Vpim_mac_tree_mag32;
#elif ACC_MAG_WIDTH == 37
class Vpim_mac_tree_mag37;
using VpimMacTree = Vpim_mac_tree_mag37;
#else
class Vpim_mac_tree;
using VpimMacTree = Vpim_mac_tree;
#endif
class VerilatedVcdC;
class VerilatedContext;

namespace aim_cosim {

class VerilatorEngine {
public:
    VerilatorEngine();
    ~VerilatorEngine();

    // Disable copy
    VerilatorEngine(const VerilatorEngine&) = delete;
    VerilatorEngine& operator=(const VerilatorEngine&) = delete;

    // Lifecycle
    void reset();

    // MAC operation: weight[16] x vector[16] → accumulate → result
    struct ComputeResult {
        std::array<uint16_t, 16> data;
        bool success;
    };

    // Single-bank MAC: drive i_wgt_flat + i_vec_flat
    ComputeResult execute_mac(
        const uint16_t weight[16],
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        uint16_t bias
    );

    // 16-bank MAC: drive i_wgt_bank_flat + i_vec_flat + i_bias_bank_flat
    ComputeResult execute_mac16(
        const std::array<std::array<uint16_t, 16>, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]
    );
    ComputeResult execute_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]
    );

    uint64_t enqueue_mac16(
        const std::array<const uint16_t*, 16>& bank_weights,
        const uint16_t vector[16],
        bool acc_en,
        bool bias_en,
        const uint16_t bias[16]
    );
    ComputeResult flush_mac16_pipeline();
    bool has_pending_mac16() const { return pending_mac16_count_ > 0; }

    // Activation function
    ComputeResult execute_af(uint8_t af_type);

    // Element-wise multiply
    ComputeResult execute_ewmul(
        const uint16_t a[16],
        const uint16_t b[16]
    );

    // VCD trace
    void enable_trace(const std::string& path);
    void disable_trace();
    bool is_tracing() const;

    // Simulation time
    uint64_t get_time() const { return sim_time_; }

private:
    std::unique_ptr<VerilatedContext> context_;
    std::unique_ptr<VpimMacTree> dut_;
#if AIM_COSIM_ENABLE_VCD_TRACE
    std::unique_ptr<VerilatedVcdC> trace_;
#endif

    uint64_t sim_time_ = 0;
    bool clock_ = false;
    bool prev_done_ = false;
    size_t pending_mac16_count_ = 0;
    int cycles_since_last_mac16_start_ = 2;
    std::array<uint16_t, 16> last_mac16_result_{};
    bool last_mac16_result_valid_ = false;

    // Core simulation
    void clock_tick();
    void eval();
    void update_trace();
    bool observe_done_rising();
    void capture_mac16_done_if_ready();
    void drive_mac16_inputs(const std::array<const uint16_t*, 16>& bank_weights,
                            const uint16_t vector[16],
                            bool acc_en,
                            bool bias_en,
                            const uint16_t bias[16]);

    // Wait for o_done rising edge
    bool wait_for_done(int max_cycles = 10000);

    // BF16 unpacking (ported from PIMPyVerilog)
    static void unpack_words_to_bf16x16(const uint32_t src[8],
                                         uint16_t dst[16]);
};

} // namespace aim_cosim
