#include "verilator_engine/verilator_engine.h"

#if ACC_MAG_WIDTH == 32
#include "Vpim_mac_tree_mag32.h"
#elif ACC_MAG_WIDTH == 37
#include "Vpim_mac_tree_mag37.h"
#else
#include "Vpim_mac_tree.h"
#endif
#include <verilated.h>
#if AIM_COSIM_ENABLE_VCD_TRACE
#include <verilated_vcd_c.h>
#endif

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace aim_cosim {

namespace {

inline uint32_t pack_bf16_pair(const uint16_t* src, int index) {
    return static_cast<uint32_t>(src[index])
         | (static_cast<uint32_t>(src[index + 1]) << 16);
}

// The RTL command clock is 500 ps in the GDDR6-AiM model, so two command
// cycles correspond to the 1 ns tCCDS MAC cadence. AF is drained separately.
constexpr int kMac16StartSpacingCycles = 2;
constexpr uint64_t kMac16LatencyCycles = 4;

} // namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

VerilatorEngine::VerilatorEngine()
    : context_(std::make_unique<VerilatedContext>()),
      dut_(std::make_unique<VpimMacTree>(context_.get()))
{
#if AIM_COSIM_ENABLE_VCD_TRACE
    context_->traceEverOn(true);
#endif

    // Initialize all DUT signals to known state
    dut_->clk = 0;
    dut_->rst_n = 1;
    dut_->i_start = 0;
    dut_->i_acc_en = 0;
    dut_->i_ewmul_en = 0;
    dut_->i_bias_en = 0;
    dut_->i_bias = 0;
    dut_->i_latch_sel_valid = 0;
    dut_->i_latch_sel = 0;
    dut_->i_latch_seed_valid = 0;
    dut_->i_latch_seed = 0;
    dut_->i_af_en = 0;
    dut_->i_af_type = 0;
    dut_->i_af_slope_bf16 = 0;
    dut_->i_all_bank_vector_en = 0;

    for (int i = 0; i < 8; ++i) {
        dut_->i_wgt_flat[i] = 0;
        dut_->i_vec_flat[i] = 0;
        dut_->i_bias_bank_flat[i] = 0;
    }
    for (int i = 0; i < 128; ++i) {
        dut_->i_wgt_bank_flat[i] = 0;
    }

    dut_->eval();
}

VerilatorEngine::~VerilatorEngine() {
    disable_trace();
}

// =============================================================================
// Lifecycle
// =============================================================================

void VerilatorEngine::reset() {
    // Assert active-low reset
    dut_->rst_n = 0;
    dut_->i_start = 0;
    dut_->i_acc_en = 0;
    dut_->i_ewmul_en = 0;
    dut_->i_bias_en = 0;
    dut_->i_bias = 0;
    dut_->i_af_en = 0;
    dut_->i_af_type = 0;
    dut_->i_af_slope_bf16 = 0;
    dut_->i_all_bank_vector_en = 0;

    for (int i = 0; i < 8; ++i) {
        dut_->i_bias_bank_flat[i] = 0;
    }
    for (int i = 0; i < 128; ++i) {
        dut_->i_wgt_bank_flat[i] = 0;
    }

    // Hold reset for 10 cycles
    for (int i = 0; i < 10; ++i) {
        clock_tick();
    }

    // Release reset
    dut_->rst_n = 1;
    for (int i = 0; i < 2; ++i) {
        clock_tick();
    }

    sim_time_ = 0;
    prev_done_ = false;
    pending_mac16_count_ = 0;
    cycles_since_last_mac16_start_ = kMac16StartSpacingCycles;
    last_mac16_result_.fill(0);
    last_mac16_result_valid_ = false;
}

// =============================================================================
// Core Simulation
// =============================================================================

void VerilatorEngine::clock_tick() {
    // Rising edge
    dut_->clk = 1;
    eval();
    update_trace();
    context_->timeInc(1);
    sim_time_++;

    // Falling edge
    dut_->clk = 0;
    eval();
    update_trace();
    context_->timeInc(1);
}

void VerilatorEngine::eval() {
    dut_->eval();
}

void VerilatorEngine::update_trace() {
#if AIM_COSIM_ENABLE_VCD_TRACE
    if (trace_) {
        trace_->dump(context_->time());
    }
#endif
}

bool VerilatorEngine::observe_done_rising() {
    const bool done_now = (dut_->o_done != 0);
    const bool rising = done_now && !prev_done_;
    prev_done_ = done_now;
    return rising;
}

void VerilatorEngine::capture_mac16_done_if_ready() {
    if (!observe_done_rising()) {
        return;
    }
    if (pending_mac16_count_ == 0) {
        return;
    }

    unpack_words_to_bf16x16(dut_->o_mac_result_flat, last_mac16_result_.data());
    last_mac16_result_valid_ = true;
    --pending_mac16_count_;
}

void VerilatorEngine::drive_mac16_inputs(
    const std::array<const uint16_t*, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en, const uint16_t bias[16]) {
    for (int bank = 0; bank < 16; ++bank) {
        const int base = bank * 8;
        for (int i = 0; i < 8; ++i) {
            dut_->i_wgt_bank_flat[base + i] = pack_bf16_pair(bank_weights[bank], 2 * i);
        }
    }

    for (int i = 0; i < 8; ++i) {
        const int lane = 2 * i;
        dut_->i_vec_flat[i] = pack_bf16_pair(vector, lane);
        dut_->i_bias_bank_flat[i] = pack_bf16_pair(bias, lane);
    }

    dut_->i_acc_en = acc_en ? 1 : 0;
    dut_->i_bias_en = bias_en ? 1 : 0;
    dut_->i_ewmul_en = 0;
    dut_->i_af_en = 0;
    dut_->i_all_bank_vector_en = 1;
}

bool VerilatorEngine::wait_for_done(int max_cycles) {
    for (int i = 0; i < max_cycles; ++i) {
        clock_tick();
        if (observe_done_rising()) {
            return true;  // Rising edge of o_done
        }
    }
    return false;  // Timeout
}

// =============================================================================
// Compute Operations
// =============================================================================

VerilatorEngine::ComputeResult
VerilatorEngine::execute_mac(const uint16_t weight[16],
                              const uint16_t vector[16],
                              bool acc_en, bool bias_en, uint16_t bias) {
    flush_mac16_pipeline();

    ComputeResult result{};

    for (int i = 0; i < 8; ++i) {
        const int lane = 2 * i;
        dut_->i_wgt_flat[i] = pack_bf16_pair(weight, lane);
        dut_->i_vec_flat[i] = pack_bf16_pair(vector, lane);
    }

    dut_->i_acc_en = acc_en ? 1 : 0;
    dut_->i_bias_en = bias_en ? 1 : 0;
    dut_->i_bias = bias;
    dut_->i_ewmul_en = 0;
    dut_->i_af_en = 0;
    dut_->i_all_bank_vector_en = 0;

    // Assert start for one cycle
    dut_->i_start = 1;
    clock_tick();
    dut_->i_start = 0;

    // Wait for completion
    result.success = wait_for_done();
    if (result.success) {
        unpack_words_to_bf16x16(dut_->o_mac_result_flat, result.data.data());
    }

    return result;
}

VerilatorEngine::ComputeResult
VerilatorEngine::execute_mac16(
    const std::array<std::array<uint16_t, 16>, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en, const uint16_t bias[16]) {
    std::array<const uint16_t*, 16> bank_weight_ptrs{};
    for (int bank = 0; bank < 16; ++bank) {
        bank_weight_ptrs[bank] = bank_weights[bank].data();
    }
    return execute_mac16(bank_weight_ptrs, vector, acc_en, bias_en, bias);
}

VerilatorEngine::ComputeResult
VerilatorEngine::execute_mac16(
    const std::array<const uint16_t*, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en, const uint16_t bias[16]) {
    flush_mac16_pipeline();

    ComputeResult result{};

    drive_mac16_inputs(bank_weights, vector, acc_en, bias_en, bias);

    // Assert start for one cycle
    dut_->i_start = 1;
    clock_tick();
    dut_->i_start = 0;

    // Wait for completion
    result.success = wait_for_done();
    if (result.success) {
        unpack_words_to_bf16x16(dut_->o_mac_result_flat, result.data.data());
    }

    return result;
}

uint64_t VerilatorEngine::enqueue_mac16(
    const std::array<const uint16_t*, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en, const uint16_t bias[16]) {
    while (cycles_since_last_mac16_start_ < kMac16StartSpacingCycles - 1) {
        dut_->i_start = 0;
        clock_tick();
        capture_mac16_done_if_ready();
        ++cycles_since_last_mac16_start_;
    }

    drive_mac16_inputs(bank_weights, vector, acc_en, bias_en, bias);

    dut_->i_start = 1;
    clock_tick();
    capture_mac16_done_if_ready();
    dut_->i_start = 0;

    ++pending_mac16_count_;
    cycles_since_last_mac16_start_ = 0;
    return kMac16LatencyCycles;
}

VerilatorEngine::ComputeResult VerilatorEngine::flush_mac16_pipeline() {
    ComputeResult result{};
    int waited_cycles = 0;
    while (pending_mac16_count_ > 0 && waited_cycles < 10000) {
        dut_->i_start = 0;
        clock_tick();
        capture_mac16_done_if_ready();
        ++cycles_since_last_mac16_start_;
        ++waited_cycles;
    }

    result.success = pending_mac16_count_ == 0 && last_mac16_result_valid_;
    if (result.success) {
        result.data = last_mac16_result_;
    }
    return result;
}

VerilatorEngine::ComputeResult
VerilatorEngine::execute_af(uint8_t af_type) {
    flush_mac16_pipeline();

    ComputeResult result{};

    dut_->i_af_en = 1;
    dut_->i_af_type = af_type;
    dut_->i_start = 0;
    dut_->i_ewmul_en = 0;

    // Assert start for one cycle
    dut_->i_start = 1;
    clock_tick();
    dut_->i_start = 0;

    // Wait for completion
    result.success = wait_for_done();
    if (result.success) {
        unpack_words_to_bf16x16(dut_->o_af_result_flat, result.data.data());
    }

    dut_->i_af_en = 0;
    return result;
}

VerilatorEngine::ComputeResult
VerilatorEngine::execute_ewmul(const uint16_t a[16], const uint16_t b[16]) {
    flush_mac16_pipeline();

    ComputeResult result{};

    for (int i = 0; i < 8; ++i) {
        const int lane = 2 * i;
        dut_->i_wgt_flat[i] = pack_bf16_pair(a, lane);
        dut_->i_vec_flat[i] = pack_bf16_pair(b, lane);
    }

    dut_->i_ewmul_en = 1;
    dut_->i_acc_en = 0;
    dut_->i_bias_en = 0;
    dut_->i_af_en = 0;

    // Assert start for one cycle
    dut_->i_start = 1;
    clock_tick();
    dut_->i_start = 0;

    // Wait for completion
    result.success = wait_for_done();
    if (result.success) {
        unpack_words_to_bf16x16(dut_->o_result_flat, result.data.data());
    }

    dut_->i_ewmul_en = 0;
    return result;
}

// =============================================================================
// VCD Tracing
// =============================================================================

void VerilatorEngine::enable_trace(const std::string& path) {
#if AIM_COSIM_ENABLE_VCD_TRACE
    if (!trace_) {
        trace_ = std::make_unique<VerilatedVcdC>();
        dut_->trace(trace_.get(), 99);
        trace_->open(path.c_str());
    }
#else
    (void)path;
    throw std::runtime_error(
        "VCD tracing was requested, but this build was configured with "
        "AIM_COSIM_ENABLE_VCD_TRACE=OFF. Reconfigure with "
        "-DAIM_COSIM_ENABLE_VCD_TRACE=ON to enable waveform dumping.");
#endif
}

void VerilatorEngine::disable_trace() {
#if AIM_COSIM_ENABLE_VCD_TRACE
    if (trace_) {
        trace_->close();
        trace_.reset();
    }
#endif
}

bool VerilatorEngine::is_tracing() const {
#if AIM_COSIM_ENABLE_VCD_TRACE
    return trace_ != nullptr;
#else
    return false;
#endif
}

// =============================================================================
// BF16 Packing / Unpacking
// =============================================================================

void VerilatorEngine::unpack_words_to_bf16x16(const uint32_t src[8],
                                                uint16_t dst[16]) {
    for (int i = 0; i < 8; ++i) {
        const uint32_t word = src[i];
        dst[2 * i] = static_cast<uint16_t>(word & 0xFFFFu);
        dst[2 * i + 1] = static_cast<uint16_t>((word >> 16) & 0xFFFFu);
    }
}

} // namespace aim_cosim
