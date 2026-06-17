#include "compute_engine/aim_compute_engine.h"

#include <algorithm>

namespace aim_cosim {

AiMComputeEngine::AiMComputeEngine() = default;
AiMComputeEngine::~AiMComputeEngine() = default;

void AiMComputeEngine::reset() {
    engine_.reset();
}

// =============================================================================
// Compute operations — delegate to VerilatorEngine with rtl_cycles measurement
// =============================================================================

IComputeEngine::ComputeResult
AiMComputeEngine::execute_mac16(
    const std::array<std::array<uint16_t, 16>, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en,
    const uint16_t bias[16]) {

    uint64_t start = engine_.get_time();
    auto raw = engine_.execute_mac16(bank_weights, vector, acc_en, bias_en, bias);
    uint64_t end = engine_.get_time();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = apply_override(end - start, m_mac16_cycle_override);
    return result;
}

IComputeEngine::ComputeResult
AiMComputeEngine::execute_mac16(
    const std::array<const uint16_t*, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en,
    const uint16_t bias[16]) {

    uint64_t start = engine_.get_time();
    auto raw = engine_.execute_mac16(bank_weights, vector, acc_en, bias_en, bias);
    uint64_t end = engine_.get_time();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = apply_override(end - start, m_mac16_cycle_override);
    return result;
}

bool AiMComputeEngine::has_pending_mac16() const {
    return engine_.has_pending_mac16();
}

IComputeEngine::ComputeResult
AiMComputeEngine::enqueue_mac16(
    const std::array<const uint16_t*, 16>& bank_weights,
    const uint16_t vector[16],
    bool acc_en, bool bias_en,
    const uint16_t bias[16]) {

    const uint64_t rtl_cycles =
        apply_override(engine_.enqueue_mac16(bank_weights, vector, acc_en, bias_en, bias),
                       m_mac16_cycle_override);

    ComputeResult result{};
    result.rtl_cycles = rtl_cycles;
    result.success = false;
    return result;
}

IComputeEngine::ComputeResult AiMComputeEngine::flush_mac16_pipeline() {
    auto raw = engine_.flush_mac16_pipeline();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = 0;
    return result;
}

IComputeEngine::ComputeResult
AiMComputeEngine::execute_af(uint8_t af_type) {

    uint64_t start = engine_.get_time();
    auto raw = engine_.execute_af(af_type);
    uint64_t end = engine_.get_time();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = apply_override(end - start, m_af16_cycle_override);
    return result;
}

IComputeEngine::ComputeResult
AiMComputeEngine::execute_ewmul(const uint16_t a[16], const uint16_t b[16]) {

    uint64_t start = engine_.get_time();
    auto raw = engine_.execute_ewmul(a, b);
    uint64_t end = engine_.get_time();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = apply_override(end - start, m_ewmul16_cycle_override);
    return result;
}

IComputeEngine::ComputeResult
AiMComputeEngine::execute_mac(
    const uint16_t weight[16],
    const uint16_t vector[16],
    bool acc_en, bool bias_en, uint16_t bias) {

    uint64_t start = engine_.get_time();
    auto raw = engine_.execute_mac(weight, vector, acc_en, bias_en, bias);
    uint64_t end = engine_.get_time();

    ComputeResult result{};
    result.data = raw.data;
    result.success = raw.success;
    result.rtl_cycles = apply_override(end - start, m_mac_cycle_override);
    return result;
}

// =============================================================================
// VCD trace
// =============================================================================

void AiMComputeEngine::enable_trace(const std::string& path) {
    engine_.enable_trace(path);
}

void AiMComputeEngine::disable_trace() {
    engine_.disable_trace();
}

bool AiMComputeEngine::is_tracing() const {
    return engine_.is_tracing();
}

uint64_t AiMComputeEngine::get_time() const {
    return engine_.get_time();
}

void AiMComputeEngine::set_mac_cycle_override(uint64_t cycles) {
    m_mac_cycle_override = cycles;
}

void AiMComputeEngine::set_mac16_cycle_override(uint64_t cycles) {
    m_mac16_cycle_override = cycles;
}

void AiMComputeEngine::set_af16_cycle_override(uint64_t cycles) {
    m_af16_cycle_override = cycles;
}

void AiMComputeEngine::set_ewmul16_cycle_override(uint64_t cycles) {
    m_ewmul16_cycle_override = cycles;
}

uint64_t AiMComputeEngine::apply_override(uint64_t measured_cycles,
                                          uint64_t override_cycles) const {
    if (override_cycles == 0) {
        return measured_cycles;
    }
    return std::max(measured_cycles, override_cycles);
}

} // namespace aim_cosim
