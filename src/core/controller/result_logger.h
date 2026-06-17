#pragma once
// =============================================================================
// ResultLogger — Buffered CSV writer for functional results + RTL timing
//
// Owns:
//   - <configured result dir>/rtl_results_ch*.csv (functional result data)
//   - <configured result dir>/rtl_timing_ch*.csv  (RTL timing divergence data)
//   - Buffered write (configurable batch size, default 4096 rows)
//   - Flush on buffer full or destruction
// =============================================================================

#include "controller/divergence_tracker.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <string>

namespace aim_cosim {

struct ResultPacket {
    uint64_t cycle;
    int command;
    int bank_id;
    int row, col;
    std::array<uint16_t, 16> data;
};

enum class ResultLogLevel : uint8_t {
    FULL,
    OFF,
};

enum class TimingLogLevel : uint8_t {
    FULL,
    SUMMARY,
    OFF,
};

struct ResultLoggerConfig {
    static constexpr uint32_t CURRENT_SCHEMA_VERSION = 1;

    size_t buffer_size = 4096;
    ResultLogLevel result_level = ResultLogLevel::FULL;
    TimingLogLevel timing_level = TimingLogLevel::FULL;
    bool rdaf16_bank0_only = false;
    uint32_t schema_version = CURRENT_SCHEMA_VERSION;
};

class ResultLogger {
public:
    static constexpr size_t DEFAULT_BUFFER_SIZE = 4096;

    void init(int channel_id, const std::string& result_log_path,
              const ResultLoggerConfig& config = {});

    // Functional result logging
    void log_result(const ResultPacket& pkt);

    // RTL timing logging (from DivergenceTracker results)
    void log_timing(const DivergenceResult& dr);

    // Direct compute timing logging (controller scoreboard path)
    void log_compute_timing(const std::string& command_name,
                            int64_t sim_issue_cycle,
                            uint64_t rtl_cycles,
                            int64_t dram_min_constraint);

    // Log EWMUL timing directly (no consumer, inline writeback)
    void log_ewmul_timing(int64_t sim_issue_cycle, uint64_t rtl_cycles);

    // Force flush all buffered data
    void flush();

    ~ResultLogger();

    bool timing_enabled() const { return m_timing_log.is_open(); }
    bool result_enabled() const { return m_result_log.is_open(); }
    const ResultLoggerConfig& config() const { return m_config; }

private:
    bool should_log_timing(const DivergenceResult& dr) const;
    bool should_log_ewmul_timing(uint64_t rtl_cycles) const;
    void flush_result_buffer();
    void flush_timing_buffer();

    std::ofstream m_result_log;
    std::ofstream m_timing_log;
    std::string m_result_buffer;
    std::string m_timing_buffer;
    size_t m_result_buffer_rows = 0;
    size_t m_timing_buffer_rows = 0;
    size_t m_buffer_size = DEFAULT_BUFFER_SIZE;
    ResultLoggerConfig m_config{};
};

} // namespace aim_cosim
