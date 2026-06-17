#include "controller/result_logger.h"
#include <charconv>
#include <filesystem>
#include <stdexcept>

namespace aim_cosim {

namespace {

template <typename T>
void append_decimal(std::string& out, T value) {
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value);
    if (ec == std::errc{}) {
        out.append(buf, ptr);
    }
}

void append_hex16(std::string& out, uint16_t value) {
    char buf[4];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value, 16);
    if (ec == std::errc{}) {
        out.append(buf, ptr);
    }
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path file_path(path);
    const auto parent = file_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

void ResultLogger::init(int channel_id, const std::string& result_log_path,
                        const ResultLoggerConfig& config) {
    if (config.schema_version != ResultLoggerConfig::CURRENT_SCHEMA_VERSION) {
        throw std::runtime_error("Unsupported ResultLogger schema_version");
    }

    m_config = config;
    m_buffer_size = m_config.buffer_size;
    m_result_buffer.reserve(m_buffer_size * 128);
    m_timing_buffer.reserve(m_buffer_size * 96);

    // Result CSV
    if (!result_log_path.empty() &&
        m_config.result_level == ResultLogLevel::FULL) {
        std::string path = result_log_path;
        auto dot_pos = path.rfind('.');
        if (dot_pos != std::string::npos) {
            path = path.substr(0, dot_pos) + "_ch" + std::to_string(channel_id) + path.substr(dot_pos);
        } else {
            path += "_ch" + std::to_string(channel_id);
        }
        ensure_parent_directory(path);
        m_result_log.open(path, std::ios::out | std::ios::trunc);
        if (m_result_log.is_open()) {
            m_result_log << "cycle,command,bank_id,row,col,data_hex\n";
        }
    }

    // Timing CSV
    if (m_config.timing_level != TimingLogLevel::OFF) {
        std::filesystem::path timing_path = result_log_path.empty()
            ? std::filesystem::path("result")
            : std::filesystem::path(result_log_path).parent_path();
        if (timing_path.empty()) {
            timing_path = "result";
        }
        timing_path /= "rtl_timing_ch" + std::to_string(channel_id) + ".csv";
        ensure_parent_directory(timing_path.string());
        m_timing_log.open(timing_path, std::ios::out | std::ios::trunc);
        if (m_timing_log.is_open()) {
            m_timing_log << "sim_issue_cycle,command,rtl_cycles,min_constraint,dram_elapsed_to_consumer,diverged\n";
        }
    }
}

void ResultLogger::log_result(const ResultPacket& pkt) {
    if (!m_result_log.is_open()) return;

    append_decimal(m_result_buffer, pkt.cycle);
    m_result_buffer.push_back(',');
    append_decimal(m_result_buffer, pkt.command);
    m_result_buffer.push_back(',');
    append_decimal(m_result_buffer, pkt.bank_id);
    m_result_buffer.push_back(',');
    append_decimal(m_result_buffer, pkt.row);
    m_result_buffer.push_back(',');
    append_decimal(m_result_buffer, pkt.col);
    m_result_buffer.push_back(',');
    for (int i = 0; i < 16; ++i) {
        if (i > 0) m_result_buffer.push_back(' ');
        append_hex16(m_result_buffer, pkt.data[i]);
    }
    m_result_buffer.push_back('\n');
    ++m_result_buffer_rows;

    if (m_result_buffer_rows >= m_buffer_size) {
        flush_result_buffer();
    }
}

void ResultLogger::log_timing(const DivergenceResult& dr) {
    if (!dr.logged || !m_timing_log.is_open() || !should_log_timing(dr)) return;

    append_decimal(m_timing_buffer, dr.sim_issue_cycle);
    m_timing_buffer.push_back(',');
    m_timing_buffer.append(dr.command_name);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, dr.rtl_cycles);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, dr.min_constraint);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, dr.dram_elapsed);
    m_timing_buffer.push_back(',');
    m_timing_buffer.append(dr.diverged ? "true" : "false");
    m_timing_buffer.push_back('\n');
    ++m_timing_buffer_rows;

    if (m_timing_buffer_rows >= m_buffer_size) {
        flush_timing_buffer();
    }
}

void ResultLogger::log_compute_timing(const std::string& command_name,
                                      int64_t sim_issue_cycle,
                                      uint64_t rtl_cycles,
                                      int64_t dram_min_constraint) {
    if (!m_timing_log.is_open() || rtl_cycles == 0) {
        return;
    }

    const bool diverged = static_cast<int64_t>(rtl_cycles) > dram_min_constraint;
    if (m_config.timing_level == TimingLogLevel::OFF) {
        return;
    }
    if (m_config.timing_level == TimingLogLevel::SUMMARY && !diverged) {
        return;
    }

    append_decimal(m_timing_buffer, sim_issue_cycle);
    m_timing_buffer.push_back(',');
    m_timing_buffer.append(command_name);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, rtl_cycles);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, dram_min_constraint);
    m_timing_buffer.push_back(',');
    append_decimal(m_timing_buffer, rtl_cycles);
    m_timing_buffer.push_back(',');
    m_timing_buffer.append(diverged ? "true" : "false");
    m_timing_buffer.push_back('\n');
    ++m_timing_buffer_rows;

    if (m_timing_buffer_rows >= m_buffer_size) {
        flush_timing_buffer();
    }
}

void ResultLogger::log_ewmul_timing(int64_t sim_issue_cycle, uint64_t rtl_cycles) {
    if (!should_log_ewmul_timing(rtl_cycles)) return;
    log_compute_timing("EWMUL16", sim_issue_cycle, rtl_cycles, /*dram_min_constraint=*/0);
}

void ResultLogger::flush() {
    flush_result_buffer();
    flush_timing_buffer();
    if (m_result_log.is_open()) {
        m_result_log.flush();
    }
    if (m_timing_log.is_open()) {
        m_timing_log.flush();
    }
}

void ResultLogger::flush_result_buffer() {
    if (m_result_buffer.empty() || !m_result_log.is_open()) return;
    m_result_log.write(m_result_buffer.data(), static_cast<std::streamsize>(m_result_buffer.size()));
    m_result_buffer.clear();
    m_result_buffer_rows = 0;
}

void ResultLogger::flush_timing_buffer() {
    if (m_timing_buffer.empty() || !m_timing_log.is_open()) return;
    m_timing_log.write(m_timing_buffer.data(), static_cast<std::streamsize>(m_timing_buffer.size()));
    m_timing_buffer.clear();
    m_timing_buffer_rows = 0;
}

ResultLogger::~ResultLogger() {
    flush();
}

bool ResultLogger::should_log_timing(const DivergenceResult& dr) const {
    switch (m_config.timing_level) {
        case TimingLogLevel::FULL:
            return true;
        case TimingLogLevel::SUMMARY:
            return dr.diverged || dr.deferred_cycles > 0 || dr.fallback_flush;
        case TimingLogLevel::OFF:
            return false;
    }
    return false;
}

bool ResultLogger::should_log_ewmul_timing(uint64_t rtl_cycles) const {
    if (rtl_cycles == 0) {
        return false;
    }
    return m_config.timing_level == TimingLogLevel::FULL;
}

} // namespace aim_cosim
