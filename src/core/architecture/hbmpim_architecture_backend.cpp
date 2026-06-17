// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include "architecture/hbmpim_architecture_backend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

namespace aim_cosim {

namespace {

std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

ResultLogLevel parse_result_log_level(const YAML::Node& logging_cfg) {
    if (!logging_cfg || !logging_cfg["result_level"]) {
        return ResultLogLevel::FULL;
    }

    const auto value = lowercase_copy(logging_cfg["result_level"].as<std::string>("full"));
    if (value == "full") {
        return ResultLogLevel::FULL;
    }
    if (value == "off") {
        return ResultLogLevel::OFF;
    }
    throw std::runtime_error("Unsupported result_level in HBM-PIM logging config");
}

TimingLogLevel parse_timing_log_level(const YAML::Node& logging_cfg) {
    if (!logging_cfg || !logging_cfg["timing_level"]) {
        return TimingLogLevel::OFF;
    }

    const auto value = lowercase_copy(logging_cfg["timing_level"].as<std::string>("off"));
    if (value == "full") {
        return TimingLogLevel::FULL;
    }
    if (value == "summary") {
        return TimingLogLevel::SUMMARY;
    }
    if (value == "off") {
        return TimingLogLevel::OFF;
    }
    throw std::runtime_error("Unsupported timing_level in HBM-PIM logging config");
}

ResultLoggerConfig parse_hbmpim_logger_config(const YAML::Node& root_cfg,
                                              const YAML::Node& verilator_cfg) {
    ResultLoggerConfig config{};
    config.result_level = ResultLogLevel::OFF;
    config.timing_level = TimingLogLevel::OFF;

    const auto hbm_cfg = root_cfg ? root_cfg["hbm_pim"] : YAML::Node{};
    if (hbm_cfg && hbm_cfg["enable_result_log"] &&
        hbm_cfg["enable_result_log"].as<bool>(false)) {
        config.result_level = ResultLogLevel::FULL;
    }

    if (verilator_cfg && verilator_cfg["logging"]) {
        const auto logging_cfg = verilator_cfg["logging"];
        config.buffer_size = logging_cfg["buffer_size"].as<size_t>(config.buffer_size);
        if (config.buffer_size == 0) {
            throw std::runtime_error("HBM-PIM logging.buffer_size must be greater than zero");
        }
        config.result_level = parse_result_log_level(logging_cfg);
        config.timing_level = parse_timing_log_level(logging_cfg);
        config.schema_version = logging_cfg["schema_version"].as<uint32_t>(config.schema_version);
    }

    return config;
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
to_word_burst(const std::vector<Ramulator::Data_t>& data_burst) {
    std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS> words{};
    for (size_t i = 0; i < words.size() && i < data_burst.size(); ++i) {
        words[i] = static_cast<uint32_t>(data_burst[i]);
    }
    return words;
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
to_fp16_burst(const std::vector<Ramulator::Data_t>& data_burst) {
    std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES> lanes{};
    for (size_t i = 0; i < lanes.size() && i < data_burst.size(); ++i) {
        lanes[i] = static_cast<uint16_t>(data_burst[i] & 0xFFFF);
    }
    return lanes;
}

} // namespace

HBMPIMArchitectureBackend::~HBMPIMArchitectureBackend() {
    flush_pending_timing();
}

void HBMPIMArchitectureBackend::init(const ArchitectureInitContext& ctx) {
    m_dram = ctx.dram;
    m_logical_pseudochannel_id = ctx.channel_id;
    m_memory_manager = std::make_unique<HBMPIMMemoryManager>();
    m_memory_manager->init(m_logical_pseudochannel_id);
    m_compute_engine = std::make_unique<HBMPIMComputeEngine>(
        HBMPIMComputeEngine::Mode::SOFTWARE);

    if (m_dram) {
        if (m_dram->m_commands.contains("ACT")) {
            m_cmd_act = m_dram->m_commands("ACT");
        }
        if (m_dram->m_commands.contains("PRE")) {
            m_cmd_pre = m_dram->m_commands("PRE");
        }
        if (m_dram->m_commands.contains("PREA")) {
            m_cmd_prea = m_dram->m_commands("PREA");
        }
        if (m_dram->m_levels.contains("bankgroup")) {
            m_bankgroup_level = m_dram->m_levels("bankgroup");
        }
        if (m_dram->m_levels.contains("bank")) {
            m_bank_level = m_dram->m_levels("bank");
        }
        if (m_dram->m_levels.contains("row")) {
            m_row_level = m_dram->m_levels("row");
        }
    }

    const auto logger_config = parse_hbmpim_logger_config(ctx.root_config, ctx.verilator_config);
    m_pairs.clear();
    m_divergence_tracker.init(ctx.dram, m_pairs);
    m_result_logger.init(ctx.channel_id, ctx.result_log_path, logger_config);
    m_initialized = true;
}

void HBMPIMArchitectureBackend::reset() {
    if (m_memory_manager) {
        m_memory_manager->reset();
    }
    if (m_compute_engine) {
        m_compute_engine->reset();
    }
    m_divergence_tracker.reset();
}

void HBMPIMArchitectureBackend::on_command_issued(const Ramulator::Request& req,
                                                  uint64_t sim_issue_cycle) {
    if (!m_initialized) {
        throw std::runtime_error("HBMPIMArchitectureBackend used before init()");
    }
    if (req.type != Ramulator::Type::HBM_PIM) {
        return;
    }

    handle_dram_command(req);
    if (req.command == req.final_command) {
        handle_hbmpim_command(req, sim_issue_cycle);
    }
}

void HBMPIMArchitectureBackend::flush_pending_timing() {
    if (!m_initialized) {
        return;
    }
    for (const auto& dr : m_divergence_tracker.flush_all_pending()) {
        if (dr.logged) {
            m_result_logger.log_timing(dr);
        }
    }
    m_result_logger.flush();
}

bool HBMPIMArchitectureBackend::can_defer_consumer(int /*command*/,
                                                   uint64_t /*now_cycle*/) const {
    return false;
}

void HBMPIMArchitectureBackend::note_consumer_deferred(int /*command*/,
                                                       uint64_t /*cycles*/) {
    // Stage 5C: HBM-PIM skeleton keeps Stage 4 ABI but does not use defer.
}

std::vector<ComputeConsumerPair> HBMPIMArchitectureBackend::get_compute_consumer_pairs() const {
    return m_pairs;
}

void HBMPIMArchitectureBackend::handle_dram_command(const Ramulator::Request& req) {
    if (!m_memory_manager) {
        return;
    }

    if (req.command == m_cmd_act) {
        const int bank_id = bank_id_from_addr_vec(req.addr_vec);
        const int row = row_from_addr_vec(req.addr_vec);
        if (bank_id >= 0 && row >= 0) {
            m_memory_manager->open_row(bank_id, row);
        }
        return;
    }

    if (req.command == m_cmd_pre) {
        const int bank_id = bank_id_from_addr_vec(req.addr_vec);
        if (bank_id >= 0) {
            m_memory_manager->close_row(bank_id);
        }
        return;
    }

    if (req.command == m_cmd_prea) {
        m_memory_manager->close_all_rows();
    }
}

void HBMPIMArchitectureBackend::handle_hbmpim_command(const Ramulator::Request& req,
                                                      uint64_t sim_issue_cycle) {
    const auto& info = req.hbm_pim_info;
    const int word_index = infer_target_word(info);

    switch (info.opcode) {
        case Ramulator::HBMPIMOpcode::HBMPIM_CRF_WR:
            if (info.crf_index >= 0) {
                if (!info.data_burst.empty()) {
                    for (size_t i = 0; i < info.data_burst.size(); ++i) {
                        m_memory_manager->write_crf(
                            info.crf_index + static_cast<int>(i),
                            static_cast<uint32_t>(info.data_burst[i]));
                    }
                } else {
                    m_memory_manager->write_crf(info.crf_index,
                                                static_cast<uint32_t>(info.data32));
                }
            }
            break;
        case Ramulator::HBMPIMOpcode::HBMPIM_GRF_A_WR:
            for_each_target_block(info.pim_block_mask, [&](int block_id) {
                if (info.data_burst.size() >= HBMPIMMemoryManager::FP16_LANES) {
                    m_memory_manager->write_grf_a_fp16_lanes(block_id, word_index,
                                                             to_fp16_burst(info.data_burst));
                } else if (info.data_burst.size() >= HBMPIMMemoryManager::REG_WORDS) {
                    m_memory_manager->write_grf_a_burst_words(block_id, word_index,
                                                              to_word_burst(info.data_burst));
                } else {
                    m_memory_manager->write_grf_a(block_id, word_index,
                                                 static_cast<uint32_t>(info.data32));
                }
            });
            break;
        case Ramulator::HBMPIMOpcode::HBMPIM_GRF_B_WR:
            for_each_target_block(info.pim_block_mask, [&](int block_id) {
                if (info.data_burst.size() >= HBMPIMMemoryManager::FP16_LANES) {
                    m_memory_manager->write_grf_b_fp16_lanes(block_id, word_index,
                                                             to_fp16_burst(info.data_burst));
                } else if (info.data_burst.size() >= HBMPIMMemoryManager::REG_WORDS) {
                    m_memory_manager->write_grf_b_burst_words(block_id, word_index,
                                                              to_word_burst(info.data_burst));
                } else {
                    m_memory_manager->write_grf_b(block_id, word_index,
                                                 static_cast<uint32_t>(info.data32));
                }
            });
            break;
        case Ramulator::HBMPIMOpcode::HBMPIM_SRF_WR:
            for_each_target_block(info.pim_block_mask, [&](int block_id) {
                if (info.data_burst.size() >= HBMPIMMemoryManager::REG_WORDS * 2) {
                    std::array<uint16_t, HBMPIMMemoryManager::REG_WORDS * 2> lanes{};
                    for (size_t i = 0; i < lanes.size(); ++i) {
                        lanes[i] = static_cast<uint16_t>(info.data_burst[i] & 0xFFFF);
                    }
                    m_memory_manager->write_srf_fp16_lanes(block_id, lanes);
                } else if (info.data_burst.size() >= HBMPIMMemoryManager::REG_WORDS) {
                    std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS> words{};
                    for (size_t i = 0; i < words.size(); ++i) {
                        words[i] = static_cast<uint32_t>(info.data_burst[i]);
                    }
                    m_memory_manager->write_srf_burst_words(block_id, words);
                } else {
                    m_memory_manager->write_srf(block_id, word_index,
                                                static_cast<uint32_t>(info.data32));
                }
            });
            break;
        case Ramulator::HBMPIMOpcode::HBMPIM_TRIGGER:
            if (m_compute_engine) {
                m_compute_engine->execute_trigger(*m_memory_manager, info);
            }
            break;
        case Ramulator::HBMPIMOpcode::HBMPIM_RD_GRF:
        case Ramulator::HBMPIMOpcode::HBMPIM_RD_SRF:
            emit_readback_if_enabled(req, sim_issue_cycle);
            break;
        case Ramulator::HBMPIMOpcode::MAX:
        case Ramulator::HBMPIMOpcode::MIN:
            break;
    }
}

void HBMPIMArchitectureBackend::emit_readback_if_enabled(const Ramulator::Request& req,
                                                         uint64_t sim_issue_cycle) {
    if (!m_result_logger.result_enabled()) {
        return;
    }

    const auto& info = req.hbm_pim_info;
    const int word_index = infer_target_word(info);
    for_each_target_block(info.pim_block_mask, [&](int block_id) {
        std::array<uint16_t, 16> payload{};
        if (info.opcode == Ramulator::HBMPIMOpcode::HBMPIM_RD_GRF) {
            payload = m_memory_manager->read_grf_a_fp16_lanes(block_id, word_index);
        } else if (info.opcode == Ramulator::HBMPIMOpcode::HBMPIM_RD_SRF) {
            payload = m_memory_manager->read_srf_fp16_lanes(block_id);
        }

        ResultPacket pkt{};
        pkt.cycle = sim_issue_cycle;
        pkt.command = req.command;
        pkt.bank_id = block_id;
        pkt.row = -1;
        pkt.col = word_index;
        pkt.data = payload;
        m_result_logger.log_result(pkt);
    });
}

int HBMPIMArchitectureBackend::infer_target_word(const Ramulator::HBMPIMRequestInfo& info) const {
    return (info.word_index >= 0)
        ? std::min<int>(info.word_index, HBMPIMMemoryManager::REG_WORDS - 1)
        : 0;
}

void HBMPIMArchitectureBackend::for_each_target_block(
    uint64_t pim_block_mask,
    const std::function<void(int block_id)>& fn) {
    const uint64_t normalized_mask =
        (pim_block_mask == 0 || static_cast<int64_t>(pim_block_mask) < 0)
            ? 0x1ULL
            : pim_block_mask;
    for (int block_id = 0; block_id < HBMPIMMemoryManager::NUM_PIM_BLOCKS; ++block_id) {
        if ((normalized_mask & (1ULL << block_id)) == 0) {
            continue;
        }
        fn(block_id);
    }
}

int HBMPIMArchitectureBackend::bank_id_from_addr_vec(const std::vector<int>& addr_vec) const {
    if (m_bank_level < 0 || static_cast<size_t>(m_bank_level) >= addr_vec.size()) {
        return -1;
    }
    const int bank = addr_vec[static_cast<size_t>(m_bank_level)];
    if (bank < 0) {
        return -1;
    }
    if (m_bankgroup_level >= 0 && static_cast<size_t>(m_bankgroup_level) < addr_vec.size()) {
        const int bg = addr_vec[static_cast<size_t>(m_bankgroup_level)];
        if (bg >= 0) {
            return bg * 4 + bank;
        }
    }
    return bank;
}

int HBMPIMArchitectureBackend::row_from_addr_vec(const std::vector<int>& addr_vec) const {
    if (m_row_level < 0 || static_cast<size_t>(m_row_level) >= addr_vec.size()) {
        return -1;
    }
    return addr_vec[static_cast<size_t>(m_row_level)];
}

} // namespace aim_cosim
