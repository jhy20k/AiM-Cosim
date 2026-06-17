// =============================================================================
// AiMDRAMControllerRTL — Controller-centric architecture with RTL co-simulation
// Registered as impl: "AiMRTL" in ramulator2 factory
//
// Responsibilities:
//   - Owns DRAM timing and the request scheduler (unchanged from baseline).
//   - Runs AiM functional dispatch directly in-controller, with controller-side
//     scoreboard ordering. By default, GDDR6-AiM PU latency is treated as
//     hidden under the DRAM protocol, so measured rtl_cycles do not extend
//     memory_system_cycles. Debug configs can opt into raw RTL-cycle gating.
//   - Retains IArchitectureBackend only for non-AiM future-work scaffolds
//     such as HBM-PIM.
// =============================================================================

#include "base/request.h"
#include "architecture/architecture_backend.h"
#include "architecture/hbmpim_architecture_backend.h"
#include "controller/aim_compute_scoreboard.h"
#include "controller/aim_row_locality.h"
#include "compute_engine/aim_compute_engine.h"
#include "controller/functional_dispatcher.h"
#include "controller/result_logger.h"
#include "data_loader/data_loader.h"
#include "dram_controller/controller.h"
#include "memory_manager/memory_manager.h"
#include "memory_system/impl/aim_region_registry.h"
#include "memory_system/memory_system.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Ramulator {

namespace {

std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

uint32_t channel_random_seed(uint32_t base_seed, int channel_id) {
    const uint32_t ch = channel_id < 0 ? 0u : static_cast<uint32_t>(channel_id);
    return base_seed ^ (0x9E3779B9u * ch);
}

aim_cosim::ResultLogLevel parse_result_log_level(const YAML::Node& logging_cfg) {
    if (!logging_cfg || !logging_cfg["result_level"]) {
        return aim_cosim::ResultLogLevel::FULL;
    }

    const auto value = lowercase_copy(logging_cfg["result_level"].as<std::string>("full"));
    if (value == "full") {
        return aim_cosim::ResultLogLevel::FULL;
    }
    if (value == "off") {
        return aim_cosim::ResultLogLevel::OFF;
    }
    throw std::runtime_error("Unsupported result_level in verilator.logging");
}

aim_cosim::TimingLogLevel parse_timing_log_level(const YAML::Node& logging_cfg) {
    if (!logging_cfg || !logging_cfg["timing_level"]) {
        return aim_cosim::TimingLogLevel::FULL;
    }

    const auto value = lowercase_copy(logging_cfg["timing_level"].as<std::string>("full"));
    if (value == "full") {
        return aim_cosim::TimingLogLevel::FULL;
    }
    if (value == "summary") {
        return aim_cosim::TimingLogLevel::SUMMARY;
    }
    if (value == "off") {
        return aim_cosim::TimingLogLevel::OFF;
    }
    throw std::runtime_error("Unsupported timing_level in verilator.logging");
}

aim_cosim::ResultLoggerConfig parse_logger_config(const YAML::Node& verilator_cfg) {
    aim_cosim::ResultLoggerConfig config{};
    if (!verilator_cfg || !verilator_cfg["logging"]) {
        return config;
    }

    const auto logging_cfg = verilator_cfg["logging"];
    config.buffer_size = logging_cfg["buffer_size"].as<size_t>(config.buffer_size);
    if (config.buffer_size == 0) {
        throw std::runtime_error("verilator.logging.buffer_size must be greater than zero");
    }
    config.result_level = parse_result_log_level(logging_cfg);
    config.timing_level = parse_timing_log_level(logging_cfg);
    config.rdaf16_bank0_only = logging_cfg["rdaf16_bank0_only"].as<bool>(config.rdaf16_bank0_only);
    config.schema_version = logging_cfg["schema_version"].as<uint32_t>(config.schema_version);
    return config;
}

} // namespace

class AiMDRAMControllerRTL final : public IDRAMController, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(
        IDRAMController, AiMDRAMControllerRTL, "AiMRTL",
        "AiM DRAM controller with Verilator RTL co-simulation.");

private:
    std::deque<Request> pending_reads;
    std::vector<Request> pending_writes;

    ReqBuffer m_active_buffer;
    ReqBuffer m_priority_buffer;
    ReqBuffer m_read_buffer;
    ReqBuffer m_write_buffer;
    ReqBuffer m_aim_buffer;

    int m_row_addr_idx = -1;
    IMemorySystem* m_memory_system = nullptr;

    float m_wr_low_watermark;
    float m_wr_high_watermark;
    bool m_is_write_mode = false;

    std::vector<IControllerPlugin*> m_plugins;

    size_t s_num_row_hits = 0;
    size_t s_num_row_misses = 0;
    size_t s_num_row_conflicts = 0;

    std::map<Type, int> s_num_RW_cycles;
    std::map<Opcode, int> s_num_AiM_cycles;
    // Future work: HBM-PIM counters support the scaffold path and are not
    // part of the validated GDDR6-AiM RTL flow.
    std::map<HBMPIMOpcode, int> s_num_HBMPIM_cycles;
    std::map<int, int> s_num_commands;
    int s_num_idle_cycles = 0;
    int s_num_active_cycles = 0;
    int s_num_precharged_cycles = 0;
    size_t s_num_scoreboard_wait_cycles_total = 0;
    size_t s_num_scoreboard_wait_cycles_producer = 0;
    size_t s_num_scoreboard_wait_cycles_rdmac16 = 0;
    size_t s_num_scoreboard_wait_cycles_rdaf16 = 0;
    size_t s_num_defer_cycles_total = 0;
    size_t s_num_defer_cycles_rdmac16 = 0;
    size_t s_num_defer_cycles_rdaf16 = 0;
    size_t s_num_divergence_count = 0;

    bool is_reg_RW_mode = false;

    // AiM direct path (Priority 1 canonical path)
    bool m_use_direct_aim = false;
    std::unique_ptr<aim_cosim::AiMMemoryManager> m_aim_mm;
    std::unique_ptr<aim_cosim::IComputeEngine> m_aim_ce;
    std::unique_ptr<aim_cosim::FunctionalDispatcher> m_aim_dispatcher;
    std::unique_ptr<aim_cosim::ResultLogger> m_aim_logger;
    aim_cosim::AiMComputeScoreboard m_aim_scoreboard;
    bool m_use_direct_aim_scoreboard_gating = false;

    // Architecture backend for future-work non-GDDR6 paths such as HBM-PIM.
    std::unique_ptr<aim_cosim::IArchitectureBackend> m_arch_backend;

    // Cached numeric command IDs for splitting defer-cycle stats by consumer.
    // Resolved once in setup() after the DRAM spec is known.
    int m_mac_cmd = -1;
    int m_mac16_cmd = -1;
    int m_af16_cmd = -1;
    int m_ewmul16_cmd = -1;
    int m_rdmac16_cmd = -1;
    int m_rdaf16_cmd = -1;
    aim_cosim::AiMRowLocalityCommandIds m_row_locality_ids{};

    bool is_arch_request_type(Type type) const {
        return (type == Type::AIM) || (type == Type::HBM_PIM);
    }

    void validate_controller_region_context(const Request& req) const {
        if (m_memory_system == nullptr || !m_memory_system->is_strict_trace_mode()) {
            return;
        }
        if (req.type != Type::AIM || req.addr == -1) {
            return;
        }

        AiMRegion region;
        if (!m_memory_system->lookup_region_for_byte(req.addr, region)) {
            throw std::runtime_error(
                fmt::format("AiMRTL controller failed to resolve strict region for byte addr {:#x}",
                            static_cast<uint64_t>(req.addr)));
        }
    }

    void account_row_locality_once(const Request& req, int preq_command) {
        if (req.type != Type::AIM || req.issue != -1) {
            return;
        }

        const auto locality =
            aim_cosim::classify_row_locality(req.final_command, preq_command, m_row_locality_ids);
        switch (locality) {
        case aim_cosim::AiMRowLocalityClass::ReadyLike:
            s_num_row_hits += 1;
            break;
        case aim_cosim::AiMRowLocalityClass::Miss:
            s_num_row_misses += 1;
            break;
        case aim_cosim::AiMRowLocalityClass::Conflict:
            s_num_row_conflicts += 1;
            break;
        case aim_cosim::AiMRowLocalityClass::None:
            break;
        }
    }

public:
    void init() override {
        m_wr_low_watermark = param<float>("wr_low_watermark")
            .desc("Threshold for switching back to read mode.")
            .default_val(0.2f);
        m_wr_high_watermark = param<float>("wr_high_watermark")
            .desc("Threshold for switching to write mode.")
            .default_val(0.8f);
        m_clock_ratio = param<uint>("clock_ratio").required();

        m_scheduler = create_child_ifce<IScheduler>();
        m_refresh = create_child_ifce<IRefreshManager>();

        if (m_config["plugins"]) {
            YAML::Node plugin_configs = m_config["plugins"];
            for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
                m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
            }
        }

        // Initialize architecture backend from YAML config. HBM-PIM is kept as
        // a future-work scaffold, while AiM preserves the existing
        // "backend only when verilator.enabled" behavior for compatibility.
        const bool verilator_enabled =
            m_config["verilator"] && m_config["verilator"]["enabled"] &&
            m_config["verilator"]["enabled"].as<bool>(false);
        std::string arch_kind;
        if (m_config["architecture"] && m_config["architecture"]["kind"]) {
            arch_kind = m_config["architecture"]["kind"].as<std::string>("");
        } else if (verilator_enabled) {
            arch_kind = "aim";
        }

        if (arch_kind == "aim" && verilator_enabled) {
            m_use_direct_aim = true;
        } else if (arch_kind == "hbm_pim") {
            // Future work: this backend is not part of the validated
            // GDDR6-AiM RTL flow.
            m_arch_backend = std::make_unique<aim_cosim::HBMPIMArchitectureBackend>();
        } else if (!arch_kind.empty() && arch_kind != "aim") {
            throw std::runtime_error(
                fmt::format("Unsupported architecture.kind '{}' for AiMRTL controller", arch_kind));
        }
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
        m_memory_system = memory_system;
        m_dram = memory_system->get_ifce<IDRAM>();
        m_row_addr_idx = m_dram->m_levels("row");
        m_priority_buffer.max_size = 512 * 3 + 32;

        // Register per-channel statistics (same as AiMDRAMController)
        for (const auto type : {Type::Read, Type::Write}) {
            s_num_RW_cycles[type] = 0;
            register_stat(s_num_RW_cycles[type])
                .name(fmt::format("CH{}_{}_cycles",
                                  m_channel_id,
                                  AiMISRInfo::convert_type_to_str(type)));
        }

        for (int opcode = (int)Opcode::MIN + 1; opcode < (int)Opcode::MAX; opcode++) {
            s_num_AiM_cycles[(Opcode)opcode] = 0;
            register_stat(s_num_AiM_cycles[(Opcode)opcode])
                .name(fmt::format("CH{}_AiM_{}_cycles", m_channel_id, AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)))
                .desc(fmt::format("total number of AiM {} cycles", AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)));
        }
        // Future work: HBM-PIM statistics are registered only for the scaffold path.
        for (int opcode = (int)HBMPIMOpcode::MIN + 1; opcode < (int)HBMPIMOpcode::MAX; opcode++) {
            auto hbm_opcode = static_cast<HBMPIMOpcode>(opcode);
            s_num_HBMPIM_cycles[hbm_opcode] = 0;
            register_stat(s_num_HBMPIM_cycles[hbm_opcode])
                .name(fmt::format("CH{}_HBMPIM_{}_cycles", m_channel_id, HBMPIMInfo::convert_opcode_to_str(hbm_opcode)))
                .desc(fmt::format("total number of HBM-PIM {} cycles", HBMPIMInfo::convert_opcode_to_str(hbm_opcode)));
        }

        for (int command_id = 0; command_id < m_dram->m_commands.size(); command_id++) {
            s_num_commands[command_id] = 0;
            register_stat(s_num_commands[command_id])
                .name(fmt::format("CH{}_num_{}_commands", m_channel_id, std::string(m_dram->m_commands(command_id))))
                .desc(fmt::format("total number of {} commands", std::string(m_dram->m_commands(command_id))));
        }

        register_stat(s_num_idle_cycles)
            .name(fmt::format("CH{}_idle_cycles", m_channel_id))
            .desc(fmt::format("total number of idle cycles"));

        register_stat(s_num_active_cycles)
            .name(fmt::format("CH{}_active_cycles", m_channel_id))
            .desc(fmt::format("total number of active cycles"));

        register_stat(s_num_precharged_cycles)
            .name(fmt::format("CH{}_precharged_cycles", m_channel_id))
            .desc(fmt::format("total number of precharged cycles"));

        if (m_dram->m_commands.contains("MAC")) {
            m_mac_cmd = m_dram->m_commands("MAC");
        }
        if (m_dram->m_commands.contains("MAC16")) {
            m_mac16_cmd = m_dram->m_commands("MAC16");
        }
        if (m_dram->m_commands.contains("AF16")) {
            m_af16_cmd = m_dram->m_commands("AF16");
        }
        if (m_dram->m_commands.contains("EWMUL16")) {
            m_ewmul16_cmd = m_dram->m_commands("EWMUL16");
        }
        if (m_dram->m_commands.contains("RDMAC16")) {
            m_rdmac16_cmd = m_dram->m_commands("RDMAC16");
        }
        if (m_dram->m_commands.contains("RDAF16")) {
            m_rdaf16_cmd = m_dram->m_commands("RDAF16");
        }
        if (m_dram->m_commands.contains("ACT")) {
            m_row_locality_ids.act = m_dram->m_commands("ACT");
        }
        if (m_dram->m_commands.contains("ACT4")) {
            m_row_locality_ids.act4 = m_dram->m_commands("ACT4");
        }
        if (m_dram->m_commands.contains("ACT16")) {
            m_row_locality_ids.act16 = m_dram->m_commands("ACT16");
        }
        if (m_dram->m_commands.contains("PRE")) {
            m_row_locality_ids.pre = m_dram->m_commands("PRE");
        }
        if (m_dram->m_commands.contains("PRE4")) {
            m_row_locality_ids.pre4 = m_dram->m_commands("PRE4");
        }
        if (m_dram->m_commands.contains("PREA")) {
            m_row_locality_ids.prea = m_dram->m_commands("PREA");
        }

        if (m_use_direct_aim) {
            register_stat(s_num_scoreboard_wait_cycles_total)
                .name(fmt::format("CH{}_scoreboard_wait_cycles_total", m_channel_id))
                .desc("total number of scheduler cycles blocked by AiM direct scoreboard");
            register_stat(s_num_scoreboard_wait_cycles_producer)
                .name(fmt::format("CH{}_scoreboard_wait_cycles_producer", m_channel_id))
                .desc("number of scheduler cycles blocked before a producer command became issue-ready");
            register_stat(s_num_scoreboard_wait_cycles_rdmac16)
                .name(fmt::format("CH{}_scoreboard_wait_cycles_rdmac16", m_channel_id))
                .desc("number of scheduler cycles blocked before RDMAC16 became visible-ready");
            register_stat(s_num_scoreboard_wait_cycles_rdaf16)
                .name(fmt::format("CH{}_scoreboard_wait_cycles_rdaf16", m_channel_id))
                .desc("number of scheduler cycles blocked before RDAF16 became visible-ready");
        } else {
            register_stat(s_num_defer_cycles_total)
                .name(fmt::format("CH{}_defer_cycles_total", m_channel_id))
                .desc("total number of scheduler cycles deferred by Stage 4 gating");
            register_stat(s_num_defer_cycles_rdmac16)
                .name(fmt::format("CH{}_defer_cycles_rdmac16", m_channel_id))
                .desc("number of scheduler cycles deferred for RDMAC16");
            register_stat(s_num_defer_cycles_rdaf16)
                .name(fmt::format("CH{}_defer_cycles_rdaf16", m_channel_id))
                .desc("number of scheduler cycles deferred for RDAF16");
            register_stat(s_num_divergence_count)
                .name(fmt::format("CH{}_divergence_count", m_channel_id))
                .desc("number of RTL timing > DRAM timing divergence events");
        }

        register_stat(s_num_row_hits)
            .name(fmt::format("CH{}_row_hits", m_channel_id))
            .desc("number of AIM requests classified as ready-like by get_preq_command()");
        register_stat(s_num_row_misses)
            .name(fmt::format("CH{}_row_misses", m_channel_id))
            .desc("number of AIM requests classified as closed-row/unopened-row misses by get_preq_command()");
        register_stat(s_num_row_conflicts)
            .name(fmt::format("CH{}_row_conflicts", m_channel_id))
            .desc("number of AIM requests classified as row conflicts by get_preq_command()");

        if (m_use_direct_aim) {
            int num_banks = 16;
            int max_rows = 16384;
            int tiles_per_row = 64;
            std::string init_mode = "zero";
            uint32_t init_seed = 42;
            std::string data_file;
            int load_row_shard_mod = 0;
            uint64_t mac_cycle_override = 0;
            uint64_t mac16_cycle_override = 0;
            uint64_t af16_cycle_override = 0;
            uint64_t ewmul16_cycle_override = 0;
            aim_cosim::ResultLoggerConfig logger_config{};
            m_use_direct_aim_scoreboard_gating = false;

            if (m_config["memory_manager"]) {
                auto mm = m_config["memory_manager"];
                num_banks = mm["num_banks"].as<int>(16);
                max_rows = mm["max_rows"].as<int>(16384);
                tiles_per_row = mm["tiles_per_row"].as<int>(64);
                init_mode = mm["init_mode"].as<std::string>("zero");
                init_seed = mm["init_seed"].as<uint32_t>(42);
                data_file = mm["data_file"].as<std::string>("");
                load_row_shard_mod = mm["load_row_shard_mod"].as<int>(0);
            }

            m_aim_mm = std::make_unique<aim_cosim::AiMMemoryManager>();
            m_aim_mm->init(num_banks, max_rows, tiles_per_row);
            if (init_mode == "random") {
                m_aim_mm->init_random(channel_random_seed(init_seed, m_channel_id));
            } else if (init_mode == "file" && !data_file.empty()) {
                m_aim_mm->init_zero();
                if (load_row_shard_mod > 0) {
                    aim_cosim::AiMDataLoader::LoadOptions load_options{};
                    load_options.row_shard_mod = load_row_shard_mod;
                    load_options.row_shard_offset = m_channel_id % load_row_shard_mod;
                    aim_cosim::AiMDataLoader::load_from_file(data_file, *m_aim_mm, load_options);
                } else {
                    aim_cosim::AiMDataLoader::load_from_file(data_file, *m_aim_mm);
                }
            } else {
                m_aim_mm->init_zero();
            }

            auto aim_engine = std::make_unique<aim_cosim::AiMComputeEngine>();
            bool vcd_trace = false;
            std::string vcd_file = "aim_rtl.vcd";
            if (m_config["verilator"]) {
                auto verilator_cfg = m_config["verilator"];
                if (verilator_cfg["vcd_trace"]) {
                    vcd_trace = verilator_cfg["vcd_trace"].as<bool>(false);
                }
                if (verilator_cfg["vcd_file"]) {
                    vcd_file = verilator_cfg["vcd_file"].as<std::string>("aim_rtl.vcd");
                }
                m_use_direct_aim_scoreboard_gating =
                    verilator_cfg["scoreboard_gating"].as<bool>(false);
                if (verilator_cfg["rtl_cycle_override"]) {
                    const auto override = verilator_cfg["rtl_cycle_override"];
                    auto read_override = [&](const char* upper_key,
                                             const char* lower_key) -> uint64_t {
                        if (override[upper_key]) {
                            return override[upper_key].as<uint64_t>(0);
                        }
                        if (override[lower_key]) {
                            return override[lower_key].as<uint64_t>(0);
                        }
                        return 0;
                    };
                    mac_cycle_override = read_override("MAC", "mac");
                    mac16_cycle_override = read_override("MAC16", "mac16");
                    af16_cycle_override = read_override("AF16", "af16");
                    ewmul16_cycle_override = read_override("EWMUL16", "ewmul16");
                }
                logger_config = parse_logger_config(verilator_cfg);
            }
            if (vcd_trace && !vcd_file.empty()) {
                aim_engine->enable_trace(vcd_file);
            }
            aim_engine->set_mac_cycle_override(mac_cycle_override);
            aim_engine->set_mac16_cycle_override(mac16_cycle_override);
            aim_engine->set_af16_cycle_override(af16_cycle_override);
            aim_engine->set_ewmul16_cycle_override(ewmul16_cycle_override);
            m_aim_ce = std::move(aim_engine);

            m_aim_logger = std::make_unique<aim_cosim::ResultLogger>();
            m_aim_logger->init(
                m_channel_id,
                m_config["verilator"]["result_log"].as<std::string>(""),
                logger_config);

            m_aim_dispatcher = std::make_unique<aim_cosim::FunctionalDispatcher>();
            m_aim_dispatcher->init(m_aim_mm.get(), m_aim_ce.get(), m_aim_logger.get());
            m_aim_scoreboard.configure(
                m_mac_cmd,
                m_mac16_cmd,
                m_af16_cmd,
                m_ewmul16_cmd,
                m_rdmac16_cmd,
                m_rdaf16_cmd);
        }

        // Hand the backend the runtime context it needs to wire up its
        // memory manager, compute engine, divergence tracker, and logger.
        if (m_arch_backend) {
            aim_cosim::ArchitectureInitContext ctx{};
            ctx.root_config = YAML::Clone(m_config);
            ctx.verilator_config = m_config["verilator"] ? YAML::Clone(m_config["verilator"]) : YAML::Node{};
            ctx.memory_manager_config = m_config["memory_manager"] ? YAML::Clone(m_config["memory_manager"]) : YAML::Node{};
            ctx.architecture_config = m_config["architecture"] ? YAML::Clone(m_config["architecture"]) : YAML::Node{};
            ctx.dram = m_dram;
            ctx.channel_id = m_channel_id;
            ctx.result_log_path = m_config["verilator"]["result_log"].as<std::string>("");
            m_arch_backend->init(ctx);

        }
    };

    void finalize() override {
        if (m_aim_logger) {
            m_aim_logger->flush();
        }
        if (m_arch_backend) {
            m_arch_backend->flush_pending_timing();
        }
    }

    bool send(Request& req) override {
        if (is_arch_request_type(req.type)) {
            if ((m_write_buffer.size() != 0) || (m_read_buffer.size() != 0))
                return false;
            if (req.type == Type::AIM) {
                req.final_command = m_dram->m_aim_request_translations((int)req.opcode);
            } else {
                req.final_command = m_dram->m_hbm_pim_request_translations((int)req.hbm_pim_info.opcode);
            }
        } else {
            if (m_aim_buffer.size() != 0)
                return false;
            req.final_command = m_dram->m_request_translations((int)req.type);
        }

        // Forward existing write requests to incoming read requests
        if (req.type == Type::Read) {
            auto compare_addr = [req](const Request& wreq) {
                return wreq.addr == req.addr;
            };
            if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(), compare_addr) != m_write_buffer.end()) {
                req.depart = m_clk + 1;
                pending_reads.push_back(req);
                return true;
            }
        }

        // Enqueue to corresponding buffer
        bool is_success = false;
        req.arrive = m_clk;
        if (req.type == Type::Read) {
            is_success = m_read_buffer.enqueue(req);
        } else if (req.type == Type::Write) {
            is_success = m_write_buffer.enqueue(req);
        } else if (is_arch_request_type(req.type)) {
            is_success = m_aim_buffer.enqueue(req);
        } else {
            throw std::runtime_error("Invalid request type!");
        }
        if (!is_success) {
            req.arrive = -1;
            return false;
        }

        return true;
    };

    bool priority_send(Request& req) override {
        if (req.type == Type::AIM) {
            req.final_command = m_dram->m_aim_request_translations((int)req.opcode);
        } else if (req.type == Type::HBM_PIM) {
            req.final_command = m_dram->m_hbm_pim_request_translations((int)req.hbm_pim_info.opcode);
        } else {
            req.final_command = m_dram->m_request_translations((int)req.type);
        }

        bool is_success = false;
        is_success = m_priority_buffer.enqueue(req);
        return is_success;
    }

    void tick() override {
        m_clk++;

        // 1. Serve completed requests
        serve_completed_reqs();

        m_refresh->tick();

        // 2. Try to find a request to serve
        ReqBuffer::iterator req_it;
        ReqBuffer* buffer = nullptr;
        bool request_found = schedule_request(req_it, buffer);

        // 3. Issue the commands to serve the request
        if (request_found) {
            if ((req_it->opcode == Opcode::ISR_EOC) || (req_it->opcode == Opcode::ISR_SYNC)) {
                req_it->depart = m_clk;
                pending_reads.push_back(*req_it);
                buffer->remove(req_it);
            } else {
                bool requires_reg_RW_mode = false;
                if (req_it->type == Type::AIM) {
                    if (AiMISRInfo::opcode_requires_reg_RW_mod(req_it->opcode)) {
                        requires_reg_RW_mode = true;
                    }
                }

                if (requires_reg_RW_mode ^ is_reg_RW_mode) {
                    req_it->command = m_dram->m_commands("TMOD");
                    is_reg_RW_mode = !is_reg_RW_mode;
                }

                if (req_it->issue == -1)
                    req_it->issue = m_clk - 1;
                m_dram->issue_command(req_it->command, req_it->addr_vec);

                uint64_t rtl_cycles = 0;
                if (m_use_direct_aim && m_aim_dispatcher &&
                    req_it->type == Type::AIM) {
                    validate_controller_region_context(*req_it);
                    rtl_cycles = m_aim_dispatcher->dispatch(*req_it, m_clk);
                    if (rtl_cycles > 0) {
                        note_compute_ready(*req_it, m_clk, rtl_cycles);
                        if (m_aim_logger) {
                            const int dram_min = m_dram->m_command_latencies(req_it->command);
                            m_aim_logger->log_compute_timing(
                                std::string(m_dram->m_commands(req_it->command)),
                                m_clk, rtl_cycles, dram_min);
                        }
                    }
                } else if (m_arch_backend) {
                    m_arch_backend->on_command_issued(*req_it, m_clk);
                }

                s_num_commands[req_it->command] += 1;

                // If issuing the last command, set depart and move to pending queue
                if (req_it->command == req_it->final_command) {
                    int latency = m_dram->m_command_latencies(req_it->command);
                    assert(latency > 0);
                    req_it->depart = m_clk + latency;
                    if (req_it->is_reader()) {
                        pending_reads.push_back(*req_it);
                    } else {
                        pending_writes.push_back(*req_it);
                    }
                    if (req_it->type == Type::AIM) {
                        s_num_AiM_cycles[req_it->opcode] += (m_clk - req_it->issue);
                    } else if (req_it->type == Type::HBM_PIM) {
                        s_num_HBMPIM_cycles[req_it->hbm_pim_info.opcode] += (m_clk - req_it->issue);
                    } else {
                        s_num_RW_cycles[req_it->type] += (m_clk - req_it->issue);
                    }
                    buffer->remove(req_it);
                } else if (!is_arch_request_type(req_it->type)) {
                    if (m_dram->m_command_meta(req_it->command).is_opening) {
                        m_active_buffer.enqueue(*req_it);
                        buffer->remove(req_it);
                    }
                }
            }
        } else if (m_read_buffer.size() == 0 && m_write_buffer.size() == 0 &&
                   m_aim_buffer.size() == 0 && pending_reads.size() == 0 &&
                   pending_writes.size() == 0) {
            s_num_idle_cycles += 1;
        }

        if (m_dram->m_open_rows[m_channel_id] == 0) {
            s_num_precharged_cycles += 1;
        } else {
            s_num_active_cycles += 1;
        }
    };

private:
    // =========================================================================
    // Scheduling & completion (unchanged from original)
    // =========================================================================

    void serve_completed_reqs() {
        if (pending_reads.size()) {
            auto& req = pending_reads[0];
            if (req.depart <= m_clk) {
                if (((req.opcode != Opcode::ISR_EOC) && (req.opcode != Opcode::ISR_SYNC)) ||
                    (pending_writes.size() == 0)) {
                    if (req.callback) {
                        req.callback(req);
                    }
                    pending_reads.pop_front();
                }
            }
        }
        auto write_req_it = pending_writes.begin();
        while (write_req_it != pending_writes.end()) {
            if (write_req_it->depart <= m_clk) {
                write_req_it = pending_writes.erase(write_req_it);
            } else {
                ++write_req_it;
            }
        }
    };

    void set_write_mode() {
        if (!m_is_write_mode) {
            if ((m_write_buffer.size() > m_wr_high_watermark * m_write_buffer.max_size) || m_read_buffer.size() == 0) {
                m_is_write_mode = true;
            }
        } else {
            if ((m_write_buffer.size() < m_wr_low_watermark * m_write_buffer.max_size) && m_read_buffer.size() != 0) {
                m_is_write_mode = false;
            }
        }
    };

    bool will_issue_tmod(const Request& req) const {
        if (req.type != Type::AIM) {
            return false;
        }

        if (!AiMISRInfo::opcode_requires_reg_RW_mod(req.opcode)) {
            return false;
        }

        return (is_reg_RW_mode == false);
    }

    int addr_to_bank_id(const std::vector<int>& addr_vec) const {
        assert(addr_vec.size() >= 3);
        return addr_vec[1] * 4 + addr_vec[2];
    }

    aim_cosim::AiMComputeScoreboard::ComputeOp make_compute_op(const Request& req) const {
        aim_cosim::AiMComputeScoreboard::ComputeOp op{};
        op.command = req.command;
        op.ewmul_bg = req.ewmul_bg;
        if (req.command == m_mac_cmd) {
            op.bank_id = addr_to_bank_id(req.addr_vec);
        }
        return op;
    }

    void note_compute_ready(const Request& req, int64_t issue_clk, uint64_t rtl_cycles) {
        const uint64_t scoreboard_cycles =
            m_use_direct_aim_scoreboard_gating ? rtl_cycles : 0;
        m_aim_scoreboard.note_ready(make_compute_op(req), issue_clk, scoreboard_cycles);
    }

    bool passes_direct_aim_compute_gate(const Request& req) {
        if (!m_use_direct_aim || req.type != Type::AIM) {
            return true;
        }

        // Only gate the final compute-producing or compute-consuming command.
        if (will_issue_tmod(req) || req.command != req.final_command) {
            return true;
        }

        const auto status = m_aim_scoreboard.gate_status(make_compute_op(req), m_clk);
        switch (status) {
        case aim_cosim::AiMComputeScoreboard::GateStatus::Pass:
            return true;
        case aim_cosim::AiMComputeScoreboard::GateStatus::ProducerNotReady:
            s_num_scoreboard_wait_cycles_total += 1;
            s_num_scoreboard_wait_cycles_producer += 1;
            return false;
        case aim_cosim::AiMComputeScoreboard::GateStatus::ConsumerNotReady:
            s_num_scoreboard_wait_cycles_total += 1;
            if (req.command == m_rdmac16_cmd) {
                s_num_scoreboard_wait_cycles_rdmac16 += 1;
            } else if (req.command == m_rdaf16_cmd) {
                s_num_scoreboard_wait_cycles_rdaf16 += 1;
            }
            return false;
        case aim_cosim::AiMComputeScoreboard::GateStatus::MissingProducer:
            throw std::runtime_error("AiM compute consumer issued before any producer completed");
        }
        return false;
    }

    // Future work: HBM-PIM conditional gating entry point. Returns true to
    // skip issuing this command on the current scheduler tick (caller will
    // revisit it next tick) and records one tick of defer credit on both the
    // local stat counters and the backend's pending row metadata.
    bool should_defer_request(const Request& req) {
        if (!m_arch_backend) {
            return false;
        }

        // We only gate the *real* consumer issue cycle. Pre-commands (e.g.
        // ACT) and the TMOD that switches into reg-RW mode must still be
        // allowed to make forward progress; gating reasserts when the
        // selected command becomes the final consumer command.
        if (will_issue_tmod(req) || req.command != req.final_command) {
            return false;
        }

        if (!m_arch_backend->can_defer_consumer(req.command, m_clk)) {
            return false;
        }

        s_num_defer_cycles_total += 1;
        if (req.command == m_rdmac16_cmd) {
            s_num_defer_cycles_rdmac16 += 1;
        } else if (req.command == m_rdaf16_cmd) {
            s_num_defer_cycles_rdaf16 += 1;
        }
        m_arch_backend->note_consumer_deferred(req.command, 1);
        return true;
    }

    bool schedule_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer) {
        bool request_found = false;

        // 1. Check the active buffer first
        if (req_it = m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
            if (m_dram->check_ready(req_it->command, req_it->addr_vec) &&
                passes_direct_aim_compute_gate(*req_it) &&
                !should_defer_request(*req_it)) {
                request_found = true;
                req_buffer = &m_active_buffer;
            }
        }

        // 2. If no request from active buffer, check the rest
        if (!request_found) {
            // 2.1 Priority buffer
            if (m_priority_buffer.size() != 0) {
                req_buffer = &m_priority_buffer;
                req_it = m_priority_buffer.begin();
                req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
                account_row_locality_once(*req_it, req_it->command);

                request_found =
                    m_dram->check_ready(req_it->command, req_it->addr_vec) &&
                    passes_direct_aim_compute_gate(*req_it) &&
                    !should_defer_request(*req_it);
                if ((request_found == false) & (m_priority_buffer.size() != 0)) {
                    return false;
                }
            }

            // 2.2 AiM or Read/Write buffers
            if (!request_found) {
                if (m_aim_buffer.size() != 0) {
                    req_it = m_aim_buffer.begin();
                    if ((req_it->opcode == Opcode::ISR_EOC) || (req_it->opcode == Opcode::ISR_SYNC)) {
                        req_buffer = &m_aim_buffer;
                        return true;
                    } else {
                        req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
                        account_row_locality_once(*req_it, req_it->command);
                        request_found =
                            m_dram->check_ready(req_it->command, req_it->addr_vec) &&
                            passes_direct_aim_compute_gate(*req_it) &&
                            !should_defer_request(*req_it);
                        req_buffer = &m_aim_buffer;
                    }
                } else {
                    set_write_mode();
                    auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
                    if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
                        request_found =
                            m_dram->check_ready(req_it->command, req_it->addr_vec) &&
                            passes_direct_aim_compute_gate(*req_it) &&
                            !should_defer_request(*req_it);
                        req_buffer = &buffer;
                    }
                }
            }
        }

        // 3. Check if issuing would close an open row needed by active buffer
        if (request_found) {
            if (m_dram->m_command_meta(req_it->command).is_closing) {
                std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(), (req_it->addr_vec).begin() + m_row_addr_idx);
                for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
                    std::vector<Addr_t> _it_rowgroup(_it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
                    if (rowgroup == _it_rowgroup) {
                        request_found = false;
                    }
                }
            }
        }

        return request_found;
    }

};

} // namespace Ramulator
