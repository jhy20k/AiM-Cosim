#pragma once

#include "dram/dram.h"

#include <cstdint>
#include <string>

#include <yaml-cpp/yaml.h>

namespace aim_cosim {

struct ArchitectureInitContext {
    YAML::Node root_config;
    YAML::Node verilator_config;
    YAML::Node memory_manager_config;
    YAML::Node architecture_config;
    Ramulator::IDRAM* dram = nullptr;
    int channel_id = -1;
    std::string result_log_path;
};

struct ComputeConsumerPair {
    const char* producer_cmd = "";
    const char* consumer_cmd = "";
    const char* level = "channel";
    uint32_t rtl_cycle_upper_bound = 0;
    bool supports_gating = false;
};

} // namespace aim_cosim
