#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include "base/request.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace aim_cosim::HBMPIMTraceLowering {

inline std::vector<Ramulator::Request> expand_repeated_trigger(
    const Ramulator::Request& trigger_req,
    size_t repeat_count) {
    if (trigger_req.type != Ramulator::Type::HBM_PIM ||
        trigger_req.hbm_pim_info.opcode != Ramulator::HBMPIMOpcode::HBMPIM_TRIGGER) {
        throw std::runtime_error("expand_repeated_trigger expects HBMPIM_TRIGGER request");
    }

    std::vector<Ramulator::Request> out;
    out.reserve(repeat_count);
    for (size_t i = 0; i < repeat_count; ++i) {
        auto req = trigger_req;
        req.host_req_id = (trigger_req.host_req_id >= 0)
            ? trigger_req.host_req_id + static_cast<int>(i)
            : trigger_req.host_req_id;
        out.push_back(req);
    }
    return out;
}

} // namespace aim_cosim::HBMPIMTraceLowering
