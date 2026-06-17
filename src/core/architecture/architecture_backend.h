#pragma once

#include "architecture/architecture_types.h"
#include "base/request.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aim_cosim {

class IArchitectureBackend {
public:
    virtual ~IArchitectureBackend() = default;

    virtual void init(const ArchitectureInitContext& ctx) = 0;
    virtual void reset() = 0;

    virtual void on_command_issued(const Ramulator::Request& req,
                                   uint64_t sim_issue_cycle) = 0;

    virtual void flush_pending_timing() = 0;
    virtual bool can_defer_consumer(int command, uint64_t now_cycle) const = 0;
    virtual void note_consumer_deferred(int command, uint64_t cycles = 1) = 0;

    virtual std::vector<ComputeConsumerPair> get_compute_consumer_pairs() const = 0;
    virtual size_t& divergence_count_ref() = 0;
};

} // namespace aim_cosim
