#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace aim_cosim {

class AiMComputeScoreboard {
public:
    struct ComputeOp {
        int command = -1;
        int bank_id = -1;
        int ewmul_bg = 0;
    };

    struct SlotSnapshot {
        int64_t issue_ready_clk = -1;
        int64_t visible_ready_clk = -1;
        uint64_t produced_epoch = 0;
    };

    enum class GateStatus {
        Pass,
        ProducerNotReady,
        ConsumerNotReady,
        MissingProducer,
    };

    static constexpr int NUM_BANKS = 16;
    static constexpr int NUM_EWMUL_BGS = 4;

    void configure(int mac_cmd, int mac16_cmd, int af16_cmd, int ewmul16_cmd,
                   int rdmac16_cmd, int rdaf16_cmd) {
        m_mac_cmd = mac_cmd;
        m_mac16_cmd = mac16_cmd;
        m_af16_cmd = af16_cmd;
        m_ewmul16_cmd = ewmul16_cmd;
        m_rdmac16_cmd = rdmac16_cmd;
        m_rdaf16_cmd = rdaf16_cmd;
        reset();
    }

    void reset() {
        m_mac16_slot = {};
        m_af16_slot = {};
        m_mac_slots = {};
        m_ewmul_bg_slots = {};
    }

    bool is_compute_producer_command(int command) const {
        return command == m_mac_cmd || command == m_mac16_cmd ||
               command == m_af16_cmd || command == m_ewmul16_cmd;
    }

    bool is_compute_consumer_command(int command) const {
        return command == m_rdmac16_cmd || command == m_rdaf16_cmd;
    }

    void note_ready(const ComputeOp& op, int64_t issue_clk, uint64_t rtl_cycles) {
        const int64_t ready_clk = issue_clk + static_cast<int64_t>(rtl_cycles);
        for (ComputeSlot* slot : resolve_slots(op)) {
            slot->issue_ready_clk = std::max(slot->issue_ready_clk, ready_clk);
            slot->visible_ready_clk = std::max(slot->visible_ready_clk, ready_clk);
            slot->produced_epoch += 1;
        }
    }

    GateStatus gate_status(const ComputeOp& op, int64_t now_clk) const {
        if (is_compute_producer_command(op.command)) {
            for (const ComputeSlot* slot : resolve_slots_const(op)) {
                if (slot->issue_ready_clk != -1 && now_clk < slot->issue_ready_clk) {
                    return GateStatus::ProducerNotReady;
                }
            }
            return GateStatus::Pass;
        }

        if (is_compute_consumer_command(op.command)) {
            for (const ComputeSlot* slot : resolve_slots_const(op)) {
                if (slot->produced_epoch == 0) {
                    return GateStatus::MissingProducer;
                }
                if (now_clk < slot->visible_ready_clk) {
                    return GateStatus::ConsumerNotReady;
                }
            }
            return GateStatus::Pass;
        }

        return GateStatus::Pass;
    }

    SlotSnapshot mac16_slot() const { return snapshot(m_mac16_slot); }
    SlotSnapshot af16_slot() const { return snapshot(m_af16_slot); }
    SlotSnapshot mac_slot(int bank_id) const { return snapshot(m_mac_slots.at(bank_id)); }
    SlotSnapshot ewmul_bg_slot(int bg_id) const {
        return snapshot(m_ewmul_bg_slots.at(bg_id));
    }

private:
    struct ComputeSlot {
        int64_t issue_ready_clk = -1;
        int64_t visible_ready_clk = -1;
        uint64_t produced_epoch = 0;
    };

    int m_mac_cmd = -1;
    int m_mac16_cmd = -1;
    int m_af16_cmd = -1;
    int m_ewmul16_cmd = -1;
    int m_rdmac16_cmd = -1;
    int m_rdaf16_cmd = -1;

    ComputeSlot m_mac16_slot{};
    ComputeSlot m_af16_slot{};
    std::array<ComputeSlot, NUM_BANKS> m_mac_slots{};
    std::array<ComputeSlot, NUM_EWMUL_BGS> m_ewmul_bg_slots{};

    static SlotSnapshot snapshot(const ComputeSlot& slot) {
        return {
            slot.issue_ready_clk,
            slot.visible_ready_clk,
            slot.produced_epoch,
        };
    }

    std::vector<ComputeSlot*> resolve_slots(const ComputeOp& op) {
        std::vector<ComputeSlot*> slots;

        if (op.command == m_mac_cmd) {
            if (op.bank_id >= 0 && op.bank_id < NUM_BANKS) {
                slots.push_back(&m_mac_slots[op.bank_id]);
            }
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_mac16_cmd) {
            for (auto& slot : m_mac_slots) {
                slots.push_back(&slot);
            }
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_af16_cmd) {
            slots.push_back(&m_af16_slot);
        } else if (op.command == m_ewmul16_cmd) {
            if (op.ewmul_bg > 0) {
                const int bg_id = op.ewmul_bg - 1;
                if (bg_id >= 0 && bg_id < NUM_EWMUL_BGS) {
                    slots.push_back(&m_ewmul_bg_slots[bg_id]);
                }
            } else {
                for (auto& slot : m_ewmul_bg_slots) {
                    slots.push_back(&slot);
                }
            }
        } else if (op.command == m_rdmac16_cmd) {
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_rdaf16_cmd) {
            slots.push_back(&m_af16_slot);
        }

        return slots;
    }

    std::vector<const ComputeSlot*> resolve_slots_const(const ComputeOp& op) const {
        std::vector<const ComputeSlot*> slots;

        if (op.command == m_mac_cmd) {
            if (op.bank_id >= 0 && op.bank_id < NUM_BANKS) {
                slots.push_back(&m_mac_slots[op.bank_id]);
            }
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_mac16_cmd) {
            for (const auto& slot : m_mac_slots) {
                slots.push_back(&slot);
            }
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_af16_cmd) {
            slots.push_back(&m_af16_slot);
        } else if (op.command == m_ewmul16_cmd) {
            if (op.ewmul_bg > 0) {
                const int bg_id = op.ewmul_bg - 1;
                if (bg_id >= 0 && bg_id < NUM_EWMUL_BGS) {
                    slots.push_back(&m_ewmul_bg_slots[bg_id]);
                }
            } else {
                for (const auto& slot : m_ewmul_bg_slots) {
                    slots.push_back(&slot);
                }
            }
        } else if (op.command == m_rdmac16_cmd) {
            slots.push_back(&m_mac16_slot);
        } else if (op.command == m_rdaf16_cmd) {
            slots.push_back(&m_af16_slot);
        }

        return slots;
    }
};

} // namespace aim_cosim
