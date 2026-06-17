#include "controller/functional_dispatcher.h"
#include <cassert>
#include <cstring>
#include "memory_system/impl/aim_region_registry.h"

namespace aim_cosim {

void FunctionalDispatcher::init(AiMMemoryManager* mm, IComputeEngine* ce,
                                ResultLogger* logger) {
    m_mm = mm;
    m_ce = ce;
    m_logger = logger;
}

uint64_t FunctionalDispatcher::dispatch(const Ramulator::Request& req, int64_t sim_cycle) {
    switch (req.command) {
        case CMD_ACT: case CMD_ACT4: case CMD_ACT16:
            handle_act(req); return 0;
        case CMD_PRE: case CMD_PRE4: case CMD_PREA:
            handle_pre(req); return 0;
        case CMD_WR:
            handle_wr(req); return 0;
        case CMD_WRGB:
            handle_wrgb(req); return 0;
        case CMD_WRMAC16:
            handle_wrmac16(req); return 0;
        case CMD_WRA16:
            handle_wra16(req); return 0;
        case CMD_RD: case CMD_RDA:
            handle_rd(req, sim_cycle); return 0;
        case CMD_RDCP:
            handle_rdcp(req); return 0;
        case CMD_WRCP:
            handle_wrcp(req); return 0;
        case CMD_MAC:
            return handle_mac(req, sim_cycle);
        case CMD_MAC16:
            return handle_mac16(req, sim_cycle);
        case CMD_AF16:
            return handle_af16(req, sim_cycle);
        case CMD_EWMUL16:
            return handle_ewmul16(req, sim_cycle);
        case CMD_RDMAC16:
            handle_rdmac16(req, sim_cycle); return 0;
        case CMD_RDAF16:
            handle_rdaf16(req, sim_cycle); return 0;
        case CMD_SYNC: case CMD_EOC:
            flush_pending_mac16(); return 0;
        default:
            return 0;
    }
}

// =========================================================================
// Row Management
// =========================================================================

void FunctionalDispatcher::handle_act(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    if (req.command == CMD_ACT) {
        m_mm->open_row(coord.bank_id, coord.row);
    } else if (req.command == CMD_ACT4) {
        int bg = req.addr_vec[1];
        for (int b = 0; b < 4; ++b)
            m_mm->open_row(bg * 4 + b, coord.row);
    } else if (req.command == CMD_ACT16) {
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b)
            m_mm->open_row(b, coord.row);
    }
}

void FunctionalDispatcher::handle_pre(const Ramulator::Request& req) {
    if (req.command == CMD_PREA) {
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b)
            m_mm->close_row(b);
    } else if (req.command == CMD_PRE4) {
        int bg = req.addr_vec[1];
        for (int b = 0; b < 4; ++b)
            m_mm->close_row(bg * 4 + b);
    } else {
        m_mm->close_row(addr_to_bank_id(req.addr_vec));
    }
}

// =========================================================================
// Data Writes
// =========================================================================

void FunctionalDispatcher::handle_wr(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int row = m_mm->get_open_row(coord.bank_id);
    if (row < 0) return;
    if (req.GPR_addr_0 >= 0 && req.GPR_addr_0 < AiMMemoryManager::NUM_GPRS) {
        const uint16_t* gpr_data = m_mm->read_gpr(static_cast<int>(req.GPR_addr_0));
        m_mm->write_bank(coord.bank_id, row, coord.tile_col, gpr_data);
    }
}

void FunctionalDispatcher::handle_wrgb(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int gpr_id = static_cast<int>(req.GPR_addr_0) + coord.tile_col;
    if (gpr_id >= 0 && gpr_id < AiMMemoryManager::NUM_GPRS) {
        const uint16_t* gpr_data = m_mm->read_gpr(gpr_id);
        m_mm->write_global_buffer(coord.tile_col, gpr_data);
    }
}

void FunctionalDispatcher::handle_wrmac16(const Ramulator::Request& req) {
    if (req.GPR_addr_0 >= 0 && req.GPR_addr_0 < AiMMemoryManager::NUM_GPRS) {
        const uint16_t* gpr_data = m_mm->read_gpr(static_cast<int>(req.GPR_addr_0));
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b)
            m_mm->write_bias(b, gpr_data[b]);
    }
}

void FunctionalDispatcher::handle_wra16(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    if (req.GPR_addr_0 >= 0 && req.GPR_addr_0 < AiMMemoryManager::NUM_GPRS) {
        const uint16_t* gpr_data = m_mm->read_gpr(static_cast<int>(req.GPR_addr_0));
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
            int row = m_mm->get_open_row(b);
            if (row >= 0)
                m_mm->write_bank(b, row, coord.tile_col, gpr_data);
        }
    }
}

// =========================================================================
// Data Reads
// =========================================================================

void FunctionalDispatcher::handle_rd(const Ramulator::Request& req, int64_t sim_cycle) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int row = m_mm->get_open_row(coord.bank_id);
    if (row < 0) return;
    const uint16_t* data = m_mm->read_bank(coord.bank_id, row, coord.tile_col);
    emit_data_packet(req.command, coord.bank_id, row, coord.tile_col, data, sim_cycle);
}

void FunctionalDispatcher::handle_rdcp(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int row = m_mm->get_open_row(coord.bank_id);
    if (row < 0) return;
    const uint16_t* data = m_mm->read_bank(coord.bank_id, row, coord.tile_col);
    m_mm->write_global_buffer(coord.tile_col, data);
}

void FunctionalDispatcher::handle_wrcp(const Ramulator::Request& req) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int row = m_mm->get_open_row(coord.bank_id);
    if (row < 0) return;
    const uint16_t* gb_data = m_mm->read_global_buffer(coord.tile_col);
    m_mm->write_bank(coord.bank_id, row, coord.tile_col, gb_data);
}

// =========================================================================
// Compute Operations
// =========================================================================

uint64_t FunctionalDispatcher::handle_mac(const Ramulator::Request& req, int64_t /*sim_cycle*/) {
    flush_pending_mac16();

    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int bank_id = coord.bank_id;
    int row = m_mm->get_open_row(bank_id);
    if (row < 0) return 0;

    const uint16_t* weight = m_mm->read_bank(bank_id, row, coord.tile_col);
    const uint16_t* vector = m_mm->read_global_buffer(coord.tile_col);
    uint16_t bias = m_mm->read_bias(bank_id);

    bool is_first_tile = (coord.tile_col == 0);
    auto result = m_ce->execute_mac(weight, vector, !is_first_tile, is_first_tile, bias);
    if (result.success) {
        m_mm->store_mac_result(bank_id, result.data.data());
        return result.rtl_cycles;
    }
    return 0;
}

uint64_t FunctionalDispatcher::handle_mac16(const Ramulator::Request& req, int64_t /*sim_cycle*/) {
    auto coord = m_mm->map_addr_vec(req.addr_vec);
    int tile_col = coord.tile_col;

    bool all_rows_open = true;
    for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
        int row = m_mm->get_open_row(b);
        if (row >= 0) {
            m_mac16_weight_ptrs[b] = m_mm->read_bank(b, row, tile_col);
        } else {
            all_rows_open = false;
        }
    }

    const uint16_t* vector = m_mm->read_global_buffer(tile_col);

    for (int b = 0; b < 16; ++b)
        m_mac16_biases[b] = m_mm->read_bias(b);

    bool is_first_tile = (tile_col == 0);
    IComputeEngine::ComputeResult result{};
    if (all_rows_open && m_ce->supports_deferred_mac16()) {
        result = m_ce->enqueue_mac16(m_mac16_weight_ptrs, vector,
                                     !is_first_tile, is_first_tile, m_mac16_biases.data());
        return result.rtl_cycles;
    } else if (all_rows_open) {
        result = m_ce->execute_mac16(m_mac16_weight_ptrs, vector,
                                     !is_first_tile, is_first_tile, m_mac16_biases.data());
    } else {
        // Preserve the legacy partial-open-row behavior by falling back to the
        // staging buffer when any bank lacks an active row.
        flush_pending_mac16();
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
            int row = m_mm->get_open_row(b);
            if (row >= 0) {
                const uint16_t* w = m_mm->read_bank(b, row, tile_col);
                std::memcpy(m_mac16_weights[b].data(), w, 16 * sizeof(uint16_t));
            }
        }
        result = m_ce->execute_mac16(m_mac16_weights, vector,
                                     !is_first_tile, is_first_tile, m_mac16_biases.data());
    }
    if (result.success) {
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
            m_mm->store_mac_result_lane(b, result.data[b]);
        }
        return result.rtl_cycles;
    }
    return 0;
}

uint64_t FunctionalDispatcher::handle_af16(const Ramulator::Request& req, int64_t /*sim_cycle*/) {
    flush_pending_mac16();

    uint8_t af_type = static_cast<uint8_t>(req.afm >= 0 ? req.afm : 0);

    auto result = m_ce->execute_af(af_type);
    if (result.success) {
        for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
            m_mm->store_af_result_lane(b, result.data[b]);
        }
        return result.rtl_cycles;
    }
    return 0;
}

uint64_t FunctionalDispatcher::handle_ewmul16(const Ramulator::Request& req, int64_t /*sim_cycle*/) {
    flush_pending_mac16();

    int tile_col = req.addr_vec.size() >= 5 ? req.addr_vec[4] : 0;
    if (tile_col < 0) tile_col = 0;

    uint64_t max_rtl_cycles = 0;
    const int start_bg = (req.ewmul_bg > 0) ? (req.ewmul_bg - 1) : 0;
    const int end_bg = (req.ewmul_bg > 0)
                           ? req.ewmul_bg
                           : static_cast<int>(Ramulator::kEwmulBankMap.size());

    for (int bg = start_bg; bg < end_bg; ++bg) {
        if (bg < 0 || bg >= static_cast<int>(Ramulator::kEwmulBankMap.size())) {
            continue;
        }
        const auto triple = Ramulator::kEwmulBankMap[bg];
        const int row_a = m_mm->get_open_row(triple.src_a);
        const int row_b = m_mm->get_open_row(triple.src_b);
        if (row_a < 0 || row_b < 0 || row_a != row_b) {
            continue;
        }

        const uint16_t* a = m_mm->read_bank(triple.src_a, row_a, tile_col);
        const uint16_t* b_data = m_mm->read_bank(triple.src_b, row_b, tile_col);

        auto result = m_ce->execute_ewmul(a, b_data);
        if (result.success) {
            m_mm->write_bank(triple.dest, row_a, tile_col, result.data.data());
            if (result.rtl_cycles > max_rtl_cycles) {
                max_rtl_cycles = result.rtl_cycles;
            }
        }
    }

    return max_rtl_cycles;
}

// =========================================================================
// Result Reads
// =========================================================================

void FunctionalDispatcher::handle_rdmac16(const Ramulator::Request& req, int64_t sim_cycle) {
    flush_pending_mac16();

    for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
        const uint16_t* data = m_mm->read_mac_result(b);
        emit_data_packet(req.command, b, /*row=*/-1, /*col=*/-1, data, sim_cycle);
    }
}

void FunctionalDispatcher::handle_rdaf16(const Ramulator::Request& req, int64_t sim_cycle) {
    flush_pending_mac16();

    if (m_logger && m_logger->config().rdaf16_bank0_only) {
        const uint16_t* data = m_mm->read_af_result(0);
        emit_data_packet(req.command, 0, /*row=*/-1, /*col=*/-1, data, sim_cycle);
        return;
    }

    for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
        const uint16_t* data = m_mm->read_af_result(b);
        emit_data_packet(req.command, b, /*row=*/-1, /*col=*/-1, data, sim_cycle);
    }
}

// =========================================================================
// Helpers
// =========================================================================

void FunctionalDispatcher::flush_pending_mac16() {
    if (!m_ce || !m_ce->has_pending_mac16()) {
        return;
    }

    auto result = m_ce->flush_mac16_pipeline();
    if (!result.success) {
        return;
    }

    for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
        m_mm->store_mac_result_lane(b, result.data[b]);
    }
}

int FunctionalDispatcher::addr_to_bank_id(const std::vector<int>& addr_vec) const {
    assert(addr_vec.size() >= 3);
    return addr_vec[1] * 4 + addr_vec[2];
}

void FunctionalDispatcher::emit_data_packet(int command, int bank_id, int row, int col,
                                            const uint16_t* data, int64_t sim_cycle) {
    ResultPacket pkt{};
    pkt.cycle = static_cast<uint64_t>(sim_cycle);
    pkt.command = command;
    pkt.bank_id = bank_id;
    pkt.row = row;
    pkt.col = col;
    std::memcpy(pkt.data.data(), data, 16 * sizeof(uint16_t));
    m_logger->log_result(pkt);
}

} // namespace aim_cosim
