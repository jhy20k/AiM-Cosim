#include "memory_manager/memory_manager.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace aim_cosim {

void AiMMemoryManager::init(int num_banks, int max_rows, int tiles_per_row) {
    num_banks_ = num_banks;
    max_rows_ = max_rows;
    tiles_per_row_ = tiles_per_row;

    // Allocate bank_data_: [num_banks][max_rows] — rows are lazy-allocated
    bank_data_.resize(num_banks_);
    for (auto& bank : bank_data_) {
        bank.resize(max_rows_);
        // Tiles within each row are NOT allocated here (lazy)
    }

    // Global buffer and GPR
    global_buffer_.resize(tiles_per_row_);
    gpr_storage_.resize(NUM_GPRS);

    reset();
}

void AiMMemoryManager::init_random(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint16_t> dist(0, 0xFFFF);

    // Store seed for lazy bank_data allocation (deterministic per-row RNG)
    random_init_seed_ = seed;

    // Randomize global buffer
    for (auto& tile : global_buffer_) {
        for (auto& elem : tile) {
            elem = dist(rng);
        }
    }

    // Randomize GPR
    for (auto& gpr : gpr_storage_) {
        for (auto& elem : gpr) {
            elem = dist(rng);
        }
    }

    // Randomize bias
    for (auto& b : bias_) {
        b = dist(rng);
    }
}

void AiMMemoryManager::init_zero() {
    for (auto& tile : global_buffer_) {
        tile.fill(0);
    }
    for (auto& gpr : gpr_storage_) {
        gpr.fill(0);
    }
    bias_.fill(0);
    for (auto& bank_mac : mac_results_) {
        bank_mac.fill(0);
    }
    for (auto& bank_af : af_results_) {
        bank_af.fill(0);
    }
    open_row_.fill(-1);
}

void AiMMemoryManager::reset() {
    init_zero();
}

// ---------------------------------------------------------------------------
// Bank data access
// ---------------------------------------------------------------------------

void AiMMemoryManager::ensure_row_allocated(int bank_id, int row) {
    assert(bank_id >= 0 && bank_id < num_banks_);
    assert(row >= 0 && row < max_rows_);
    auto& row_tiles = bank_data_[bank_id][row];
    if (row_tiles.empty()) {
        row_tiles.resize(tiles_per_row_);
        if (random_init_seed_ != 0) {
            // Deterministic per-(bank, row) seed derived from init seed
            uint32_t row_seed = random_init_seed_ ^
                (static_cast<uint32_t>(bank_id) * 131071u +
                 static_cast<uint32_t>(row) * 7u);
            std::mt19937 row_rng(row_seed);
            std::uniform_int_distribution<uint16_t> dist(0, 0xFFFF);
            for (auto& tile : row_tiles) {
                for (auto& elem : tile) {
                    elem = dist(row_rng);
                }
            }
        } else {
            for (auto& tile : row_tiles) {
                tile.fill(0);
            }
        }
    }
}

void AiMMemoryManager::write_bank(int bank_id, int row, int col,
                                   const uint16_t data[ELEMENTS_PER_TILE]) {
    ensure_row_allocated(bank_id, row);
    assert(col >= 0 && col < tiles_per_row_);
    std::memcpy(bank_data_[bank_id][row][col].data(), data,
                ELEMENTS_PER_TILE * sizeof(uint16_t));
}

void AiMMemoryManager::write_bank_row(int bank_id, int row, const uint16_t* data,
                                      int num_tiles) {
    ensure_row_allocated(bank_id, row);
    assert(num_tiles >= 0 && num_tiles <= tiles_per_row_);
    std::memcpy(bank_data_[bank_id][row].data()->data(), data,
                static_cast<size_t>(num_tiles) * ELEMENTS_PER_TILE * sizeof(uint16_t));
}

const uint16_t* AiMMemoryManager::read_bank(int bank_id, int row, int col) const {
    assert(bank_id >= 0 && bank_id < num_banks_);
    assert(row >= 0 && row < max_rows_);
    assert(col >= 0 && col < tiles_per_row_);
    const auto& row_tiles = bank_data_[bank_id][row];
    if (row_tiles.empty()) {
        // Return zeros for unallocated rows
        static const std::array<uint16_t, ELEMENTS_PER_TILE> zeros{};
        return zeros.data();
    }
    return bank_data_[bank_id][row][col].data();
}

// ---------------------------------------------------------------------------
// Global Buffer
// ---------------------------------------------------------------------------

void AiMMemoryManager::write_global_buffer(int tile_col,
                                            const uint16_t data[ELEMENTS_PER_TILE]) {
    assert(tile_col >= 0 && tile_col < tiles_per_row_);
    std::memcpy(global_buffer_[tile_col].data(), data,
                ELEMENTS_PER_TILE * sizeof(uint16_t));
}

const uint16_t* AiMMemoryManager::read_global_buffer(int tile_col) const {
    assert(tile_col >= 0 && tile_col < tiles_per_row_);
    return global_buffer_[tile_col].data();
}

// ---------------------------------------------------------------------------
// GPR
// ---------------------------------------------------------------------------

void AiMMemoryManager::write_gpr(int gpr_id, const uint16_t data[ELEMENTS_PER_TILE]) {
    assert(gpr_id >= 0 && gpr_id < NUM_GPRS);
    std::memcpy(gpr_storage_[gpr_id].data(), data,
                ELEMENTS_PER_TILE * sizeof(uint16_t));
}

const uint16_t* AiMMemoryManager::read_gpr(int gpr_id) const {
    assert(gpr_id >= 0 && gpr_id < NUM_GPRS);
    return gpr_storage_[gpr_id].data();
}

// ---------------------------------------------------------------------------
// Bias
// ---------------------------------------------------------------------------

void AiMMemoryManager::write_bias(int bank_id, uint16_t bias) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    bias_[bank_id] = bias;
}

uint16_t AiMMemoryManager::read_bias(int bank_id) const {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    return bias_[bank_id];
}

// ---------------------------------------------------------------------------
// Result storage
// ---------------------------------------------------------------------------

void AiMMemoryManager::store_mac_result(int bank_id,
                                         const uint16_t result[ELEMENTS_PER_TILE]) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    std::memcpy(mac_results_[bank_id].data(), result,
                ELEMENTS_PER_TILE * sizeof(uint16_t));
}

void AiMMemoryManager::store_mac_result_lane(int bank_id, uint16_t result) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    mac_results_[bank_id].fill(0);
    mac_results_[bank_id][0] = result;
}

const uint16_t* AiMMemoryManager::read_mac_result(int bank_id) const {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    return mac_results_[bank_id].data();
}

void AiMMemoryManager::store_af_result(int bank_id,
                                        const uint16_t result[ELEMENTS_PER_TILE]) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    std::memcpy(af_results_[bank_id].data(), result,
                ELEMENTS_PER_TILE * sizeof(uint16_t));
}

void AiMMemoryManager::store_af_result_lane(int bank_id, uint16_t result) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    af_results_[bank_id].fill(0);
    af_results_[bank_id][0] = result;
}

const uint16_t* AiMMemoryManager::read_af_result(int bank_id) const {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    return af_results_[bank_id].data();
}

// ---------------------------------------------------------------------------
// Row open tracking
// ---------------------------------------------------------------------------

void AiMMemoryManager::open_row(int bank_id, int row) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    open_row_[bank_id] = row;
}

void AiMMemoryManager::close_row(int bank_id) {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    open_row_[bank_id] = -1;
}

int AiMMemoryManager::get_open_row(int bank_id) const {
    assert(bank_id >= 0 && bank_id < NUM_BANKS);
    return open_row_[bank_id];
}

// ---------------------------------------------------------------------------
// Address mapping: ramulator2 addr_vec → MemoryCoord
// addr_vec layout: [channel, bankgroup, bank, row, column]
// ---------------------------------------------------------------------------

AiMMemoryManager::MemoryCoord
AiMMemoryManager::map_addr_vec(const std::vector<int>& addr_vec) const {
    // addr_vec indices per GDDR6: channel=0, bankgroup=1, bank=2, row=3, column=4
    assert(addr_vec.size() >= 5);
    MemoryCoord coord;
    coord.bank_id = addr_vec[1] * 4 + addr_vec[2];  // bankgroup * 4 + bank
    coord.row = addr_vec[3];
    coord.tile_col = addr_vec[4];
    return coord;
}

} // namespace aim_cosim
