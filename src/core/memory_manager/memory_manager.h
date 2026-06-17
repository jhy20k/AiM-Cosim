#pragma once

// AiMMemoryManager — BF16 data storage with AiMX zone separation
// See doc2 Section 2.5.1 for full design

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace aim_cosim {

class AiMMemoryManager {
public:
    // AiMX device memory map zones
    enum class Zone { INST, CFR, GPR, DA, DMA };

    // Hardware constants
    static constexpr int NUM_BANKS = 16;
    static constexpr int MAX_ROWS = 16384;
    static constexpr int TILES_PER_ROW = 64;      // 1024 cols / 16 elements per tile
    static constexpr int ELEMENTS_PER_TILE = 16;   // 256-bit / 16-bit = 16 BF16s
    static constexpr int NUM_GPRS = 32;

    AiMMemoryManager() = default;
    ~AiMMemoryManager() = default;

    // Initialization
    void init(int num_banks, int max_rows, int tiles_per_row);
    void init_random(uint32_t seed);
    void init_zero();
    void reset();

    // Bank data (AiM Zone — weight storage)
    void write_bank(int bank_id, int row, int col, const uint16_t data[ELEMENTS_PER_TILE]);
    void write_bank_row(int bank_id, int row, const uint16_t* data, int num_tiles);
    const uint16_t* read_bank(int bank_id, int row, int col) const;

    // Global Buffer (GPR Zone — activation vector)
    void write_global_buffer(int tile_col, const uint16_t data[ELEMENTS_PER_TILE]);
    const uint16_t* read_global_buffer(int tile_col) const;

    // GPR registers
    void write_gpr(int gpr_id, const uint16_t data[ELEMENTS_PER_TILE]);
    const uint16_t* read_gpr(int gpr_id) const;

    // Bias (per-bank, loaded via ISR_WR_BIAS / WRMAC16)
    void write_bias(int bank_id, uint16_t bias);
    uint16_t read_bias(int bank_id) const;

    // MAC/AF result storage
    void store_mac_result(int bank_id, const uint16_t result[ELEMENTS_PER_TILE]);
    void store_mac_result_lane(int bank_id, uint16_t result);
    const uint16_t* read_mac_result(int bank_id) const;
    void store_af_result(int bank_id, const uint16_t result[ELEMENTS_PER_TILE]);
    void store_af_result_lane(int bank_id, uint16_t result);
    const uint16_t* read_af_result(int bank_id) const;

    // Row open tracking (ACT/PRE commands)
    void open_row(int bank_id, int row);
    void close_row(int bank_id);
    int get_open_row(int bank_id) const;

    // Address mapping: ramulator2 addr_vec → Memory Manager coordinates
    struct MemoryCoord {
        int bank_id;    // bankgroup * 4 + bank
        int row;
        int tile_col;
    };
    MemoryCoord map_addr_vec(const std::vector<int>& addr_vec) const;

    // Accessors
    int num_banks() const { return num_banks_; }
    int max_rows() const { return max_rows_; }
    int tiles_per_row() const { return tiles_per_row_; }

private:
    int num_banks_ = NUM_BANKS;
    int max_rows_ = MAX_ROWS;
    int tiles_per_row_ = TILES_PER_ROW;

    // AiM Zone (DRAM): bank data [bank][row][tile][16 BF16]
    // Lazy allocation: outer vector sized at init, inner rows allocated on access
    std::vector<std::vector<std::vector<std::array<uint16_t, ELEMENTS_PER_TILE>>>> bank_data_;

    // GPR Zone: Global Buffer + GPR registers
    std::vector<std::array<uint16_t, ELEMENTS_PER_TILE>> global_buffer_;
    std::vector<std::array<uint16_t, ELEMENTS_PER_TILE>> gpr_storage_;

    // Per-bank bias (loaded by WRMAC16)
    std::array<uint16_t, NUM_BANKS> bias_{};

    // Result storage
    std::array<std::array<uint16_t, ELEMENTS_PER_TILE>, NUM_BANKS> mac_results_{};
    std::array<std::array<uint16_t, ELEMENTS_PER_TILE>, NUM_BANKS> af_results_{};

    // Row open tracking
    std::array<int, NUM_BANKS> open_row_{};  // -1 = closed

    // Random init seed (0 = zero-fill on lazy alloc, >0 = deterministic random)
    uint32_t random_init_seed_ = 0;

    // Ensure row is allocated (lazy)
    void ensure_row_allocated(int bank_id, int row);
};

} // namespace aim_cosim
