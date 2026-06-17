#pragma once

#include <cstdint>
#include <string>

namespace aim_cosim {

class AiMMemoryManager;  // forward declaration

class AiMDataLoader {
public:
    // Binary data file format (v2 — backward-compatible):
    // [header: magic(4B) + num_banks(4B) + rows_per_bank(4B) + tiles_per_row(4B)]
    // [bank0_data: rows × tiles × 16 × sizeof(uint16_t)]
    // [bank1_data: ...]
    // [global_buffer: tiles × 16 × sizeof(uint16_t)]
    // [gpr_data: 32 × 16 × sizeof(uint16_t)]          (optional, v2)
    // [bias_data: 16 × sizeof(uint16_t)]               (optional, v2)
    //
    // v1 files (no GPR/bias) are auto-detected by checking remaining file size.
    static constexpr uint32_t MAGIC = 0x41494D44;  // "AIMD"

    struct FileHeader {
        uint32_t magic;
        uint32_t num_banks;
        uint32_t rows_per_bank;
        uint32_t tiles_per_row;
    };

    struct LoadOptions {
        int row_shard_mod = 0;
        int row_shard_offset = 0;
    };

    static void load_from_file(const std::string& path, AiMMemoryManager& mm);
    static void load_from_file(const std::string& path, AiMMemoryManager& mm,
                               const LoadOptions& options);
    static void save_to_file(const std::string& path, const AiMMemoryManager& mm);
};

} // namespace aim_cosim
