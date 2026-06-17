#include "data_loader/data_loader.h"
#include "memory_manager/memory_manager.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace aim_cosim {

void AiMDataLoader::load_from_file(const std::string& path, AiMMemoryManager& mm) {
    load_from_file(path, mm, LoadOptions{});
}

void AiMDataLoader::load_from_file(const std::string& path, AiMMemoryManager& mm,
                                   const LoadOptions& options) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("AiMDataLoader: cannot open file: " + path);
    }

    FileHeader header{};
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != MAGIC) {
        throw std::runtime_error("AiMDataLoader: invalid magic in file: " + path);
    }

    int num_banks = static_cast<int>(header.num_banks);
    int rows = static_cast<int>(header.rows_per_bank);
    int tiles = static_cast<int>(header.tiles_per_row);

    // Load bank data
    uint16_t tile_buf[AiMMemoryManager::ELEMENTS_PER_TILE];
    std::vector<uint16_t> row_buf(
        static_cast<size_t>(tiles) * AiMMemoryManager::ELEMENTS_PER_TILE);
    const bool row_sharded =
        options.row_shard_mod > 0 &&
        options.row_shard_offset >= 0 &&
        options.row_shard_offset < options.row_shard_mod;
    const bool selective_row_sharded = row_sharded && options.row_shard_mod > 1;
    const std::streamoff row_bytes =
        static_cast<std::streamoff>(tiles) *
        static_cast<std::streamoff>(sizeof(tile_buf));
    const std::streamoff bank_bytes =
        static_cast<std::streamoff>(rows) * row_bytes;
    const std::streamoff bank_data_begin = static_cast<std::streamoff>(sizeof(FileHeader));
    const std::streamoff bank_data_end =
        bank_data_begin + static_cast<std::streamoff>(num_banks) * bank_bytes;

    if (selective_row_sharded) {
        for (int b = 0; b < num_banks; ++b) {
            const std::streamoff bank_begin =
                bank_data_begin + static_cast<std::streamoff>(b) * bank_bytes;
            for (int r = options.row_shard_offset; r < rows; r += options.row_shard_mod) {
                ifs.seekg(bank_begin + static_cast<std::streamoff>(r) * row_bytes, std::ios::beg);
                ifs.read(reinterpret_cast<char*>(row_buf.data()), row_bytes);
                mm.write_bank_row(b, r, row_buf.data(), tiles);
            }
        }
        ifs.seekg(bank_data_end, std::ios::beg);
    } else {
        for (int b = 0; b < num_banks; ++b) {
            for (int r = 0; r < rows; ++r) {
                if (row_sharded && (r % options.row_shard_mod) != options.row_shard_offset) {
                    ifs.seekg(row_bytes, std::ios::cur);
                    continue;
                }
                ifs.read(reinterpret_cast<char*>(row_buf.data()), row_bytes);
                mm.write_bank_row(b, r, row_buf.data(), tiles);
            }
        }
    }

    // Load global buffer
    for (int t = 0; t < tiles; ++t) {
        ifs.read(reinterpret_cast<char*>(tile_buf), sizeof(tile_buf));
        mm.write_global_buffer(t, tile_buf);
    }

    if (!ifs.good()) {
        throw std::runtime_error("AiMDataLoader: read error or unexpected EOF: " + path);
    }

    // v2 extension: load GPR and bias if data remains
    ifs.peek();
    if (!ifs.eof()) {
        for (int g = 0; g < AiMMemoryManager::NUM_GPRS; ++g) {
            ifs.read(reinterpret_cast<char*>(tile_buf), sizeof(tile_buf));
            if (ifs.gcount() == sizeof(tile_buf)) {
                mm.write_gpr(g, tile_buf);
            }
        }
        // Bias: 16 x uint16
        uint16_t bias_buf[AiMMemoryManager::NUM_BANKS];
        ifs.read(reinterpret_cast<char*>(bias_buf), sizeof(bias_buf));
        if (ifs.gcount() == sizeof(bias_buf)) {
            for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
                mm.write_bias(b, bias_buf[b]);
            }
        }
    }
}

void AiMDataLoader::save_to_file(const std::string& path, const AiMMemoryManager& mm) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("AiMDataLoader: cannot create file: " + path);
    }

    FileHeader header{};
    header.magic = MAGIC;
    header.num_banks = static_cast<uint32_t>(mm.num_banks());
    header.rows_per_bank = static_cast<uint32_t>(mm.max_rows());
    header.tiles_per_row = static_cast<uint32_t>(mm.tiles_per_row());
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Save bank data
    for (int b = 0; b < mm.num_banks(); ++b) {
        for (int r = 0; r < mm.max_rows(); ++r) {
            for (int t = 0; t < mm.tiles_per_row(); ++t) {
                const uint16_t* data = mm.read_bank(b, r, t);
                ofs.write(reinterpret_cast<const char*>(data), AiMMemoryManager::ELEMENTS_PER_TILE * sizeof(uint16_t));
            }
        }
    }

    // Save global buffer
    for (int t = 0; t < mm.tiles_per_row(); ++t) {
        const uint16_t* data = mm.read_global_buffer(t);
        ofs.write(reinterpret_cast<const char*>(data), AiMMemoryManager::ELEMENTS_PER_TILE * sizeof(uint16_t));
    }

    // v2 extension: save GPR and bias
    for (int g = 0; g < AiMMemoryManager::NUM_GPRS; ++g) {
        const uint16_t* data = mm.read_gpr(g);
        ofs.write(reinterpret_cast<const char*>(data), AiMMemoryManager::ELEMENTS_PER_TILE * sizeof(uint16_t));
    }
    for (int b = 0; b < AiMMemoryManager::NUM_BANKS; ++b) {
        uint16_t bias = mm.read_bias(b);
        ofs.write(reinterpret_cast<const char*>(&bias), sizeof(uint16_t));
    }

    if (!ofs.good()) {
        throw std::runtime_error("AiMDataLoader: write error: " + path);
    }
}

} // namespace aim_cosim
