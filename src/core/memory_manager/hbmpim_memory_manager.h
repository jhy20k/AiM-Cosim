#pragma once

// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.

#include <array>
#include <cstdint>

namespace aim_cosim {

class HBMPIMMemoryManager {
public:
    static constexpr int NUM_BANKS = 16;
    static constexpr int NUM_PIM_BLOCKS = 8;
    static constexpr int CRF_DEPTH = 32;
    static constexpr int REG_WORDS = 8;
    static constexpr int FP16_LANES = REG_WORDS * 2;

    struct BurstWord16 {
        union {
            uint32_t w[REG_WORDS];
            uint16_t fp16[FP16_LANES];
        };

        BurstWord16() : w{} {}
    };

    struct BlockState {
        std::array<BurstWord16, REG_WORDS> grf_a{};
        std::array<BurstWord16, REG_WORDS> grf_b{};
        BurstWord16 srf{};
        BurstWord16 m_out{};
        BurstWord16 a_out{};
    };

    void init(int logical_pseudochannel_id);
    void reset();

    int logical_pseudochannel_id() const { return m_logical_pseudochannel_id; }

    void write_crf(int index, uint32_t value);
    uint32_t read_crf(int index) const;

    void write_grf_a(int block_id, int word_index, uint32_t value);
    void write_grf_b(int block_id, int word_index, uint32_t value);
    uint32_t read_grf_a(int block_id, int word_index) const;
    uint32_t read_grf_b(int block_id, int word_index) const;
    void write_grf_a_burst_words(int block_id, int word_index,
                                 const std::array<uint32_t, REG_WORDS>& values);
    void write_grf_b_burst_words(int block_id, int word_index,
                                 const std::array<uint32_t, REG_WORDS>& values);
    std::array<uint32_t, REG_WORDS> read_grf_a_burst_words(int block_id, int word_index) const;
    std::array<uint32_t, REG_WORDS> read_grf_b_burst_words(int block_id, int word_index) const;
    void write_grf_a_fp16_lanes(int block_id, int word_index,
                                const std::array<uint16_t, FP16_LANES>& values);
    void write_grf_b_fp16_lanes(int block_id, int word_index,
                                const std::array<uint16_t, FP16_LANES>& values);
    std::array<uint16_t, FP16_LANES> read_grf_a_fp16_lanes(int block_id, int word_index) const;
    std::array<uint16_t, FP16_LANES> read_grf_b_fp16_lanes(int block_id, int word_index) const;

    void write_srf(int block_id, int word_index, uint32_t value);
    uint32_t read_srf(int block_id, int word_index) const;
    void write_srf_burst_words(int block_id,
                               const std::array<uint32_t, REG_WORDS>& values);
    std::array<uint32_t, REG_WORDS> read_srf_burst_words(int block_id) const;
    void write_srf_fp16_lanes(int block_id,
                              const std::array<uint16_t, FP16_LANES>& values);
    std::array<uint16_t, FP16_LANES> read_srf_fp16_lanes(int block_id) const;

    void set_m_out(int block_id, uint32_t value);
    void set_a_out(int block_id, uint32_t value);
    uint32_t read_m_out(int block_id) const;
    uint32_t read_a_out(int block_id) const;
    void write_m_out_fp16_lanes(int block_id, const std::array<uint16_t, FP16_LANES>& values);
    void write_a_out_fp16_lanes(int block_id, const std::array<uint16_t, FP16_LANES>& values);
    std::array<uint16_t, FP16_LANES> read_m_out_fp16_lanes(int block_id) const;
    std::array<uint16_t, FP16_LANES> read_a_out_fp16_lanes(int block_id) const;

    void write_bank_burst_words(int bank_id, const std::array<uint32_t, REG_WORDS>& values);
    std::array<uint32_t, REG_WORDS> read_bank_burst_words(int bank_id) const;
    void write_bank_fp16_lanes(int bank_id, const std::array<uint16_t, FP16_LANES>& values);
    std::array<uint16_t, FP16_LANES> read_bank_fp16_lanes(int bank_id) const;

    void open_row(int bank_id, int row);
    void close_row(int bank_id);
    void close_all_rows();
    int get_open_row(int bank_id) const;

    uint32_t pc() const { return m_pc; }
    void set_pc(uint32_t value);
    void advance_pc();

    uint32_t loop_counter() const { return m_loop_counter; }
    void set_loop_counter(uint32_t value) { m_loop_counter = value; }

    bool exit_flag() const { return m_exit_flag; }
    void set_exit_flag(bool value) { m_exit_flag = value; }

    static bool valid_block_id(int block_id);
    static bool valid_bank_id(int bank_id);
    static bool valid_word_index(int word_index);
    static int block_id_from_bank(int bank_id);

private:
    int m_logical_pseudochannel_id = -1;
    std::array<uint32_t, CRF_DEPTH> m_crf{};
    std::array<BlockState, NUM_PIM_BLOCKS> m_blocks{};
    std::array<BurstWord16, NUM_BANKS> m_bank_bursts{};
    std::array<int, NUM_BANKS> m_open_row{};
    uint32_t m_pc = 0;
    uint32_t m_loop_counter = 0;
    bool m_exit_flag = false;

    static void fill_burst_words(BurstWord16& burst,
                                 const std::array<uint32_t, REG_WORDS>& values);
    static std::array<uint32_t, REG_WORDS> copy_burst_words(const BurstWord16& burst);
    static void fill_burst_fp16(BurstWord16& burst,
                                const std::array<uint16_t, FP16_LANES>& values);
    static std::array<uint16_t, FP16_LANES> copy_burst_fp16(const BurstWord16& burst);
    static void fill_burst_scalar(BurstWord16& burst, uint32_t value);

    void check_block_and_word(int block_id, int word_index) const;
    void check_bank_id(int bank_id) const;
    void check_crf_index(int index) const;
};

} // namespace aim_cosim
