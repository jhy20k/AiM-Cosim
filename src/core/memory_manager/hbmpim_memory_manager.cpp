// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include "memory_manager/hbmpim_memory_manager.h"

#include <algorithm>
#include <stdexcept>

namespace aim_cosim {

void HBMPIMMemoryManager::init(int logical_pseudochannel_id) {
    m_logical_pseudochannel_id = logical_pseudochannel_id;
    reset();
}

void HBMPIMMemoryManager::reset() {
    m_crf.fill(0);
    m_blocks = {};
    m_bank_bursts = {};
    m_open_row.fill(-1);
    m_pc = 0;
    m_loop_counter = 0;
    m_exit_flag = false;
}

void HBMPIMMemoryManager::write_crf(int index, uint32_t value) {
    check_crf_index(index);
    m_crf[static_cast<size_t>(index)] = value;
}

uint32_t HBMPIMMemoryManager::read_crf(int index) const {
    check_crf_index(index);
    return m_crf[static_cast<size_t>(index)];
}

void HBMPIMMemoryManager::write_grf_a(int block_id, int word_index, uint32_t value) {
    check_block_and_word(block_id, word_index);
    fill_burst_scalar(m_blocks[static_cast<size_t>(block_id)].grf_a[static_cast<size_t>(word_index)],
                      value);
}

void HBMPIMMemoryManager::write_grf_b(int block_id, int word_index, uint32_t value) {
    check_block_and_word(block_id, word_index);
    fill_burst_scalar(m_blocks[static_cast<size_t>(block_id)].grf_b[static_cast<size_t>(word_index)],
                      value);
}

uint32_t HBMPIMMemoryManager::read_grf_a(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return m_blocks[static_cast<size_t>(block_id)]
        .grf_a[static_cast<size_t>(word_index)]
        .w[0];
}

uint32_t HBMPIMMemoryManager::read_grf_b(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return m_blocks[static_cast<size_t>(block_id)]
        .grf_b[static_cast<size_t>(word_index)]
        .w[0];
}

void HBMPIMMemoryManager::write_grf_a_burst_words(
    int block_id, int word_index, const std::array<uint32_t, REG_WORDS>& values) {
    check_block_and_word(block_id, word_index);
    fill_burst_words(m_blocks[static_cast<size_t>(block_id)].grf_a[static_cast<size_t>(word_index)],
                     values);
}

void HBMPIMMemoryManager::write_grf_b_burst_words(
    int block_id, int word_index, const std::array<uint32_t, REG_WORDS>& values) {
    check_block_and_word(block_id, word_index);
    fill_burst_words(m_blocks[static_cast<size_t>(block_id)].grf_b[static_cast<size_t>(word_index)],
                     values);
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
HBMPIMMemoryManager::read_grf_a_burst_words(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return copy_burst_words(
        m_blocks[static_cast<size_t>(block_id)].grf_a[static_cast<size_t>(word_index)]);
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
HBMPIMMemoryManager::read_grf_b_burst_words(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return copy_burst_words(
        m_blocks[static_cast<size_t>(block_id)].grf_b[static_cast<size_t>(word_index)]);
}

void HBMPIMMemoryManager::write_grf_a_fp16_lanes(
    int block_id, int word_index, const std::array<uint16_t, FP16_LANES>& values) {
    check_block_and_word(block_id, word_index);
    fill_burst_fp16(
        m_blocks[static_cast<size_t>(block_id)].grf_a[static_cast<size_t>(word_index)], values);
}

void HBMPIMMemoryManager::write_grf_b_fp16_lanes(
    int block_id, int word_index, const std::array<uint16_t, FP16_LANES>& values) {
    check_block_and_word(block_id, word_index);
    fill_burst_fp16(
        m_blocks[static_cast<size_t>(block_id)].grf_b[static_cast<size_t>(word_index)], values);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_grf_a_fp16_lanes(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return copy_burst_fp16(
        m_blocks[static_cast<size_t>(block_id)].grf_a[static_cast<size_t>(word_index)]);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_grf_b_fp16_lanes(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return copy_burst_fp16(
        m_blocks[static_cast<size_t>(block_id)].grf_b[static_cast<size_t>(word_index)]);
}

void HBMPIMMemoryManager::write_srf(int block_id, int word_index, uint32_t value) {
    check_block_and_word(block_id, word_index);
    fill_burst_scalar(m_blocks[static_cast<size_t>(block_id)].srf, value);
}

uint32_t HBMPIMMemoryManager::read_srf(int block_id, int word_index) const {
    check_block_and_word(block_id, word_index);
    return m_blocks[static_cast<size_t>(block_id)].srf.w[static_cast<size_t>(word_index)];
}

void HBMPIMMemoryManager::write_srf_burst_words(
    int block_id, const std::array<uint32_t, REG_WORDS>& values) {
    check_block_and_word(block_id, 0);
    fill_burst_words(m_blocks[static_cast<size_t>(block_id)].srf, values);
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
HBMPIMMemoryManager::read_srf_burst_words(int block_id) const {
    check_block_and_word(block_id, 0);
    std::array<uint32_t, REG_WORDS> values{};
    return copy_burst_words(m_blocks[static_cast<size_t>(block_id)].srf);
}

void HBMPIMMemoryManager::write_srf_fp16_lanes(
    int block_id, const std::array<uint16_t, FP16_LANES>& values) {
    check_block_and_word(block_id, 0);
    fill_burst_fp16(m_blocks[static_cast<size_t>(block_id)].srf, values);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_srf_fp16_lanes(int block_id) const {
    check_block_and_word(block_id, 0);
    return copy_burst_fp16(m_blocks[static_cast<size_t>(block_id)].srf);
}

void HBMPIMMemoryManager::set_m_out(int block_id, uint32_t value) {
    check_block_and_word(block_id, 0);
    fill_burst_scalar(m_blocks[static_cast<size_t>(block_id)].m_out, value);
}

void HBMPIMMemoryManager::set_a_out(int block_id, uint32_t value) {
    check_block_and_word(block_id, 0);
    fill_burst_scalar(m_blocks[static_cast<size_t>(block_id)].a_out, value);
}

uint32_t HBMPIMMemoryManager::read_m_out(int block_id) const {
    check_block_and_word(block_id, 0);
    return m_blocks[static_cast<size_t>(block_id)].m_out.w[0];
}

uint32_t HBMPIMMemoryManager::read_a_out(int block_id) const {
    check_block_and_word(block_id, 0);
    return m_blocks[static_cast<size_t>(block_id)].a_out.w[0];
}

void HBMPIMMemoryManager::write_m_out_fp16_lanes(
    int block_id, const std::array<uint16_t, FP16_LANES>& values) {
    check_block_and_word(block_id, 0);
    fill_burst_fp16(m_blocks[static_cast<size_t>(block_id)].m_out, values);
}

void HBMPIMMemoryManager::write_a_out_fp16_lanes(
    int block_id, const std::array<uint16_t, FP16_LANES>& values) {
    check_block_and_word(block_id, 0);
    fill_burst_fp16(m_blocks[static_cast<size_t>(block_id)].a_out, values);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_m_out_fp16_lanes(int block_id) const {
    check_block_and_word(block_id, 0);
    return copy_burst_fp16(m_blocks[static_cast<size_t>(block_id)].m_out);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_a_out_fp16_lanes(int block_id) const {
    check_block_and_word(block_id, 0);
    return copy_burst_fp16(m_blocks[static_cast<size_t>(block_id)].a_out);
}

void HBMPIMMemoryManager::write_bank_burst_words(
    int bank_id, const std::array<uint32_t, REG_WORDS>& values) {
    check_bank_id(bank_id);
    fill_burst_words(m_bank_bursts[static_cast<size_t>(bank_id)], values);
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
HBMPIMMemoryManager::read_bank_burst_words(int bank_id) const {
    check_bank_id(bank_id);
    return copy_burst_words(m_bank_bursts[static_cast<size_t>(bank_id)]);
}

void HBMPIMMemoryManager::write_bank_fp16_lanes(
    int bank_id, const std::array<uint16_t, FP16_LANES>& values) {
    check_bank_id(bank_id);
    fill_burst_fp16(m_bank_bursts[static_cast<size_t>(bank_id)], values);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::read_bank_fp16_lanes(int bank_id) const {
    check_bank_id(bank_id);
    return copy_burst_fp16(m_bank_bursts[static_cast<size_t>(bank_id)]);
}

void HBMPIMMemoryManager::open_row(int bank_id, int row) {
    check_bank_id(bank_id);
    m_open_row[static_cast<size_t>(bank_id)] = row;
}

void HBMPIMMemoryManager::close_row(int bank_id) {
    check_bank_id(bank_id);
    m_open_row[static_cast<size_t>(bank_id)] = -1;
}

void HBMPIMMemoryManager::close_all_rows() {
    m_open_row.fill(-1);
}

int HBMPIMMemoryManager::get_open_row(int bank_id) const {
    check_bank_id(bank_id);
    return m_open_row[static_cast<size_t>(bank_id)];
}

void HBMPIMMemoryManager::set_pc(uint32_t value) {
    if (value >= static_cast<uint32_t>(CRF_DEPTH)) {
        throw std::out_of_range("HBMPIMMemoryManager pc out of range");
    }
    m_pc = value;
}

void HBMPIMMemoryManager::advance_pc() {
    if (m_pc + 1 >= static_cast<uint32_t>(CRF_DEPTH)) {
        throw std::runtime_error(
            "HBMPIMMemoryManager PC advanced beyond CRF depth without EXIT");
    }
    ++m_pc;
}

bool HBMPIMMemoryManager::valid_block_id(int block_id) {
    return block_id >= 0 && block_id < NUM_PIM_BLOCKS;
}

bool HBMPIMMemoryManager::valid_bank_id(int bank_id) {
    return bank_id >= 0 && bank_id < NUM_BANKS;
}

bool HBMPIMMemoryManager::valid_word_index(int word_index) {
    return word_index >= 0 && word_index < REG_WORDS;
}

int HBMPIMMemoryManager::block_id_from_bank(int bank_id) {
    if (!valid_bank_id(bank_id)) {
        throw std::out_of_range("HBMPIMMemoryManager bank_id out of range");
    }
    return bank_id / 2;
}

void HBMPIMMemoryManager::check_block_and_word(int block_id, int word_index) const {
    if (!valid_block_id(block_id)) {
        throw std::out_of_range("HBMPIMMemoryManager block_id out of range");
    }
    if (!valid_word_index(word_index)) {
        throw std::out_of_range("HBMPIMMemoryManager word_index out of range");
    }
}

void HBMPIMMemoryManager::check_bank_id(int bank_id) const {
    if (!valid_bank_id(bank_id)) {
        throw std::out_of_range("HBMPIMMemoryManager bank_id out of range");
    }
}

void HBMPIMMemoryManager::check_crf_index(int index) const {
    if (index < 0 || index >= CRF_DEPTH) {
        throw std::out_of_range("HBMPIMMemoryManager crf_index out of range");
    }
}

void HBMPIMMemoryManager::fill_burst_words(
    BurstWord16& burst, const std::array<uint32_t, REG_WORDS>& values) {
    std::copy(values.begin(), values.end(), burst.w);
}

std::array<uint32_t, HBMPIMMemoryManager::REG_WORDS>
HBMPIMMemoryManager::copy_burst_words(const BurstWord16& burst) {
    std::array<uint32_t, REG_WORDS> values{};
    std::copy(std::begin(burst.w), std::end(burst.w), values.begin());
    return values;
}

void HBMPIMMemoryManager::fill_burst_fp16(
    BurstWord16& burst, const std::array<uint16_t, FP16_LANES>& values) {
    std::copy(values.begin(), values.end(), burst.fp16);
}

std::array<uint16_t, HBMPIMMemoryManager::FP16_LANES>
HBMPIMMemoryManager::copy_burst_fp16(const BurstWord16& burst) {
    std::array<uint16_t, FP16_LANES> values{};
    std::copy(std::begin(burst.fp16), std::end(burst.fp16), values.begin());
    return values;
}

void HBMPIMMemoryManager::fill_burst_scalar(BurstWord16& burst, uint32_t value) {
    for (auto& word : burst.w) {
        word = value;
    }
}

} // namespace aim_cosim
