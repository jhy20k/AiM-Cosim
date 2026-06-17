// Future work: HBM-PIM support is an experimental scaffold and is not part of
// the validated GDDR6-AiM RTL flow.
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "Rank.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;

namespace {

constexpr unsigned kPimRegRa = 0x3fff;

BurstType make_uniform_burst(uint16_t value) {
    BurstType burst;
    burst.set(value, value, value, value, value, value, value, value,
              value, value, value, value, value, value, value, value);
    return burst;
}

BurstType make_srf_burst() {
    BurstType burst;
    burst.set(0x4000, 0x4000, 0x4000, 0x4000,
              0x4000, 0x4000, 0x4000, 0x4000,
              0x3c00, 0x3c00, 0x3c00, 0x3c00,
              0x3c00, 0x3c00, 0x3c00, 0x3c00);
    return burst;
}

std::string burst_hex(const BurstType& burst) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << std::setw(4) << static_cast<unsigned>(burst.u16Data_[i]);
    }
    return oss.str();
}

bool all_lane_eq(const BurstType& burst, uint16_t value) {
    for (int i = 0; i < 16; ++i) {
        if (burst.u16Data_[i] != value) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    try {
        auto mem = std::make_shared<MultiChannelMemorySystem>(
            "/home/jhlee/PIMSimulator/ini/HBM2_samsung_2M_16B_x64.ini",
            "/home/jhlee/PIMSimulator/system_hbm_1ch.ini",
            "/home/jhlee/PIMSimulator",
            "hbmpim_stage5f_compare",
            256 * 16);

        PIMKernel kernel(mem, 1, 1);

        BurstType null_burst;
        BurstType hab_pim_ctrl;
        BurstType hab_ctrl;
        kernel.setControl(&hab_pim_ctrl, true, 0, false, false);
        kernel.setControl(&hab_ctrl, false, 0, false, false);

        BurstType grf_a0 = make_uniform_burst(0x3c00);
        BurstType grf_b0 = make_uniform_burst(0x4000);
        BurstType srf = make_srf_burst();
        BurstType grf_a3_readback;

        std::vector<PIMCmd> program{
            PIMCmd(PIMCmdType::MOV, PIMOpdType::GRF_A, PIMOpdType::GRF_A, 0, 1, 0, 0, 0),
            PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::GRF_B,
                   0, 2, 1, 0),
            PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::SRF_M,
                   0, 3, 2, 0),
            PIMCmd(PIMCmdType::EXIT, 0),
        };

        kernel.changePIMMode(dramMode::SB, dramMode::HAB);
        kernel.addTransactionAll(true, 0, 0, kPimRegRa, 0x08, "WRITE_GRF_A0", &grf_a0, true);
        kernel.addTransactionAll(true, 0, 0, kPimRegRa, 0x18, "WRITE_GRF_B0", &grf_b0, true);
        kernel.addTransactionAll(true, 0, 0, kPimRegRa, 0x01, "WRITE_SRF", &srf, true);
        kernel.programCrf(program);
        kernel.addTransactionAll(true, 0, 0, kPimRegRa, 0x00, "ENTER_HAB_PIM", &hab_pim_ctrl,
                                 true);

        for (int i = 0; i < 4; ++i) {
            kernel.addTransactionAll(false, 0, 0, 0, 0, "TRIGGER", &null_burst, true);
        }

        kernel.addTransactionAll(true, 0, 0, kPimRegRa, 0x00, "LEAVE_HAB_PIM", &hab_ctrl, true);
        kernel.changePIMMode(dramMode::HAB, dramMode::SB);
        kernel.addTransactionAll(false, 0, 0, kPimRegRa, 0x0b, "READ_GRF_A3", &grf_a3_readback,
                                 true);
        kernel.runPIM();

        auto* rank = mem->channels[0]->ranks->at(0);
        const auto& srf_state = rank->pimRank->pimBlocks[0].srf;

        const bool grf_ok = all_lane_eq(grf_a3_readback, 0x4600);
        bool srf_ok = true;
        for (int i = 0; i < 8; ++i) {
            srf_ok &= (srf_state.u16Data_[i] == 0x4000);
        }
        for (int i = 8; i < 16; ++i) {
            srf_ok &= (srf_state.u16Data_[i] == 0x3c00);
        }

        std::cout << "PIMSIM_TOTAL_CYCLES=" << kernel.getCycle() << "\n";
        std::cout << "PIMSIM_GRF_A3_HEX=" << burst_hex(grf_a3_readback) << "\n";
        std::cout << "PIMSIM_SRF_HEX=" << burst_hex(srf_state) << "\n";
        std::cout << "PIMSIM_STATUS=" << ((grf_ok && srf_ok) ? "PASS" : "FAIL") << "\n";
        return (grf_ok && srf_ok) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "PIMSIM_STATUS=ERROR\n";
        std::cerr << "PIMSIM_ERROR=" << e.what() << "\n";
        return 2;
    }
}
