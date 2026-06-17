`timescale 1ns / 1ps

// =============================================================================
// WITHOUT Multi-Latch Accumulator Wrapper
// - Interface-compatible wrapper around `accumulator`
// - Forces single-latch mode (latch0 fixed) to model baseline behavior
// =============================================================================

module without_multilatch_accumulator (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        i_start,
    input  logic        i_acc_en,
    input  logic        i_acc_clear,

    input  logic        i_bank_lr_valid,
    input  logic        i_bank_lr,
    input  logic        i_latch_sel_valid, // kept for interface compatibility
    input  logic        i_latch_sel,       // kept for interface compatibility

    input  logic        i_bias_en,
    input  logic [15:0] i_bias,
    input  logic [9:0]  i_max_exp,

    input  logic        i_sign,
    input  logic [31:0] i_ma_sum,

    output logic        o_done,
    output logic [15:0] o_result,

    output logic        o_latch0_sign,
    output logic [49:0] o_latch0_mag,
    output logic        o_latch1_sign,
    output logic [49:0] o_latch1_mag
);

    accumulator u_accumulator (
        .clk              (clk),
        .rst_n            (rst_n),
        .i_start          (i_start),
        .i_acc_en         (i_acc_en),
        .i_acc_clear      (i_acc_clear),
        .i_bank_lr_valid  (i_bank_lr_valid),
        .i_bank_lr        (i_bank_lr),
        .i_latch_sel_valid(1'b1), // force external selection
        .i_latch_sel      (1'b0), // fixed latch0
        .i_latch_seed_valid(1'b0),
        .i_latch_seed     (1'b0),
        .i_bias_en        (i_bias_en),
        .i_bias           (i_bias),
        .i_max_exp        (i_max_exp),
        .i_sign           (i_sign),
        .i_ma_sum         (i_ma_sum),
        .o_done           (o_done),
        .o_result         (o_result),
        .o_latch0_sign    (o_latch0_sign),
        .o_latch0_mag     (o_latch0_mag),
        .o_latch1_sign    (o_latch1_sign),
        .o_latch1_mag     (o_latch1_mag)
    );

    // Prevent unused-input lint warnings while preserving interface shape.
    logic unused_latch_inputs;
    assign unused_latch_inputs = i_latch_sel_valid ^ i_latch_sel;

endmodule
