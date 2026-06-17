`timescale 1ns / 1ps

// =============================================================================
// PIM MAC Tree Top Module
// - Existing scalar MAC path (PU -> AdderTree -> Accumulator)
// - Added all-bank vector16 MAC/AF path for MAC16 / AF16
// =============================================================================

module pim_mac_tree (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          i_start,
    input  logic          i_acc_en,
    input  logic          i_ewmul_en,
    input  logic          i_bias_en,
    input  logic [15:0]   i_bias,
    input  logic          i_latch_sel_valid,
    input  logic          i_latch_sel,
    input  logic          i_latch_seed_valid,
    input  logic          i_latch_seed,

    // AF control (Phase 5)
    input  logic          i_af_en,
    input  logic [1:0]    i_af_type,
    input  logic [15:0]   i_af_slope_bf16,

    // Scalar packed inputs (16 elements * 16 bits)
    input  logic [255:0]  i_wgt_flat,
    input  logic [255:0]  i_vec_flat,

    // All-bank vector16 packed inputs
    input  logic          i_all_bank_vector_en,
    input  logic [4095:0] i_wgt_bank_flat,
    input  logic [255:0]  i_bias_bank_flat,

    // Outputs
    output logic          o_done,
    output logic [15:0]   o_result,
    output logic [255:0]  o_mac_result_flat,
    output logic [255:0]  o_result_flat,
    output logic [255:0]  o_af_result_flat,

    // Debug Ports
    output logic          dbg_tree_done,
    output logic          dbg_tree_sign,
    output logic [31:0]   dbg_tree_mag,
    output logic [9:0]    dbg_tree_exp
);

    // ========================================================================
    // Unpack shared vector/scalar weights (256b -> 16 x 16b)
    // ========================================================================
    logic [15:0] w_man [0:15];
    logic [9:0]  w_exp [0:15];
    logic [15:0] w_sign;

    logic [15:0] v_man [0:15];
    logic [9:0]  v_exp [0:15];
    logic [15:0] v_sign;

    // ========================================================================
    // Unpack all-bank weights (16 banks x 16 BF16)
    // ========================================================================
    logic [15:0] abk_w_man [0:15][0:15];
    logic [9:0]  abk_w_exp [0:15][0:15];
    logic [15:0] abk_w_sign [0:15];

    function automatic logic [15:0] bf16_mul_rne(
        input logic [15:0] a,
        input logic [15:0] b
    );
        logic sign_a;
        logic sign_b;
        logic sign_out;
        logic [7:0] exp_a;
        logic [7:0] exp_b;
        logic [6:0] frac_a;
        logic [6:0] frac_b;
        logic a_is_zero;
        logic b_is_zero;
        logic a_is_inf;
        logic b_is_inf;
        logic a_is_nan;
        logic b_is_nan;
        logic [7:0] real_exp_a;
        logic [7:0] real_exp_b;
        logic [7:0] man_a;
        logic [7:0] man_b;
        logic [15:0] prod;
        logic norm_shift;
        logic [15:0] norm_prod;
        logic guard_bit;
        logic sticky_bit;
        logic lsb_bit;
        logic round_up;
        logic [7:0] frac_round;
        logic [6:0] frac_out;
        integer exp_work;

        sign_a = a[15];
        sign_b = b[15];
        sign_out = sign_a ^ sign_b;
        exp_a = a[14:7];
        exp_b = b[14:7];
        frac_a = a[6:0];
        frac_b = b[6:0];

        a_is_zero = (exp_a == 8'd0) && (frac_a == 7'd0);
        b_is_zero = (exp_b == 8'd0) && (frac_b == 7'd0);
        a_is_inf = (exp_a == 8'hFF) && (frac_a == 7'd0);
        b_is_inf = (exp_b == 8'hFF) && (frac_b == 7'd0);
        a_is_nan = (exp_a == 8'hFF) && (frac_a != 7'd0);
        b_is_nan = (exp_b == 8'hFF) && (frac_b != 7'd0);

        if (a_is_nan || b_is_nan || ((a_is_inf || b_is_inf) && (a_is_zero || b_is_zero))) begin
            bf16_mul_rne = 16'h7FC1;
        end else if (a_is_inf || b_is_inf) begin
            bf16_mul_rne = {sign_out, 8'hFF, 7'h00};
        end else if (a_is_zero || b_is_zero) begin
            bf16_mul_rne = {sign_out, 8'h00, 7'h00};
        end else begin
            real_exp_a = (exp_a == 8'd0) ? 8'd1 : exp_a;
            real_exp_b = (exp_b == 8'd0) ? 8'd1 : exp_b;
            man_a = {(exp_a != 8'd0), frac_a};
            man_b = {(exp_b != 8'd0), frac_b};
            prod = man_a * man_b;
            norm_shift = prod[15];
            norm_prod = norm_shift ? (prod >> 1) : prod;
            exp_work = $signed({1'b0, real_exp_a}) + $signed({1'b0, real_exp_b}) - 127;
            if (norm_shift) begin
                exp_work = exp_work + 1;
            end

            guard_bit = norm_prod[6];
            sticky_bit = |norm_prod[5:0];
            lsb_bit = norm_prod[7];
            round_up = guard_bit && (sticky_bit || lsb_bit);
            frac_round = {1'b0, norm_prod[13:7]} + (round_up ? 8'd1 : 8'd0);
            frac_out = frac_round[6:0];
            if (frac_round[7]) begin
                frac_out = 7'd0;
                exp_work = exp_work + 1;
            end

            if (exp_work >= 255) begin
                bf16_mul_rne = {sign_out, 8'hFF, 7'h00};
            end else if (exp_work <= 0) begin
                bf16_mul_rne = {sign_out, 8'h00, 7'h00};
            end else begin
                bf16_mul_rne = {sign_out, exp_work[7:0], frac_out};
            end
        end
    endfunction

    generate
        for (genvar k = 0; k < 16; k = k + 1) begin : UNPACK
            assign w_sign[k] = i_wgt_flat[16*k + 15];
            assign w_exp[k]  = {2'b0, i_wgt_flat[16*k + 14 : 16*k + 7]};
            assign w_man[k]  = {9'b0, i_wgt_flat[16*k + 6 : 16*k]};

            assign v_sign[k] = i_vec_flat[16*k + 15];
            assign v_exp[k]  = {2'b0, i_vec_flat[16*k + 14 : 16*k + 7]};
            assign v_man[k]  = {9'b0, i_vec_flat[16*k + 6 : 16*k]};
        end
    endgenerate

    generate
        for (genvar bank = 0; bank < 16; bank = bank + 1) begin : UNPACK_ABK_BANK
            for (genvar lane = 0; lane < 16; lane = lane + 1) begin : UNPACK_ABK_LANE
                localparam int FLAT_IDX = (bank * 16) + lane;
                assign abk_w_sign[bank][lane] = i_wgt_bank_flat[16*FLAT_IDX + 15];
                assign abk_w_exp[bank][lane]  = {2'b0, i_wgt_bank_flat[16*FLAT_IDX + 14 : 16*FLAT_IDX + 7]};
                assign abk_w_man[bank][lane]  = {9'b0, i_wgt_bank_flat[16*FLAT_IDX + 6 : 16*FLAT_IDX]};
            end
        end
    endgenerate

    // ========================================================================
    // Shared scalar datapath PU Array
    // ========================================================================
    logic [15:0] pu_sign;
    logic [9:0]  pu_exp [0:15];
    logic [15:0] pu_man [0:15];

    generate
        for (genvar k = 0; k < 16; k = k + 1) begin : PU_ARRAY
            assign pu_sign[k] = w_sign[k] ^ v_sign[k];
            assign pu_exp[k] = w_exp[k] + v_exp[k] - 10'd127;
            assign pu_man[k] = (w_man[k] | 16'h0080) * (v_man[k] | 16'h0080);
        end
    endgenerate

    // ========================================================================
    // All-bank vector16 PU Arrays
    // ========================================================================
    logic [15:0] abk_pu_sign [0:15];
    logic [9:0]  abk_pu_exp [0:15][0:15];
    logic [15:0] abk_pu_man [0:15][0:15];

    generate
        for (genvar bank = 0; bank < 16; bank = bank + 1) begin : ABK_PU_BANK
            for (genvar lane = 0; lane < 16; lane = lane + 1) begin : ABK_PU_LANE
                assign abk_pu_sign[bank][lane] = abk_w_sign[bank][lane] ^ v_sign[lane];
                assign abk_pu_exp[bank][lane] = abk_w_exp[bank][lane] + v_exp[lane] - 10'd127;
                assign abk_pu_man[bank][lane] = (abk_w_man[bank][lane] | 16'h0080) * (v_man[lane] | 16'h0080);
            end
        end
    endgenerate

    // ========================================================================
    // Start mode split
    // ========================================================================
    logic scalar_mac_start;
    logic abk_mac_start;
    logic scalar_af_start;
    logic abk_af_start;
    logic ewmul_start;

    assign scalar_mac_start = i_start & (~i_af_en) & (~i_ewmul_en) & (~i_all_bank_vector_en);
    assign abk_mac_start = i_start & (~i_af_en) & (~i_ewmul_en) & i_all_bank_vector_en;
    assign scalar_af_start = i_start & i_af_en & (~i_all_bank_vector_en);
    assign abk_af_start = i_start & i_af_en & i_all_bank_vector_en;
    assign ewmul_start = i_start & i_ewmul_en;

    // ========================================================================
    // Scalar Adder Tree Instance
    // ========================================================================
    logic        tree_done;
    logic        tree_sign_out;
    logic [31:0] tree_man_out;
    logic [9:0]  tree_exp_out;

    adder_tree u_adder_tree (
        .clk          (clk),
        .rst_n        (rst_n),
        .i_start      (scalar_mac_start),
        .i_sign       (pu_sign),
        .i_exp        (pu_exp),
        .i_man        (pu_man),
        .o_done       (tree_done),
        .o_sign       (tree_sign_out),
        .o_result_man (tree_man_out),
        .o_common_exp (tree_exp_out)
    );

    // ========================================================================
    // All-bank vector16 Adder Tree Instances
    // ========================================================================
    logic [15:0] abk_tree_done;
    logic [15:0] abk_tree_sign_out;
    logic [31:0] abk_tree_man_out [0:15];
    logic [9:0]  abk_tree_exp_out [0:15];

    generate
        for (genvar bank = 0; bank < 16; bank = bank + 1) begin : ABK_TREE
            adder_tree u_adder_tree_bank (
                .clk          (clk),
                .rst_n        (rst_n),
                .i_start      (abk_mac_start),
                .i_sign       (abk_pu_sign[bank]),
                .i_exp        (abk_pu_exp[bank]),
                .i_man        (abk_pu_man[bank]),
                .o_done       (abk_tree_done[bank]),
                .o_sign       (abk_tree_sign_out[bank]),
                .o_result_man (abk_tree_man_out[bank]),
                .o_common_exp (abk_tree_exp_out[bank])
            );
        end
    endgenerate

    // ========================================================================
    // Accumulator control pipeline
    // ========================================================================
    // The adder tree presents its output to the accumulator two clock edges
    // after the MAC start is accepted by this top level.  Keep the MAC
    // transaction controls in the same pipeline so dependent MACAB streams can
    // be issued at the 2-command-cycle (1 ns at tCK=500 ps) interval without
    // sampling the next transaction's acc/bias controls.
    logic        acc_en_d0, acc_en_d1;
    logic        acc_bias_en_d0, acc_bias_en_d1;
    logic [15:0] acc_bias_bf16_d0, acc_bias_bf16_d1;
    logic [255:0] acc_bias_bank_flat_d0, acc_bias_bank_flat_d1;
    logic        acc_latch_sel_valid_d0, acc_latch_sel_valid_d1;
    logic        acc_latch_sel_d0, acc_latch_sel_d1;
    logic        acc_latch_seed_valid_d0, acc_latch_seed_valid_d1;
    logic        acc_latch_seed_d0, acc_latch_seed_d1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_en_d0 <= 1'b0;
            acc_en_d1 <= 1'b0;
            acc_bias_en_d0 <= 1'b0;
            acc_bias_en_d1 <= 1'b0;
            acc_bias_bf16_d0 <= 16'd0;
            acc_bias_bf16_d1 <= 16'd0;
            acc_bias_bank_flat_d0 <= 256'd0;
            acc_bias_bank_flat_d1 <= 256'd0;
            acc_latch_sel_valid_d0 <= 1'b0;
            acc_latch_sel_valid_d1 <= 1'b0;
            acc_latch_sel_d0 <= 1'b0;
            acc_latch_sel_d1 <= 1'b0;
            acc_latch_seed_valid_d0 <= 1'b0;
            acc_latch_seed_valid_d1 <= 1'b0;
            acc_latch_seed_d0 <= 1'b0;
            acc_latch_seed_d1 <= 1'b0;
        end else begin
            acc_en_d1 <= acc_en_d0;
            acc_bias_en_d1 <= acc_bias_en_d0;
            acc_bias_bf16_d1 <= acc_bias_bf16_d0;
            acc_bias_bank_flat_d1 <= acc_bias_bank_flat_d0;
            acc_latch_sel_valid_d1 <= acc_latch_sel_valid_d0;
            acc_latch_sel_d1 <= acc_latch_sel_d0;
            acc_latch_seed_valid_d1 <= acc_latch_seed_valid_d0;
            acc_latch_seed_d1 <= acc_latch_seed_d0;

            if (scalar_mac_start || abk_mac_start) begin
                acc_en_d0 <= i_acc_en;
                acc_bias_en_d0 <= i_bias_en;
                acc_bias_bf16_d0 <= i_bias;
                acc_bias_bank_flat_d0 <= i_bias_bank_flat;
                acc_latch_sel_valid_d0 <= i_latch_sel_valid;
                acc_latch_sel_d0 <= i_latch_sel;
                acc_latch_seed_valid_d0 <= i_latch_seed_valid;
                acc_latch_seed_d0 <= i_latch_seed;
            end else begin
                acc_en_d0 <= 1'b0;
                acc_bias_en_d0 <= 1'b0;
                acc_bias_bf16_d0 <= 16'd0;
                acc_bias_bank_flat_d0 <= 256'd0;
                acc_latch_sel_valid_d0 <= 1'b0;
                acc_latch_sel_d0 <= 1'b0;
                acc_latch_seed_valid_d0 <= 1'b0;
                acc_latch_seed_d0 <= 1'b0;
            end
        end
    end

    // ========================================================================
    // Scalar Accumulator Instance
    // ========================================================================
    logic        acc_done;
    logic [15:0] acc_result;
    logic        acc_latch0_sign_unused;
    logic [49:0] acc_latch0_mag_unused;
    logic        acc_latch1_sign_unused;
    logic [49:0] acc_latch1_mag_unused;

    accumulator u_accumulator (
        .clk       (clk),
        .rst_n     (rst_n),
        .i_start   (tree_done),
        .i_acc_en  (acc_en_d1),
        .i_acc_clear(1'b0),
        .i_bank_lr_valid(1'b0),
        .i_bank_lr(1'b0),
        .i_latch_sel_valid(acc_latch_sel_valid_d1),
        .i_latch_sel(acc_latch_sel_d1),
        .i_latch_seed_valid(acc_latch_seed_valid_d1),
        .i_latch_seed(acc_latch_seed_d1),
        .i_bias_en (acc_bias_en_d1),
        .i_bias    (acc_bias_bf16_d1),
        .i_max_exp (tree_exp_out),
        .i_sign    (tree_sign_out),
        .i_ma_sum  (tree_man_out),
        .o_done    (acc_done),
        .o_result  (acc_result),
        .o_latch0_sign(acc_latch0_sign_unused),
        .o_latch0_mag (acc_latch0_mag_unused),
        .o_latch1_sign(acc_latch1_sign_unused),
        .o_latch1_mag (acc_latch1_mag_unused)
    );

    // ========================================================================
    // All-bank vector16 Accumulator Instances
    // ========================================================================
    logic [15:0] abk_acc_done;
    logic [15:0] abk_acc_result [0:15];
    logic [15:0] abk_acc_latch0_sign_unused;
    logic [15:0] abk_acc_latch1_sign_unused;
    logic [49:0] abk_acc_latch0_mag_unused [0:15];
    logic [49:0] abk_acc_latch1_mag_unused [0:15];

    generate
        for (genvar bank = 0; bank < 16; bank = bank + 1) begin : ABK_ACC
            accumulator u_accumulator_bank (
                .clk       (clk),
                .rst_n     (rst_n),
                .i_start   (abk_tree_done[bank]),
                .i_acc_en  (acc_en_d1),
                .i_acc_clear(1'b0),
                .i_bank_lr_valid(1'b0),
                .i_bank_lr(1'b0),
                .i_latch_sel_valid(acc_latch_sel_valid_d1),
                .i_latch_sel(acc_latch_sel_d1),
                .i_latch_seed_valid(acc_latch_seed_valid_d1),
                .i_latch_seed(acc_latch_seed_d1),
                .i_bias_en (acc_bias_en_d1),
                .i_bias    (acc_bias_bank_flat_d1[16*bank +: 16]),
                .i_max_exp (abk_tree_exp_out[bank]),
                .i_sign    (abk_tree_sign_out[bank]),
                .i_ma_sum  (abk_tree_man_out[bank]),
                .o_done    (abk_acc_done[bank]),
                .o_result  (abk_acc_result[bank]),
                .o_latch0_sign(abk_acc_latch0_sign_unused[bank]),
                .o_latch0_mag (abk_acc_latch0_mag_unused[bank]),
                .o_latch1_sign(abk_acc_latch1_sign_unused[bank]),
                .o_latch1_mag (abk_acc_latch1_mag_unused[bank])
            );
        end
    endgenerate

    // ========================================================================
    // AF Path
    // ========================================================================
    logic [15:0] last_mac_result_bf16;
    logic [15:0] af_result;
    logic        af_done;
    logic [15:0] abk_af_result [0:15];
    logic [15:0] abk_af_done;
    logic [255:0] last_mac_result_flat_q;
    logic [255:0] mac_result_flat_q;
    logic [255:0] af_result_flat_q;
    logic [255:0] ewmul_result_flat_d;
    logic [255:0] ewmul_result_flat_q;
    logic        ewmul_done_q;

    generate
        for (genvar ew_k = 0; ew_k < 16; ew_k = ew_k + 1) begin : EWMUL_ARRAY
            assign ewmul_result_flat_d[16*ew_k +: 16] =
                bf16_mul_rne(i_wgt_flat[16*ew_k +: 16], i_vec_flat[16*ew_k +: 16]);
        end
    endgenerate

    assign o_mac_result_flat = mac_result_flat_q;
    assign o_result_flat = ewmul_result_flat_q;
    assign o_af_result_flat = af_result_flat_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            last_mac_result_flat_q <= 256'd0;
            mac_result_flat_q <= 256'd0;
            af_result_flat_q <= 256'd0;
            ewmul_result_flat_q <= 256'd0;
            ewmul_done_q <= 1'b0;
        end else begin
            ewmul_done_q <= ewmul_start;
            if (ewmul_start) begin
                ewmul_result_flat_q <= ewmul_result_flat_d;
            end
            if (acc_done) begin
                last_mac_result_flat_q <= {16{acc_result}};
                mac_result_flat_q <= {16{acc_result}};
            end
            if (abk_acc_done[0]) begin
                for (int bank = 0; bank < 16; bank = bank + 1) begin
                    last_mac_result_flat_q[16*bank +: 16] <= abk_acc_result[bank];
                    mac_result_flat_q[16*bank +: 16] <= abk_acc_result[bank];
                end
            end
            if (af_done) begin
                af_result_flat_q <= {16{af_result}};
            end
            if (abk_af_done[0]) begin
                for (int bank = 0; bank < 16; bank = bank + 1) begin
                    af_result_flat_q[16*bank +: 16] <= abk_af_result[bank];
                end
            end
        end
    end

    activation_unit u_activation_unit (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (scalar_af_start),
        .i_af_type      (i_af_type),
        .i_af_slope_bf16(i_af_slope_bf16),
        .i_af_in_bf16   (last_mac_result_bf16),
        .o_done         (af_done),
        .o_af_out_bf16  (af_result)
    );

    generate
        for (genvar bank = 0; bank < 16; bank = bank + 1) begin : ABK_AF
            activation_unit u_activation_unit_bank (
                .clk            (clk),
                .rst_n          (rst_n),
                .i_start        (abk_af_start),
                .i_af_type      (i_af_type),
                .i_af_slope_bf16(i_af_slope_bf16),
                .i_af_in_bf16   (last_mac_result_flat_q[16*bank +: 16]),
                .o_done         (abk_af_done[bank]),
                .o_af_out_bf16  (abk_af_result[bank])
            );
        end
    endgenerate

    // ========================================================================
    // Output arbitration (scalar / vector16 MAC, AF, EWMUL)
    // ========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            last_mac_result_bf16 <= 16'd0;
            o_done <= 1'b0;
            o_result <= 16'd0;
        end else begin
            o_done <= 1'b0;

            if (acc_done) begin
                last_mac_result_bf16 <= acc_result;
                o_done <= 1'b1;
                o_result <= acc_result;
            end

            if (abk_acc_done[0]) begin
                last_mac_result_bf16 <= abk_acc_result[0];
                o_done <= 1'b1;
                o_result <= abk_acc_result[0];
            end

            if (af_done) begin
                o_done <= 1'b1;
                o_result <= af_result;
            end

            if (abk_af_done[0]) begin
                o_done <= 1'b1;
                o_result <= abk_af_result[0];
            end

            if (ewmul_done_q) begin
                o_done <= 1'b1;
                o_result <= ewmul_result_flat_q[15:0];
            end
        end
    end

    // ========================================================================
    // Debug Port Assignments
    // ========================================================================
    assign dbg_tree_done = tree_done;
    assign dbg_tree_sign = tree_sign_out;
    assign dbg_tree_mag  = tree_man_out;
    assign dbg_tree_exp  = tree_exp_out;

endmodule
