`timescale 1ns / 1ps

// =============================================================================
// Parameterized Accumulator Module
// - Based on accumulator.v (50-bit Q30 baseline)
// - MAG_WIDTH: magnitude bit width (default 50)
// - Q_POINT:   hidden bit position / normalization target (default 30)
//
// Configurations:
//   mag50/Q30: MAG_WIDTH=50, Q_POINT=30 (matches original accumulator.v)
//   mag37/Q30: MAG_WIDTH=37, Q_POINT=30 (reduced guard bits)
//   mag32/Q24: MAG_WIDTH=32, Q_POINT=24 (guard=8, frac=17)
// =============================================================================

module accumulator_param #(
    parameter MAG_WIDTH = 50,
    parameter Q_POINT  = 30
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        i_start,
    input  logic        i_acc_en,
    input  logic        i_acc_clear,

    input  logic        i_bank_lr_valid,
    input  logic        i_bank_lr,
    input  logic        i_latch_sel_valid,
    input  logic        i_latch_sel,
    input  logic        i_latch_seed_valid,
    input  logic        i_latch_seed,

    input  logic        i_bias_en,
    input  logic [15:0] i_bias,
    input  logic [9:0]  i_max_exp,

    input  logic        i_sign,
    input  logic [31:0] i_ma_sum,

    output logic        o_done,
    output logic [15:0] o_result,

    // Debug/verification visibility for phase-1 features
    output logic        o_latch0_sign,
    output logic [MAG_WIDTH-1:0] o_latch0_mag,
    output logic        o_latch1_sign,
    output logic [MAG_WIDTH-1:0] o_latch1_mag
);

    // -------------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------------
    localparam INPUT_SHIFT = Q_POINT - 22;

    // Safety assertions
    initial begin
        if (MAG_WIDTH < Q_POINT + 2) begin
            $fatal(1, "MAG_WIDTH (%0d) must be >= Q_POINT + 2 (%0d)", MAG_WIDTH, Q_POINT + 2);
        end
        if (Q_POINT < 9) begin
            $fatal(1, "Q_POINT (%0d) must be >= 9", Q_POINT);
        end
    end

    // -------------------------------------------------------------------------
    // Internal state: L/R lane accumulators (sign + magnitude + exponent)
    // -------------------------------------------------------------------------
    logic        s_accum_l_sign;
    logic [MAG_WIDTH-1:0] s_accum_l_mag;
    logic [9:0]  s_accum_l_exp;

    logic        s_accum_r_sign;
    logic [MAG_WIDTH-1:0] s_accum_r_mag;
    logic [9:0]  s_accum_r_exp;

    // Multi-latch checkpoints for long accumulation chains (>2KB use case)
    logic        s_latch0_sign;
    logic [MAG_WIDTH-1:0] s_latch0_mag;
    logic [9:0]  s_latch0_exp;

    logic        s_latch1_sign;
    logic [MAG_WIDTH-1:0] s_latch1_mag;
    logic [9:0]  s_latch1_exp;

    logic        s_lr_toggle;
    logic        s_latch_toggle;

    // -------------------------------------------------------------------------
    // Next-state signals
    // -------------------------------------------------------------------------
    logic        c_next_accum_l_sign;
    logic [MAG_WIDTH-1:0] c_next_accum_l_mag;
    logic [9:0]  c_next_accum_l_exp;

    logic        c_next_accum_r_sign;
    logic [MAG_WIDTH-1:0] c_next_accum_r_mag;
    logic [9:0]  c_next_accum_r_exp;

    logic        c_next_latch0_sign;
    logic [MAG_WIDTH-1:0] c_next_latch0_mag;
    logic [9:0]  c_next_latch0_exp;

    logic        c_next_latch1_sign;
    logic [MAG_WIDTH-1:0] c_next_latch1_mag;
    logic [9:0]  c_next_latch1_exp;

    logic        c_next_lr_toggle;
    logic        c_next_latch_toggle;
    logic        c_next_done;
    logic [15:0] c_next_result;

    // -------------------------------------------------------------------------
    // Temporary combinational signals
    // -------------------------------------------------------------------------
    logic        c_input_sign;
    logic [MAG_WIDTH-1:0] c_input_mag;

    logic        c_lane_l_work_sign;
    logic [MAG_WIDTH-1:0] c_lane_l_work_mag;
    logic [9:0]  c_lane_l_work_exp;

    logic        c_lane_r_work_sign;
    logic [MAG_WIDTH-1:0] c_lane_r_work_mag;
    logic [9:0]  c_lane_r_work_exp;

    logic        c_selected_bank;
    logic        c_selected_latch;

    logic        c_update_prev_sign;
    logic [MAG_WIDTH-1:0] c_update_prev_mag;
    logic [9:0]  c_update_prev_exp;
    logic [9:0]  c_update_common_exp;

    logic        c_update_prev_aligned_sign;
    logic [MAG_WIDTH-1:0] c_update_prev_aligned_mag;
    logic        c_update_input_aligned_sign;
    logic [MAG_WIDTH-1:0] c_update_input_aligned_mag;

    logic        c_update_sum_sign;
    logic [MAG_WIDTH-1:0] c_update_sum_mag;
    logic        c_update_norm_sign;
    logic [MAG_WIDTH-1:0] c_update_norm_mag;
    logic [9:0]  c_update_norm_exp;

    logic [9:0]  c_merge_common_exp;
    logic        c_merge_l_aligned_sign;
    logic [MAG_WIDTH-1:0] c_merge_l_aligned_mag;
    logic        c_merge_r_aligned_sign;
    logic [MAG_WIDTH-1:0] c_merge_r_aligned_mag;

    logic        c_merge_sum_sign;
    logic [MAG_WIDTH-1:0] c_merge_sum_mag;

    logic        c_bias_sign;
    logic [MAG_WIDTH-1:0] c_bias_mag;
    logic [9:0]  c_bias_exp;
    logic [9:0]  c_bias_common_exp;
    logic        c_merge_aligned_sign;
    logic [MAG_WIDTH-1:0] c_merge_aligned_mag;
    logic        c_bias_aligned_sign;
    logic [MAG_WIDTH-1:0] c_bias_aligned_mag;

    logic        c_merge_plus_bias_sign;
    logic [MAG_WIDTH-1:0] c_merge_plus_bias_mag;
    logic [9:0]  c_merge_plus_bias_exp;

    logic        c_merge_norm_sign;
    logic [MAG_WIDTH-1:0] c_merge_norm_mag;
    logic [9:0]  c_merge_norm_exp;

    // -------------------------------------------------------------------------
    // Utility functions/tasks
    // -------------------------------------------------------------------------
    function automatic [5:0] find_msb (
        input logic [MAG_WIDTH-1:0] value
    );
        integer idx;
        begin
            find_msb = 6'd0;
            for (idx = 0; idx < MAG_WIDTH; idx = idx + 1) begin
                if (value[idx]) begin
                    find_msb = idx[5:0];
                end
            end
        end
    endfunction

    function automatic [9:0] max_exp (
        input logic [9:0] a,
        input logic [9:0] b
    );
        begin
            if (a >= b) begin
                max_exp = a;
            end else begin
                max_exp = b;
            end
        end
    endfunction

    task automatic align_value (
        input  logic        in_sign,
        input  logic [MAG_WIDTH-1:0] in_mag,
        input  logic [9:0]  in_exp,
        input  logic [9:0]  target_exp,
        output logic        out_sign,
        output logic [MAG_WIDTH-1:0] out_mag
    );
        logic [9:0] diff;
        begin
            if (in_mag == {MAG_WIDTH{1'b0}}) begin
                out_sign = 1'b0;
                out_mag = {MAG_WIDTH{1'b0}};
            end else begin
                if (target_exp >= in_exp) begin
                    diff = target_exp - in_exp;
                end else begin
                    diff = 10'd0;
                end

                if (diff >= MAG_WIDTH) begin
                    out_mag = {MAG_WIDTH{1'b0}};
                    out_sign = 1'b0;
                end else begin
                    out_mag = in_mag >> diff[5:0];
                    if (out_mag == {MAG_WIDTH{1'b0}}) begin
                        out_sign = 1'b0;
                    end else begin
                        out_sign = in_sign;
                    end
                end
            end
        end
    endtask

    task automatic add_signed_mag (
        input  logic        a_sign,
        input  logic [MAG_WIDTH-1:0] a_mag,
        input  logic        b_sign,
        input  logic [MAG_WIDTH-1:0] b_mag,
        output logic        sum_sign,
        output logic [MAG_WIDTH-1:0] sum_mag
    );
        logic [MAG_WIDTH:0] add_ext;
        begin
            if (a_mag == {MAG_WIDTH{1'b0}} && b_mag == {MAG_WIDTH{1'b0}}) begin
                sum_sign = 1'b0;
                sum_mag = {MAG_WIDTH{1'b0}};
            end else if (a_mag == {MAG_WIDTH{1'b0}}) begin
                sum_sign = b_sign;
                sum_mag = b_mag;
            end else if (b_mag == {MAG_WIDTH{1'b0}}) begin
                sum_sign = a_sign;
                sum_mag = a_mag;
            end else if (a_sign == b_sign) begin
                add_ext = {1'b0, a_mag} + {1'b0, b_mag};
                if (add_ext[MAG_WIDTH]) begin
                    sum_mag = {MAG_WIDTH{1'b1}};
                end else begin
                    sum_mag = add_ext[MAG_WIDTH-1:0];
                end
                if (sum_mag == {MAG_WIDTH{1'b0}}) begin
                    sum_sign = 1'b0;
                end else begin
                    sum_sign = a_sign;
                end
            end else begin
                if (a_mag >= b_mag) begin
                    sum_mag = a_mag - b_mag;
                    if (sum_mag == {MAG_WIDTH{1'b0}}) begin
                        sum_sign = 1'b0;
                    end else begin
                        sum_sign = a_sign;
                    end
                end else begin
                    sum_mag = b_mag - a_mag;
                    if (sum_mag == {MAG_WIDTH{1'b0}}) begin
                        sum_sign = 1'b0;
                    end else begin
                        sum_sign = b_sign;
                    end
                end
            end
        end
    endtask

    task automatic normalize (
        input  logic        in_sign,
        input  logic [MAG_WIDTH-1:0] in_mag,
        input  logic [9:0]  in_exp,
        output logic        out_sign,
        output logic [MAG_WIDTH-1:0] out_mag,
        output logic [9:0]  out_exp
    );
        logic [5:0]  msb_idx;
        logic [MAG_WIDTH-1:0] norm_mag;
        logic [11:0] exp_sum;
        logic [11:0] exp_adj;
        begin
            if (in_mag == {MAG_WIDTH{1'b0}}) begin
                out_sign = 1'b0;
                out_mag = {MAG_WIDTH{1'b0}};
                out_exp = 10'd0;
            end else begin
                msb_idx = find_msb(in_mag);

                if (msb_idx < Q_POINT[5:0]) begin
                    norm_mag = in_mag << (Q_POINT[5:0] - msb_idx);
                end else begin
                    norm_mag = in_mag >> (msb_idx - Q_POINT[5:0]);
                end

                exp_sum = {2'b00, in_exp} + {6'd0, msb_idx};

                if (exp_sum <= Q_POINT[11:0]) begin
                    out_sign = 1'b0;
                    out_mag = {MAG_WIDTH{1'b0}};
                    out_exp = 10'd0;
                end else begin
                    exp_adj = exp_sum - Q_POINT[11:0];
                    if (exp_adj > 12'd1023) begin
                        out_sign = in_sign;
                        out_mag = {MAG_WIDTH{1'b1}};
                        out_exp = 10'h3FF;
                    end else begin
                        out_sign = in_sign;
                        out_mag = norm_mag;
                        out_exp = exp_adj[9:0];
                    end
                end
            end
        end
    endtask

    task automatic make_bf16_rne (
        input  logic        in_sign,
        input  logic [MAG_WIDTH-1:0] in_mag,
        input  logic [9:0]  in_exp,
        output logic [15:0] out_bf16
    );
        logic       l_bit;
        logic       g_bit;
        logic       s_bit;
        logic       round_up;
        logic [7:0] mant_tmp;
        logic [6:0] mant_final;
        logic [10:0] exp_work;
        begin
            if (in_mag == {MAG_WIDTH{1'b0}}) begin
                out_bf16 = 16'd0;
            end else begin
                l_bit = in_mag[Q_POINT-7];
                g_bit = in_mag[Q_POINT-8];
                s_bit = |in_mag[Q_POINT-9:0];

                round_up = g_bit & (s_bit | l_bit);
                mant_tmp = {1'b0, in_mag[Q_POINT-1:Q_POINT-7]} + {7'd0, round_up};

                exp_work = {1'b0, in_exp};
                if (mant_tmp[7]) begin
                    mant_final = 7'd0;
                    exp_work = exp_work + 11'd1;
                end else begin
                    mant_final = mant_tmp[6:0];
                end

                if (exp_work == 11'd0) begin
                    out_bf16 = {in_sign, 15'd0};
                end else if (exp_work >= 11'd255) begin
                    out_bf16 = {in_sign, 8'hFF, 7'd0};
                end else begin
                    out_bf16 = {in_sign, exp_work[7:0], mant_final};
                end
            end
        end
    endtask

    task automatic decode_bf16 (
        input  logic [15:0] in_bf16,
        output logic        out_sign,
        output logic [MAG_WIDTH-1:0] out_mag,
        output logic [9:0]  out_exp
    );
        logic [7:0] exp_field;
        logic [6:0] man_field;
        begin
            out_sign = 1'b0;
            out_mag = {MAG_WIDTH{1'b0}};
            out_exp = 10'd0;

            exp_field = in_bf16[14:7];
            man_field = in_bf16[6:0];

            // Flush denormals to zero for deterministic accumulator behavior.
            if (exp_field == 8'd0) begin
                out_sign = 1'b0;
                out_mag = {MAG_WIDTH{1'b0}};
                out_exp = 10'd0;
            end else if (exp_field == 8'hFF) begin
                // Clamp NaN/Inf input to max-finite before accumulation.
                out_sign = in_bf16[15];
                out_mag = {{(MAG_WIDTH-Q_POINT-1){1'b0}}, 1'b1, 7'h7F, {(Q_POINT-7){1'b0}}};
                out_exp = 10'd254;
            end else begin
                out_sign = in_bf16[15];
                out_mag = {{(MAG_WIDTH-Q_POINT-1){1'b0}}, 1'b1, man_field, {(Q_POINT-7){1'b0}}};
                out_exp = {2'b00, exp_field};
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Combinational next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        c_next_accum_l_sign = s_accum_l_sign;
        c_next_accum_l_mag = s_accum_l_mag;
        c_next_accum_l_exp = s_accum_l_exp;

        c_next_accum_r_sign = s_accum_r_sign;
        c_next_accum_r_mag = s_accum_r_mag;
        c_next_accum_r_exp = s_accum_r_exp;

        c_next_latch0_sign = s_latch0_sign;
        c_next_latch0_mag = s_latch0_mag;
        c_next_latch0_exp = s_latch0_exp;

        c_next_latch1_sign = s_latch1_sign;
        c_next_latch1_mag = s_latch1_mag;
        c_next_latch1_exp = s_latch1_exp;

        c_next_lr_toggle = s_lr_toggle;
        c_next_latch_toggle = s_latch_toggle;

        c_next_done = 1'b0;
        c_next_result = o_result;

        c_input_sign = 1'b0;
        c_input_mag = {MAG_WIDTH{1'b0}};

        c_lane_l_work_sign = s_accum_l_sign;
        c_lane_l_work_mag = s_accum_l_mag;
        c_lane_l_work_exp = s_accum_l_exp;

        c_lane_r_work_sign = s_accum_r_sign;
        c_lane_r_work_mag = s_accum_r_mag;
        c_lane_r_work_exp = s_accum_r_exp;

        c_selected_bank = s_lr_toggle;
        c_selected_latch = s_latch_toggle;

        c_update_prev_sign = 1'b0;
        c_update_prev_mag = {MAG_WIDTH{1'b0}};
        c_update_prev_exp = 10'd0;
        c_update_common_exp = 10'd0;

        c_update_prev_aligned_sign = 1'b0;
        c_update_prev_aligned_mag = {MAG_WIDTH{1'b0}};
        c_update_input_aligned_sign = 1'b0;
        c_update_input_aligned_mag = {MAG_WIDTH{1'b0}};

        c_update_sum_sign = 1'b0;
        c_update_sum_mag = {MAG_WIDTH{1'b0}};

        c_update_norm_sign = 1'b0;
        c_update_norm_mag = {MAG_WIDTH{1'b0}};
        c_update_norm_exp = 10'd0;

        c_merge_common_exp = 10'd0;
        c_merge_l_aligned_sign = 1'b0;
        c_merge_l_aligned_mag = {MAG_WIDTH{1'b0}};
        c_merge_r_aligned_sign = 1'b0;
        c_merge_r_aligned_mag = {MAG_WIDTH{1'b0}};

        c_merge_sum_sign = 1'b0;
        c_merge_sum_mag = {MAG_WIDTH{1'b0}};

        c_bias_sign = 1'b0;
        c_bias_mag = {MAG_WIDTH{1'b0}};
        c_bias_exp = 10'd0;
        c_bias_common_exp = 10'd0;
        c_merge_aligned_sign = 1'b0;
        c_merge_aligned_mag = {MAG_WIDTH{1'b0}};
        c_bias_aligned_sign = 1'b0;
        c_bias_aligned_mag = {MAG_WIDTH{1'b0}};

        c_merge_plus_bias_sign = 1'b0;
        c_merge_plus_bias_mag = {MAG_WIDTH{1'b0}};
        c_merge_plus_bias_exp = 10'd0;

        c_merge_norm_sign = 1'b0;
        c_merge_norm_mag = {MAG_WIDTH{1'b0}};
        c_merge_norm_exp = 10'd0;

        if (i_acc_clear) begin
            c_next_accum_l_sign = 1'b0;
            c_next_accum_l_mag = {MAG_WIDTH{1'b0}};
            c_next_accum_l_exp = 10'd0;

            c_next_accum_r_sign = 1'b0;
            c_next_accum_r_mag = {MAG_WIDTH{1'b0}};
            c_next_accum_r_exp = 10'd0;

            c_next_latch0_sign = 1'b0;
            c_next_latch0_mag = {MAG_WIDTH{1'b0}};
            c_next_latch0_exp = 10'd0;

            c_next_latch1_sign = 1'b0;
            c_next_latch1_mag = {MAG_WIDTH{1'b0}};
            c_next_latch1_exp = 10'd0;

            c_next_lr_toggle = 1'b0;
            c_next_latch_toggle = 1'b0;

            c_next_done = 1'b0;
            c_next_result = 16'd0;
        end else if (i_start) begin
            c_input_sign = i_sign;
            c_input_mag = MAG_WIDTH'(i_ma_sum) << INPUT_SHIFT;

            if (!i_acc_en) begin
                c_lane_l_work_sign = 1'b0;
                c_lane_l_work_mag = {MAG_WIDTH{1'b0}};
                c_lane_l_work_exp = 10'd0;

                c_lane_r_work_sign = 1'b0;
                c_lane_r_work_mag = {MAG_WIDTH{1'b0}};
                c_lane_r_work_exp = 10'd0;
            end

            if (i_latch_seed_valid && i_acc_en) begin
                if (!i_latch_seed) begin
                    c_lane_l_work_sign = s_latch0_sign;
                    c_lane_l_work_mag = s_latch0_mag;
                    c_lane_l_work_exp = s_latch0_exp;
                end else begin
                    c_lane_l_work_sign = s_latch1_sign;
                    c_lane_l_work_mag = s_latch1_mag;
                    c_lane_l_work_exp = s_latch1_exp;
                end
                c_lane_r_work_sign = 1'b0;
                c_lane_r_work_mag = {MAG_WIDTH{1'b0}};
                c_lane_r_work_exp = 10'd0;
            end

            if (i_bank_lr_valid) begin
                c_selected_bank = i_bank_lr;
            end else begin
                c_selected_bank = s_lr_toggle;
            end

            if (!c_selected_bank) begin
                c_update_prev_sign = c_lane_l_work_sign;
                c_update_prev_mag = c_lane_l_work_mag;
                c_update_prev_exp = c_lane_l_work_exp;
            end else begin
                c_update_prev_sign = c_lane_r_work_sign;
                c_update_prev_mag = c_lane_r_work_mag;
                c_update_prev_exp = c_lane_r_work_exp;
            end

            if (c_update_prev_mag == {MAG_WIDTH{1'b0}}) begin
                c_update_common_exp = i_max_exp;
                c_update_prev_aligned_sign = 1'b0;
                c_update_prev_aligned_mag = {MAG_WIDTH{1'b0}};
                c_update_input_aligned_sign = c_input_sign;
                c_update_input_aligned_mag = c_input_mag;
            end else begin
                c_update_common_exp = max_exp(c_update_prev_exp, i_max_exp);
                align_value(
                    c_update_prev_sign,
                    c_update_prev_mag,
                    c_update_prev_exp,
                    c_update_common_exp,
                    c_update_prev_aligned_sign,
                    c_update_prev_aligned_mag
                );
                align_value(
                    c_input_sign,
                    c_input_mag,
                    i_max_exp,
                    c_update_common_exp,
                    c_update_input_aligned_sign,
                    c_update_input_aligned_mag
                );
            end

            add_signed_mag(
                c_update_prev_aligned_sign,
                c_update_prev_aligned_mag,
                c_update_input_aligned_sign,
                c_update_input_aligned_mag,
                c_update_sum_sign,
                c_update_sum_mag
            );

            normalize(
                c_update_sum_sign,
                c_update_sum_mag,
                c_update_common_exp,
                c_update_norm_sign,
                c_update_norm_mag,
                c_update_norm_exp
            );

            if (!c_selected_bank) begin
                c_lane_l_work_sign = c_update_norm_sign;
                c_lane_l_work_mag = c_update_norm_mag;
                c_lane_l_work_exp = c_update_norm_exp;
            end else begin
                c_lane_r_work_sign = c_update_norm_sign;
                c_lane_r_work_mag = c_update_norm_mag;
                c_lane_r_work_exp = c_update_norm_exp;
            end

            if ((c_lane_l_work_mag == {MAG_WIDTH{1'b0}}) && (c_lane_r_work_mag == {MAG_WIDTH{1'b0}})) begin
                c_merge_sum_sign = 1'b0;
                c_merge_sum_mag = {MAG_WIDTH{1'b0}};
                c_merge_common_exp = 10'd0;
            end else begin
                c_merge_common_exp = max_exp(c_lane_l_work_exp, c_lane_r_work_exp);

                align_value(
                    c_lane_l_work_sign,
                    c_lane_l_work_mag,
                    c_lane_l_work_exp,
                    c_merge_common_exp,
                    c_merge_l_aligned_sign,
                    c_merge_l_aligned_mag
                );

                align_value(
                    c_lane_r_work_sign,
                    c_lane_r_work_mag,
                    c_lane_r_work_exp,
                    c_merge_common_exp,
                    c_merge_r_aligned_sign,
                    c_merge_r_aligned_mag
                );

                add_signed_mag(
                    c_merge_l_aligned_sign,
                    c_merge_l_aligned_mag,
                    c_merge_r_aligned_sign,
                    c_merge_r_aligned_mag,
                    c_merge_sum_sign,
                    c_merge_sum_mag
                );
            end

            c_merge_plus_bias_sign = c_merge_sum_sign;
            c_merge_plus_bias_mag = c_merge_sum_mag;
            c_merge_plus_bias_exp = c_merge_common_exp;

            if (i_bias_en && (i_bias != 16'd0)) begin
                decode_bf16(
                    i_bias,
                    c_bias_sign,
                    c_bias_mag,
                    c_bias_exp
                );

                if (c_bias_mag != {MAG_WIDTH{1'b0}}) begin
                    c_bias_common_exp = max_exp(c_merge_common_exp, c_bias_exp);

                    align_value(
                        c_merge_sum_sign,
                        c_merge_sum_mag,
                        c_merge_common_exp,
                        c_bias_common_exp,
                        c_merge_aligned_sign,
                        c_merge_aligned_mag
                    );

                    align_value(
                        c_bias_sign,
                        c_bias_mag,
                        c_bias_exp,
                        c_bias_common_exp,
                        c_bias_aligned_sign,
                        c_bias_aligned_mag
                    );

                    add_signed_mag(
                        c_merge_aligned_sign,
                        c_merge_aligned_mag,
                        c_bias_aligned_sign,
                        c_bias_aligned_mag,
                        c_merge_plus_bias_sign,
                        c_merge_plus_bias_mag
                    );
                    c_merge_plus_bias_exp = c_bias_common_exp;
                end
            end

            normalize(
                c_merge_plus_bias_sign,
                c_merge_plus_bias_mag,
                c_merge_plus_bias_exp,
                c_merge_norm_sign,
                c_merge_norm_mag,
                c_merge_norm_exp
            );

            make_bf16_rne(
                c_merge_norm_sign,
                c_merge_norm_mag,
                c_merge_norm_exp,
                c_next_result
            );

            if (i_bias_en && !i_acc_en) begin
                c_next_accum_l_sign = c_merge_norm_sign;
                c_next_accum_l_mag = c_merge_norm_mag;
                c_next_accum_l_exp = c_merge_norm_exp;

                c_next_accum_r_sign = 1'b0;
                c_next_accum_r_mag = {MAG_WIDTH{1'b0}};
                c_next_accum_r_exp = 10'd0;
            end else begin
                c_next_accum_l_sign = c_lane_l_work_sign;
                c_next_accum_l_mag = c_lane_l_work_mag;
                c_next_accum_l_exp = c_lane_l_work_exp;

                c_next_accum_r_sign = c_lane_r_work_sign;
                c_next_accum_r_mag = c_lane_r_work_mag;
                c_next_accum_r_exp = c_lane_r_work_exp;
            end

            if (i_latch_sel_valid) begin
                c_selected_latch = i_latch_sel;
            end else begin
                c_selected_latch = s_latch_toggle;
            end

            if (!c_selected_latch) begin
                c_next_latch0_sign = c_merge_norm_sign;
                c_next_latch0_mag = c_merge_norm_mag;
                c_next_latch0_exp = c_merge_norm_exp;
            end else begin
                c_next_latch1_sign = c_merge_norm_sign;
                c_next_latch1_mag = c_merge_norm_mag;
                c_next_latch1_exp = c_merge_norm_exp;
            end

            if (!i_bank_lr_valid) begin
                c_next_lr_toggle = ~s_lr_toggle;
            end

            if (!i_latch_sel_valid) begin
                c_next_latch_toggle = ~s_latch_toggle;
            end

            c_next_done = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Sequential state update
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_accum_l_sign <= 1'b0;
            s_accum_l_mag <= {MAG_WIDTH{1'b0}};
            s_accum_l_exp <= 10'd0;

            s_accum_r_sign <= 1'b0;
            s_accum_r_mag <= {MAG_WIDTH{1'b0}};
            s_accum_r_exp <= 10'd0;

            s_latch0_sign <= 1'b0;
            s_latch0_mag <= {MAG_WIDTH{1'b0}};
            s_latch0_exp <= 10'd0;

            s_latch1_sign <= 1'b0;
            s_latch1_mag <= {MAG_WIDTH{1'b0}};
            s_latch1_exp <= 10'd0;

            s_lr_toggle <= 1'b0;
            s_latch_toggle <= 1'b0;

            o_done <= 1'b0;
            o_result <= 16'd0;
        end else begin
            s_accum_l_sign <= c_next_accum_l_sign;
            s_accum_l_mag <= c_next_accum_l_mag;
            s_accum_l_exp <= c_next_accum_l_exp;

            s_accum_r_sign <= c_next_accum_r_sign;
            s_accum_r_mag <= c_next_accum_r_mag;
            s_accum_r_exp <= c_next_accum_r_exp;

            s_latch0_sign <= c_next_latch0_sign;
            s_latch0_mag <= c_next_latch0_mag;
            s_latch0_exp <= c_next_latch0_exp;

            s_latch1_sign <= c_next_latch1_sign;
            s_latch1_mag <= c_next_latch1_mag;
            s_latch1_exp <= c_next_latch1_exp;

            s_lr_toggle <= c_next_lr_toggle;
            s_latch_toggle <= c_next_latch_toggle;

            o_done <= c_next_done;
            o_result <= c_next_result;
        end
    end

    assign o_latch0_sign = s_latch0_sign;
    assign o_latch0_mag = s_latch0_mag;
    assign o_latch1_sign = s_latch1_sign;
    assign o_latch1_mag = s_latch1_mag;

endmodule
