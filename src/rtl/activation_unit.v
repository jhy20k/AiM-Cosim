`timescale 1ns / 1ps

// =============================================================================
// activation_unit
// - BF16 in/out
// - Internal arithmetic in Q8.16 (24-bit two's complement)
// - AF type: 0=ReLU, 1=LeakyReLU
// - Intrinsic fixed latency (no external delay input)
// =============================================================================

module activation_unit (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         i_start,
    input  logic [1:0]   i_af_type,
    input  logic [15:0]  i_af_slope_bf16,
    input  logic [15:0]  i_af_in_bf16,
    output logic         o_done,
    output logic [15:0]  o_af_out_bf16
);

    // ------------------------------------------------------------------------
    // Local constants
    // ------------------------------------------------------------------------
    localparam logic [1:0] AF_LEAKY = 2'd1;
    localparam logic [23:0] Q816_MIN_TC = 24'h800000;
    localparam logic [23:0] Q816_DEFAULT_SLOPE_TC = 24'h00028F;  // ~0.01 in Q8.16
    localparam logic [7:0] AF_FIXED_LATENCY = 8'd4;

    // ------------------------------------------------------------------------
    // Datapath signals
    // ------------------------------------------------------------------------
    logic [23:0] af_in_q816_tc;
    logic [23:0] slope_q816_tc;
    logic [23:0] slope_q816_tc_effective;
    logic [23:0] af_out_q816_tc;

    logic [23:0] in_abs_q816;
    logic [23:0] slope_abs_q816;
    logic [47:0] mul_full_q1632;
    logic [31:0] mul_q32;
    logic [23:0] mul_abs_q816;

    logic [15:0] af_result_comb_bf16;
    logic [15:0] af_result_latched;
    logic [7:0]  af_latency_cnt;
    logic        af_busy;

    // ------------------------------------------------------------------------
    // BF16 -> Q8.16 conversion for input and slope
    // ------------------------------------------------------------------------
    bf16_to_q816 u_bf16_to_q816_input (
        .i_bf16(i_af_in_bf16),
        .o_q_tc(af_in_q816_tc)
    );

    bf16_to_q816 u_bf16_to_q816_slope (
        .i_bf16(i_af_slope_bf16),
        .o_q_tc(slope_q816_tc)
    );

    // Q8.16 -> BF16 (RNE) conversion is separated as a reusable module.
    q816_to_bf16_rne u_q816_to_bf16_rne (
        .i_q816_tc(af_out_q816_tc),
        .o_bf16(af_result_comb_bf16)
    );

    // ------------------------------------------------------------------------
    // Main AF datapath (ReLU / LeakyReLU) - combinational
    // ------------------------------------------------------------------------
    always_comb begin
        af_out_q816_tc = 24'd0;
        slope_q816_tc_effective = slope_q816_tc;

        in_abs_q816 = 24'd0;
        slope_abs_q816 = 24'd0;
        mul_full_q1632 = 48'd0;
        mul_q32 = 32'd0;
        mul_abs_q816 = 24'd0;

        if (slope_q816_tc_effective == 24'd0) begin
            slope_q816_tc_effective = Q816_DEFAULT_SLOPE_TC;
        end

        case (i_af_type)
            AF_LEAKY: begin
                if (af_in_q816_tc[23]) begin
                    // LeakyReLU(x<0) = slope * x (slope treated as non-negative gain).
                    in_abs_q816 = (~af_in_q816_tc) + 24'd1;
                    if (slope_q816_tc_effective[23]) begin
                        slope_abs_q816 = (~slope_q816_tc_effective) + 24'd1;
                    end else begin
                        slope_abs_q816 = slope_q816_tc_effective;
                    end

                    // Q8.16 x Q8.16 => Q16.32 (48-bit intermediate).
                    // Keep upper 32 bits after >>16 to get Q8.16 aligned value.
                    mul_full_q1632 = in_abs_q816 * slope_abs_q816;
                    mul_q32 = mul_full_q1632[47:16];
                    mul_abs_q816 = mul_q32[23:0];

                    if (mul_q32 == 32'd0) begin
                        af_out_q816_tc = 24'd0;
                    end else if (mul_q32[31:24] != 8'd0 || mul_abs_q816 >= Q816_MIN_TC) begin
                        af_out_q816_tc = Q816_MIN_TC;
                    end else begin
                        af_out_q816_tc = (~mul_abs_q816) + 24'd1;
                    end
                end else begin
                    af_out_q816_tc = af_in_q816_tc;
                end
            end

            default: begin
                // ReLU default
                if (af_in_q816_tc[23]) begin
                    af_out_q816_tc = 24'd0;
                end else begin
                    af_out_q816_tc = af_in_q816_tc;
                end
            end
        endcase

    end

    // ------------------------------------------------------------------------
    // Fixed-latency AF completion
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            af_busy <= 1'b0;
            af_latency_cnt <= 8'd0;
            af_result_latched <= 16'd0;
            o_done <= 1'b0;
            o_af_out_bf16 <= 16'd0;
        end else begin
            o_done <= 1'b0;

            if (i_start && !af_busy) begin
                af_busy <= 1'b1;
                af_latency_cnt <= AF_FIXED_LATENCY;
                af_result_latched <= af_result_comb_bf16;
            end else if (af_busy) begin
                if (af_latency_cnt <= 8'd1) begin
                    af_busy <= 1'b0;
                    o_done <= 1'b1;
                    o_af_out_bf16 <= af_result_latched;
                end else begin
                    af_latency_cnt <= af_latency_cnt - 8'd1;
                end
            end
        end
    end

endmodule
