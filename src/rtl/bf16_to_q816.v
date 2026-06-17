`timescale 1ns / 1ps

// =============================================================================
// bf16_to_q816
// - Convert BF16 input to signed fixed-point Q8.16 (24-bit two's complement)
// =============================================================================

module bf16_to_q816 (
    input  logic [15:0] i_bf16,
    output logic [23:0] o_q_tc
);
    localparam logic [23:0] Q816_MAX_TC = 24'h7FFFFF;  // +127.9999847
    localparam logic [23:0] Q816_MIN_TC = 24'h800000;  // -128.0

    logic [7:0] exp_bits;
    logic [6:0] frac_bits;
    logic [7:0] mantissa8;
    integer exp_unbiased;
    integer shift_amt;
    logic [47:0] abs_q48;
    logic [23:0] abs_q24;

    always_comb begin
        exp_bits = i_bf16[14:7];
        frac_bits = i_bf16[6:0];
        mantissa8 = 8'd0;
        exp_unbiased = 0;
        shift_amt = 0;
        abs_q48 = 48'd0;
        abs_q24 = 24'd0;
        o_q_tc = 24'd0;

        if (exp_bits == 8'h00) begin
            // Subnormal is approximated as zero in this path.
            o_q_tc = 24'd0;
        end else if (exp_bits == 8'hFF) begin
            // Treat special encodings as saturated numeric values.
            o_q_tc = i_bf16[15] ? Q816_MIN_TC : Q816_MAX_TC;
        end else begin
            mantissa8 = {1'b1, frac_bits};
            // mantissa8 has binary point after bit[7].
            // Convert to Q8.16 by shifting with (frac_bits - 7) = 9.
            exp_unbiased = {24'd0, exp_bits};
            exp_unbiased = exp_unbiased - 127;
            shift_amt = exp_unbiased + 9;
            abs_q48 = {40'd0, mantissa8};

            if (shift_amt >= 0) begin
                abs_q48 = abs_q48 << shift_amt;
            end else begin
                abs_q48 = abs_q48 >> (-shift_amt);
            end

            abs_q24 = abs_q48[23:0];

            if (!i_bf16[15]) begin
                if (abs_q48[47:24] != 24'd0 || abs_q24 > Q816_MAX_TC) begin
                    o_q_tc = Q816_MAX_TC;
                end else begin
                    o_q_tc = abs_q24;
                end
            end else begin
                if (abs_q48[47:24] != 24'd0 || abs_q24 >= Q816_MIN_TC) begin
                    o_q_tc = Q816_MIN_TC;
                end else begin
                    o_q_tc = (~abs_q24) + 24'd1;
                end
            end
        end
    end
endmodule
