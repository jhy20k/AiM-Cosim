`timescale 1ns / 1ps

// =============================================================================
// Processing Unit (PU) Module - BF16 Multiplier
// Refactored to follow ECE 5745 Verilog Coding Rules
// =============================================================================

module pu (
    input  logic [15:0] i_a,    // BF16 Operand A
    input  logic [15:0] i_b,    // BF16 Operand B

    // Output: Extended Format for Adder Tree
    output logic        o_sign,
    output logic [9:0]  o_exp,      // 10-bit Extended Exponent
    output logic [15:0] o_mantissa
);

    // ========================================================================
    // 1. Unpacking & Special Case Detection (Combinational)
    // ========================================================================
    logic        sign_a;
    logic [7:0]  raw_exp_a;
    logic [6:0]  man_a;

    logic        sign_b;
    logic [7:0]  raw_exp_b;
    logic [6:0]  man_b;

    assign sign_a    = i_a[15];
    assign raw_exp_a = i_a[14:7];
    assign man_a     = i_a[6:0];

    assign sign_b    = i_b[15];
    assign raw_exp_b = i_b[14:7];
    assign man_b     = i_b[6:0];

    // Check for Zero / Sub-normal
    logic exp_a_is_zero;
    logic exp_b_is_zero;

    assign exp_a_is_zero = (raw_exp_a == 8'd0);
    assign exp_b_is_zero = (raw_exp_b == 8'd0);

    // Hidden Bit Logic
    logic hidden_a;
    logic hidden_b;

    assign hidden_a = !exp_a_is_zero;
    assign hidden_b = !exp_b_is_zero;

    // Exponent Adjustment
    logic [7:0] real_exp_a;
    logic [7:0] real_exp_b;

    assign real_exp_a = exp_a_is_zero ? 8'd1 : raw_exp_a;
    assign real_exp_b = exp_b_is_zero ? 8'd1 : raw_exp_b;

    // Special Value Flags
    logic a_is_inf;
    logic b_is_inf;
    logic a_is_nan;
    logic b_is_nan;
    logic a_is_pure_zero;
    logic b_is_pure_zero;

    assign a_is_inf       = (raw_exp_a == 8'hFF) && (man_a == 7'd0);
    assign b_is_inf       = (raw_exp_b == 8'hFF) && (man_b == 7'd0);
    assign a_is_nan       = (raw_exp_a == 8'hFF) && (man_a != 7'd0);
    assign b_is_nan       = (raw_exp_b == 8'hFF) && (man_b != 7'd0);
    assign a_is_pure_zero = exp_a_is_zero && (man_a == 7'd0);
    assign b_is_pure_zero = exp_b_is_zero && (man_b == 7'd0);

    // Result Exception Logic
    logic res_is_nan;
    logic res_is_inf;
    logic res_is_zero;

    assign res_is_nan  = a_is_nan | b_is_nan | (a_is_inf & b_is_pure_zero) | (a_is_pure_zero & b_is_inf);
    assign res_is_inf  = !res_is_nan & (a_is_inf | b_is_inf);
    assign res_is_zero = !res_is_nan & !res_is_inf & (a_is_pure_zero | b_is_pure_zero);

    // ========================================================================
    // 2. Sign Calculation (XOR)
    // ========================================================================
    assign o_sign = res_is_nan ? 1'b0 : (sign_a ^ sign_b);

    // ========================================================================
    // 3. Mantissa Multiplication & Normalization
    // ========================================================================
    logic [7:0]  full_man_a;
    logic [7:0]  full_man_b;
    logic [15:0] raw_mantissa_mul;
    logic        mantissa_overflow;
    logic [15:0] normalized_man;

    assign full_man_a = {hidden_a, man_a};
    assign full_man_b = {hidden_b, man_b};

    // 16-bit Product (Q14 Format if normalized: 1.0 is bit 14)
    assign raw_mantissa_mul = full_man_a * full_man_b;

    // Overflow(>=2.0): Shift mantissa >> 1 to maintain normalization
    assign mantissa_overflow = raw_mantissa_mul[15];
    assign normalized_man    = mantissa_overflow ? (raw_mantissa_mul >> 1) : raw_mantissa_mul;

    // Exception Output Handling
    assign o_mantissa = res_is_nan  ? 16'hFFFF :
                        res_is_inf  ? 16'd0 :
                        res_is_zero ? 16'd0 :
                        normalized_man;

    // ========================================================================
    // 4. Exponent Calculation (10-bit Extended)
    // ========================================================================
    logic signed [9:0] exp_a_ext;
    logic signed [9:0] exp_b_ext;
    logic signed [9:0] bias;
    logic signed [9:0] ovf_ext;
    logic signed [9:0] exp_calc;
    logic        [9:0] final_exp;

    assign exp_a_ext = {2'b00, real_exp_a};
    assign exp_b_ext = {2'b00, real_exp_b};
    assign bias      = 10'sd127;
    assign ovf_ext   = {9'b0, mantissa_overflow};
    assign exp_calc  = exp_a_ext + exp_b_ext - bias + ovf_ext;

    always_comb begin
        if (res_is_nan || res_is_inf) begin
            final_exp = 10'd255;
        end
        else if (res_is_zero) begin
            final_exp = 10'd0;
        end
        else begin
            final_exp = exp_calc;
        end
    end

    assign o_exp = final_exp;

endmodule
