`timescale 1ns / 1ps

// =============================================================================
// Adder Tree Module - Binary Reduction with Exponent Alignment
// Refactored to follow ECE 5745 Verilog Coding Rules
// =============================================================================

module adder_tree (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        i_start,

    // [Input] From PU Array (16 channels)
    input  logic [15:0] i_sign,           // 1-bit Sign x 16
    input  logic [9:0]  i_exp  [0:15],    // 10-bit Exponent
    input  logic [15:0] i_man  [0:15],    // 16-bit Mantissa

    // [Output] To Accumulator
    output logic        o_done,
    output logic        o_sign,           // Final result sign
    output logic [31:0] o_result_man,     // Final magnitude
    output logic [9:0]  o_common_exp      // Common exponent (10-bit)
);

    // ========================================================================
    // Pipeline Registers
    // ========================================================================

    // Stage 1 Result Registers
    logic        s1_valid;
    logic [9:0]  s1_max_exp;
    logic signed [31:0] s1_aligned_val [0:15];

    // ========================================================================
    // Stage 1: Combinational Logic - Max Exp Search & Alignment
    // ========================================================================
    logic signed [9:0]  c_max_exp;
    logic signed [9:0]  c_exp_diff;
    logic        [31:0] c_shifted_man;
    logic signed [31:0] c_aligned_temp [0:15];

    always_comb begin
        // Default values
        c_max_exp    = i_exp[0];
        c_exp_diff   = 10'sd0;
        c_shifted_man = 32'd0;

        // Initialize array
        for (int j = 0; j < 16; j = j + 1) begin
            c_aligned_temp[j] = 32'sd0;
        end

        // 1-1. Max Exponent Search
        for (int i = 1; i < 16; i = i + 1) begin
            if ($signed(i_exp[i]) > $signed(c_max_exp)) begin
                c_max_exp = i_exp[i];
            end
        end

        // 1-2. Alignment & 2's Complement Conversion
        for (int i = 0; i < 16; i = i + 1) begin
            // Shift Amount (always >= 0)
            c_exp_diff = c_max_exp - $signed(i_exp[i]);

            // Mantissa Shift
            // Upper 8bit: Integer Headroom
            // Middle 16bit: Actual Data
            // Lower 8bit: Fractional Precision
            if (c_exp_diff >= 10'sd32) begin
                c_shifted_man = 32'd0;
            end else begin
                c_shifted_man = {8'b0, i_man[i], 8'b0} >> c_exp_diff[4:0];
            end

            // Apply Sign (Convert to 2's Complement)
            if (i_sign[i]) begin
                c_aligned_temp[i] = -($signed(c_shifted_man));
            end else begin
                c_aligned_temp[i] = $signed(c_shifted_man);
            end
        end
    end

    // ========================================================================
    // Stage 1: Sequential Logic
    // ========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_valid   <= 1'b0;
            s1_max_exp <= 10'd0;
            for (int i = 0; i < 16; i = i + 1) begin
                s1_aligned_val[i] <= 32'sd0;
            end
        end else begin
            if (i_start) begin
                s1_max_exp <= c_max_exp;
                for (int i = 0; i < 16; i = i + 1) begin
                    s1_aligned_val[i] <= c_aligned_temp[i];
                end
                s1_valid <= 1'b1;
            end else begin
                s1_valid <= 1'b0;
            end
        end
    end

    // ========================================================================
    // Stage 2: Combinational Logic - Adder Tree & Output Formatting
    // ========================================================================
    logic signed [31:0] c_tree_lvl1 [0:7];
    logic signed [31:0] c_tree_lvl2 [0:3];
    logic signed [31:0] c_tree_lvl3 [0:1];
    logic signed [31:0] c_tree_final;
    logic               c_final_sign;
    logic        [31:0] c_final_mag;

    always_comb begin
        // Default values
        for (int i = 0; i < 8; i = i + 1) c_tree_lvl1[i] = 32'sd0;
        for (int i = 0; i < 4; i = i + 1) c_tree_lvl2[i] = 32'sd0;
        for (int i = 0; i < 2; i = i + 1) c_tree_lvl3[i] = 32'sd0;
        c_tree_final = 32'sd0;
        c_final_sign = 1'b0;
        c_final_mag  = 32'd0;

        if (s1_valid) begin
            // 2-1. Adder Tree
            for (int i = 0; i < 8; i = i + 1) begin
                c_tree_lvl1[i] = s1_aligned_val[2*i] + s1_aligned_val[2*i+1];
            end
            for (int i = 0; i < 4; i = i + 1) begin
                c_tree_lvl2[i] = c_tree_lvl1[2*i] + c_tree_lvl1[2*i+1];
            end
            for (int i = 0; i < 2; i = i + 1) begin
                c_tree_lvl3[i] = c_tree_lvl2[2*i] + c_tree_lvl2[2*i+1];
            end
            c_tree_final = c_tree_lvl3[0] + c_tree_lvl3[1];

            // 2-2. Final Format Conversion
            c_final_sign = c_tree_final[31]; // MSB is Sign

            if (c_final_sign) begin
                c_final_mag = ~(c_tree_final) + 32'd1; // 2's Comp -> Magnitude
            end else begin
                c_final_mag = c_tree_final;
            end
        end
    end

    // ========================================================================
    // Stage 2: Sequential Logic
    // ========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            o_done       <= 1'b0;
            o_sign       <= 1'b0;
            o_result_man <= 32'd0;
            o_common_exp <= 10'd0;
        end else begin
            if (s1_valid) begin
                o_sign       <= c_final_sign;
                o_result_man <= c_final_mag;
                o_common_exp <= s1_max_exp;
                o_done       <= 1'b1;
            end else begin
                o_done <= 1'b0;
            end
        end
    end

endmodule
