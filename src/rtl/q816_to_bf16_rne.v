`timescale 1ns / 1ps

// =============================================================================
// q816_to_bf16_rne
// - Converts signed Q8.16 (24-bit two's complement) to BF16
// - Round-to-nearest-even (RNE)
// =============================================================================

module q816_to_bf16_rne (
    input  logic [23:0] i_q816_tc,
    output logic [15:0] o_bf16
);

    function automatic [4:0] find_msb24 (
        input logic [23:0] value
    );
        integer i;
        begin
            find_msb24 = 5'd0;
            for (i = 0; i < 24; i = i + 1) begin
                if (value[i]) begin
                    find_msb24 = i[4:0];
                end
            end
        end
    endfunction

    function automatic [23:0] q816_tc_abs (
        input logic [23:0] in_tc
    );
        begin
            if (in_tc[23]) begin
                q816_tc_abs = (~in_tc) + 24'd1;
            end else begin
                q816_tc_abs = in_tc;
            end
        end
    endfunction

    typedef logic [7:0] byte_t;

    logic        in_sign;
    logic [23:0] in_q_abs;
    logic [4:0]  msb_idx;
    integer      msb_idx_i;
    integer      exp_unbiased;
    integer      exp_work;
    integer      shift_amt;
    logic [8:0]  sig9;
    logic [7:0]  sig_byte;
    logic [6:0]  sig_frac;
    logic        guard_bit;
    logic        sticky_bit;
    logic        lsb_bit;
    integer      j;

    always_comb begin
        in_sign = 1'b0;
        in_q_abs = 24'd0;
        msb_idx = 5'd0;
        msb_idx_i = 0;
        exp_unbiased = 0;
        exp_work = 0;
        shift_amt = 0;
        sig9 = 9'd0;
        sig_byte = 8'd0;
        sig_frac = 7'd0;
        guard_bit = 1'b0;
        sticky_bit = 1'b0;
        lsb_bit = 1'b0;
        o_bf16 = 16'd0;

        if (i_q816_tc != 24'd0) begin
            in_sign = i_q816_tc[23];
            in_q_abs = q816_tc_abs(i_q816_tc);
            msb_idx = find_msb24(in_q_abs);
            msb_idx_i = {27'd0, msb_idx};
            exp_unbiased = msb_idx_i - 16;
            exp_work = exp_unbiased + 127;

            if (exp_work <= 0) begin
                o_bf16 = 16'd0;
            end else if (exp_work >= 255) begin
                o_bf16 = {in_sign, 8'hFF, 7'd0};
            end else begin
                guard_bit = 1'b0;
                sticky_bit = 1'b0;
                sig9 = 9'd0;
                sig_byte = 8'd0;

                if (msb_idx >= 5'd7) begin
                    shift_amt = msb_idx_i - 7;
                    sig_byte = byte_t'(in_q_abs >> shift_amt);
                    sig9 = {1'b0, sig_byte};
                    if (shift_amt > 0) begin
                        guard_bit = in_q_abs[shift_amt - 1];
                    end
                    for (j = 0; j < 24; j = j + 1) begin
                        if ((shift_amt > 1) && (j < (shift_amt - 1))) begin
                            sticky_bit = sticky_bit | in_q_abs[j];
                        end
                    end
                end else begin
                    shift_amt = 7 - msb_idx_i;
                    sig_byte = byte_t'(in_q_abs << shift_amt);
                    sig9 = {1'b0, sig_byte};
                end

                lsb_bit = sig9[0];
                if (guard_bit && (sticky_bit || lsb_bit)) begin
                    sig9 = sig9 + 9'd1;
                end

                if (sig9[8]) begin
                    exp_work = exp_work + 1;
                    sig_frac = 7'd0;
                end else begin
                    sig_frac = sig9[6:0];
                end

                if (exp_work >= 255) begin
                    o_bf16 = {in_sign, 8'hFF, 7'd0};
                end else begin
                    o_bf16 = {in_sign, exp_work[7:0], sig_frac};
                end
            end
        end
    end

endmodule
