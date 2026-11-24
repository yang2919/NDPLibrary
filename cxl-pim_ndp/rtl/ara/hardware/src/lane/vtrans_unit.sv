// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0
// Author: Jae Hyung Ko

module vtrans_unit
  import ara_pkg::*;
  import rvv_pkg::*;
#(
    parameter int unsigned IdxWidth = 3
)(
    input  logic                clk_i,
    input  logic                rst_ni,
    input  elen_t               operand_i,
    input  vew_e                sew_i,
    input  logic                is_exp_i,
    input  logic                vld_i,
    input  logic [IdxWidth-1:0] idx_i,
    output elen_t               result_o,
    output logic                vld_o,
    output logic [IdxWidth-1:0] idx_o
);

  if (LatFNonComp != 3)
    $error("vtrans_unit is a fixed 3-stage pipeline; set LatFNonComp = 3");

  logic [3:0][31:0] slot_in;
  always_comb begin
    slot_in = '0;
    if (sew_i == EW32) begin
      slot_in[0] = operand_i[31:0];
      slot_in[1] = operand_i[63:32];
    end else begin
      for (int h = 0; h < 4; h++)
        slot_in[h] = vtrans_pkg::vtrans_h2s(operand_i[h*16 +: 16]);
    end
  end

  logic [2:0] is_exp_q;
  vew_e [2:0] sew_q;
  logic [2:0] vld_q;
  logic [2:0][IdxWidth-1:0] idx_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      is_exp_q <= '0;
      sew_q    <= {3{EW8}};
      vld_q    <= '0;
      idx_q    <= '0;
    end else begin
      is_exp_q <= {is_exp_q[1:0], is_exp_i};
      sew_q    <= {sew_q[1:0], sew_i};
      vld_q    <= {vld_q[1:0], vld_i};
      idx_q    <= {idx_q[1:0], idx_i};
    end
  end
  assign vld_o = vld_q[2];
  assign idx_o = idx_q[2];

  vtrans_pkg::vt_exp_s0_t [3:0] exp_s0_q;
  vtrans_pkg::vt_exp_s1_t [3:0] exp_s1_q;
  vtrans_pkg::vt_exp_s2_t [3:0] exp_s2_q;
  vtrans_pkg::vt_rcp_s0_t [3:0] rcp_s0_q, rcp_s1_q;
  vtrans_pkg::vt_rcp_s2_t [3:0] rcp_s2_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      exp_s0_q <= '0; exp_s1_q <= '0; exp_s2_q <= '0;
      rcp_s0_q <= '0; rcp_s1_q <= '0; rcp_s2_q <= '0;
    end else begin
      for (int j = 0; j < 4; j++) begin
        exp_s0_q[j] <= vtrans_pkg::vtrans_exp_s0(slot_in[j]);
        exp_s1_q[j] <= vtrans_pkg::vtrans_exp_s1(exp_s0_q[j]);
        exp_s2_q[j] <= vtrans_pkg::vtrans_exp_s2(exp_s1_q[j]);
        rcp_s0_q[j] <= vtrans_pkg::vtrans_rcp_s0(slot_in[j]);
        rcp_s1_q[j] <= rcp_s0_q[j];
        rcp_s2_q[j] <= vtrans_pkg::vtrans_rcp_s2(rcp_s1_q[j]);
      end
    end
  end

  logic [3:0][31:0] slot_out;
  always_comb begin
    for (int j = 0; j < 4; j++)
      slot_out[j] = is_exp_q[2] ? vtrans_pkg::vtrans_exp_s3(exp_s2_q[j])
                                : vtrans_pkg::vtrans_rcp_s3(rcp_s2_q[j]);
    if (sew_q[2] == EW32)
      result_o = {slot_out[1], slot_out[0]};
    else begin
      for (int h = 0; h < 4; h++)
        result_o[h*16 +: 16] = vtrans_pkg::vtrans_s2h(slot_out[h]);
    end
  end

endmodule
