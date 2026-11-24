// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Modified 2026 by KETI for the VHK158 support.
// Author: Jae Hyung Ko
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Description:
// Ara's SIMD ALU, operating on elements 64-bit wide, and generating 64 bits per cycle.

module simd_alu import ara_pkg::*; import rvv_pkg::*; #(
    // Support for fixed-point data types
    parameter  fixpt_support_e FixPtSupport = FixedPointEnable,

    parameter  int    unsigned NumPipeRegs  = 0,
    // Dependant parameters. DO NOT CHANGE!
    localparam int    unsigned DataWidth    = $bits(elen_t),
    localparam int    unsigned StrbWidth    = DataWidth/8,
    localparam type            strb_t       = logic [StrbWidth-1:0]
  ) (
    input  logic       clk_i,
    input  logic       rst_ni,
    input  elen_t      operand_a_i,
    input  elen_t      operand_b_i,
    input  logic       valid_i,
    input  logic       vm_i,
    input  strb_t      mask_i,
    input  logic       narrowing_select_i,
    input  ara_op_e    op_i,
    input  vew_e       vew_i,
    output vxsat_t     vxsat_o,
    input  strb_t      rm,
    input  vxrm_t      vxrm_i,
    output elen_t      result_o,

    output logic       ready_o,
    input  logic       ready_i,
    output logic       valid_o
  );

`include "common_cells/registers.svh"

  ///////////////////
  //  Definitions  //
  ///////////////////

  typedef union packed {
    logic [0:0][63:0] w64;
    logic [1:0][31:0] w32;
    logic [3:0][15:0] w16;
    logic [7:0][ 7:0] w8;
  } alu_operand_t;

  elen_t     operand_a_q;
  elen_t     operand_b_q;
  logic      valid_q;
  logic      vm_q;
  strb_t     mask_q;
  logic      narrowing_select_q;
  ara_op_e   op_q;
  vew_e      vew_q;
  strb_t     rm_q;
  vxrm_t     vxrm_q;

  elen_t     [NumPipeRegs:0] opa_pipe_d, opb_pipe_d;
  logic      [NumPipeRegs:0] valid_pipe_d, vm_pipe_d, narsel_pipe_d;
  strb_t     [NumPipeRegs:0] mask_pipe_d, rm_pipe_d;
  ara_op_e   [NumPipeRegs:0] op_pipe_d;
  vew_e      [NumPipeRegs:0] vew_pipe_d;
  vxrm_t     [NumPipeRegs:0] vxrm_pipe_d;

  logic      [NumPipeRegs:0] stage_ready;

  assign opa_pipe_d[0]    = operand_a_i;
  assign opb_pipe_d[0]    = operand_b_i;
  assign valid_pipe_d[0]  = valid_i;
  assign vm_pipe_d[0]     = vm_i;
  assign mask_pipe_d[0]   = mask_i;
  assign narsel_pipe_d[0] = narrowing_select_i;
  assign op_pipe_d[0]     = op_i;
  assign vew_pipe_d[0]    = vew_i;
  assign rm_pipe_d[0]     = rm;
  assign vxrm_pipe_d[0]   = vxrm_i;

  if (NumPipeRegs > 0) begin : gen_pipeline
    elen_t     [NumPipeRegs-1:0] opa_pipe_q, opb_pipe_q;
    logic      [NumPipeRegs-1:0] valid_pipe_q, vm_pipe_q, narsel_pipe_q;
    strb_t     [NumPipeRegs-1:0] mask_pipe_q, rm_pipe_q;
    ara_op_e   [NumPipeRegs-1:0] op_pipe_q;
    vew_e      [NumPipeRegs-1:0] vew_pipe_q;
    vxrm_t     [NumPipeRegs-1:0] vxrm_pipe_q;

    for (genvar i = 0; i < NumPipeRegs; i++) begin : gen_pipeline_stages

      assign opa_pipe_d[i+1]    = opa_pipe_q[i];
      assign opb_pipe_d[i+1]    = opb_pipe_q[i];
      assign valid_pipe_d[i+1]  = valid_pipe_q[i];
      assign vm_pipe_d[i+1]     = vm_pipe_q[i];
      assign mask_pipe_d[i+1]   = mask_pipe_q[i];
      assign narsel_pipe_d[i+1] = narsel_pipe_q[i];
      assign op_pipe_d[i+1]     = op_pipe_q[i];
      assign vew_pipe_d[i+1]    = vew_pipe_q[i];
      assign rm_pipe_d[i+1]     = rm_pipe_q[i];
      assign vxrm_pipe_d[i+1]   = vxrm_pipe_q[i];

      assign stage_ready[i] = stage_ready[i+1] | ~valid_pipe_q[i];

      logic reg_ena;
      assign reg_ena = stage_ready[i];

      `FFL(valid_pipe_q[i],  valid_pipe_d[i],  reg_ena, '0)
      `FFL(opa_pipe_q[i],    opa_pipe_d[i],    reg_ena, '0)
      `FFL(opb_pipe_q[i],    opb_pipe_d[i],    reg_ena, '0)
      `FFL(vm_pipe_q[i],     vm_pipe_d[i],     reg_ena, '0)
      `FFL(mask_pipe_q[i],   mask_pipe_d[i],   reg_ena, '0)
      `FFL(narsel_pipe_q[i], narsel_pipe_d[i], reg_ena, '0)
      `FFL(op_pipe_q[i],     op_pipe_d[i],     reg_ena, ara_op_e'('0))
      `FFL(vew_pipe_q[i],    vew_pipe_d[i],    reg_ena, vew_e'('0))
      `FFL(rm_pipe_q[i],     rm_pipe_d[i],     reg_ena, '0)
      `FFL(vxrm_pipe_q[i],   vxrm_pipe_d[i],   reg_ena, vxrm_t'('0))
    end
  end

  assign ready_o = stage_ready[0];

  assign stage_ready[NumPipeRegs] = ready_i;

  assign operand_a_q        = opa_pipe_d[NumPipeRegs];
  assign operand_b_q        = opb_pipe_d[NumPipeRegs];
  assign valid_q            = valid_pipe_d[NumPipeRegs];
  assign vm_q               = vm_pipe_d[NumPipeRegs];
  assign mask_q             = mask_pipe_d[NumPipeRegs];
  assign narrowing_select_q = narsel_pipe_d[NumPipeRegs];
  assign op_q               = op_pipe_d[NumPipeRegs];
  assign vew_q              = vew_pipe_d[NumPipeRegs];
  assign rm_q               = rm_pipe_d[NumPipeRegs];
  assign vxrm_q             = vxrm_pipe_d[NumPipeRegs];

  assign valid_o = valid_q;

  alu_operand_t opa, opb, res;
  assign opa      = operand_a_q;
  assign opb      = operand_b_q;
  assign result_o = res;

  typedef union packed {
    logic [0:0][71:0] w64;
    logic [1:0][35:0] w32;
    logic [3:0][17:0] w16;
    logic [7:0][ 8:0] w8;
  } alu_sat_operand_t;

  alu_sat_operand_t sat_sum, sat_sub;
  vxsat_t     vxsat;
  vxrm_t      vxrm;
  logic       r;

  assign vxrm = vxrm_q;
  assign vxsat_o = vxsat;
  ///////////////////
  //  Comparisons  //
  ///////////////////

  // Comparison instructions that use signed operands
  logic is_signed;
  assign is_signed = op_q inside {VMAX, VREDMAX, VMIN, VREDMIN, VMSLT, VMSLE, VMSGT};
  // Compare operands.

  logic [7:0] less;
  logic [7:0] equal;

  always_comb begin: p_comparison
    // Default assignment
    less  = '0;
    equal = '0;

    if (valid_q) begin
      unique case (vew_q)
        EW8 : for (int b = 0; b < 8; b++) less[1*b] =
            $signed({is_signed & opb.w8 [b][ 7], opb.w8 [b]}) <
            $signed({is_signed & opa.w8 [b][ 7], opa.w8 [b]});
        EW16: for (int b = 0; b < 4; b++) less[2*b] =
            $signed({is_signed & opb.w16[b][15], opb.w16[b]}) <
            $signed({is_signed & opa.w16[b][15], opa.w16[b]});
        EW32: for (int b = 0; b < 2; b++) less[4*b] =
            $signed({is_signed & opb.w32[b][31], opb.w32[b]}) <
            $signed({is_signed & opa.w32[b][31], opa.w32[b]});
        EW64: for (int b = 0; b < 1; b++) less[8*b] =
            $signed({is_signed & opb.w64[b][63], opb.w64[b]}) <
            $signed({is_signed & opa.w64[b][63], opa.w64[b]});
        default:;
      endcase

      unique case (vew_q)
        EW8 : for (int b = 0; b < 8; b++) equal[1*b] = opa.w8 [b] == opb.w8 [b];
        EW16: for (int b = 0; b < 4; b++) equal[2*b] = opa.w16[b] == opb.w16[b];
        EW32: for (int b = 0; b < 2; b++) equal[4*b] = opa.w32[b] == opb.w32[b];
        EW64: for (int b = 0; b < 1; b++) equal[8*b] = opa.w64[b] == opb.w64[b];
        default:;
      endcase
    end
  end : p_comparison

  ///////////
  //  ALU  //
  ///////////

  always_comb begin: p_alu
    // Default assignment
    res       = '0;
    vxsat.w64 = '0;

    if (valid_q)
      unique case (op_q)
        // Logical operations
        VAND, VREDAND: res = operand_a_q & operand_b_q;
        VOR, VREDOR  : res = operand_a_q | operand_b_q;
        VXOR, VREDXOR: res = operand_a_q ^ operand_b_q;

        // Mask logical operations
        VMAND   : res = operand_a_q & operand_b_q;
        VMANDNOT: res = ~operand_a_q & operand_b_q;
        VMNAND  : res = ~(operand_a_q & operand_b_q);
        VMOR    : res = operand_a_q | operand_b_q;
        VMNOR   : res = ~(operand_a_q | operand_b_q);
        VMORNOT : res = ~operand_a_q | operand_b_q;
        VMXOR   : res = operand_a_q ^ operand_b_q;
        VMXNOR  : res = ~(operand_a_q ^ operand_b_q);

        // Mask operands pass-through
        VCPOP, VFIRST, VMSBF, VMSOF, VMSIF, VIOTA: res = operand_b_q;

        // Mask operands pass-through
        VRGATHER, VRGATHEREI16, VCOMPRESS: res = operand_a_q;

        // Arithmetic instructions
        VSADDU: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [8:0] sum = opa.w8[b] + opb.w8[b];
                vxsat.w8[b]   = sum[8];
                res.w8[b]     = vxsat.w8[b] ? {8{1'b1}} : sum[7:0];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sum = opa.w16[b] + opb.w16[b];
                vxsat.w16[b]   = {2{sum[16]}};
                res.w16[b]     = &vxsat.w16[b] ? {16{1'b1}} : sum[15:0];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sum = opa.w32[b] + opb.w32[b];
                vxsat.w32[b]   = {4{sum[32]}};
                res.w32[b]     = &vxsat.w32[b] ? {32{1'b1}} : sum[31:0];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sum = opa.w64[b] + opb.w64[b];
                vxsat.w64[b]   = {8{sum[64]}};
                res.w64[b]     = &vxsat.w64[b] ? {64{1'b1}} : sum[63:0];
              end
          endcase
        VSADD: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [8:0] sum = opa.w8[b] + opb.w8[b];
                vxsat.w8[b]   = (sum[7]^opa.w8[b][7]) & ~(opa.w8[b][7] ^ opb.w8[b][7]);
                res.w8[b]     = vxsat.w8[b] ? (sum[7] ? {1'b0, {7{1'b1}}} : {1'b1, {7{1'b0}}} ) : sum[7:0];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sum = opa.w16[b] + opb.w16[b];
                vxsat.w16[b]   = {2{(sum[15] ^ opa.w16[b][15]) & ~(opa.w16[b][15] ^ opb.w16[b][15])}};
                res.w16[b]     = &vxsat.w16[b] ? (sum[15] ? {1'b0, {15{1'b1}}} : {1'b1, {15{1'b0}}} ) : sum[15:0];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sum = opa.w32[b] + opb.w32[b];
                vxsat.w32[b]   = {4{(sum[31] ^ opa.w32[b][31]) && !(opa.w32[b][31] ^ opb.w32[b][31])}};
                res.w32[b]     = &vxsat.w32[b] ? (sum[31] ? {1'b0, {31{1'b1}}} : {1'b1, {31{1'b0}}} ) : sum[31:0];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sum = opa.w64[b] + opb.w64[b];
                vxsat.w64[b]   = {8{(sum[63] ^ opa.w64[b][63]) & ~(opa.w64[b][63] ^ opb.w64[b][63])}};
                res.w64[b]     = &vxsat.w64[b] ? (sum[63] ? {1'b0, {63{1'b1}}} : {1'b1, {63{1'b0}}} ) : sum[63:0];
              end
          endcase
        VAADD, VAADDU: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
              automatic logic [ 8:0] sum = opa.w8 [b] + opb.w8 [b];
                unique case (vxrm)
                  2'b00: r = sum[0];
                  2'b01: r = &sum[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sum[1] & (sum[0]!=0);
                endcase
                res.w8[b] = (op_q == VAADDU) ? sum[8:1] + r : {sum[7], sum[7:1]} + r;
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sum = opa.w16[b] + opb.w16[b];
                unique case (vxrm)
                  2'b00: r = sum[0];
                  2'b01: r = &sum[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sum[1] & (sum[0]!=0);
                endcase
                res.w16[b] = (op_q == VAADDU) ? sum[16:1] + r : {sum[15], sum[15:1]} + r;
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sum = opa.w32[b] + opb.w32[b];
                unique case (vxrm)
                  2'b00: r = sum[0];
                  2'b01: r = &sum[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sum[1] & (sum[0]!=0);
                endcase
                res.w32[b] = (op_q == VAADDU) ? sum[32:1] + r : {sum[31], sum[31:1]} + r;
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sum = opa.w64[b] + opb.w64[b];
                unique case (vxrm)
                  2'b00: r = sum[0];
                  2'b01: r = &sum[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sum[1] & (sum[0]!=0);
                endcase
                res.w64[b] = (op_q == VAADDU) ? sum[64:1] + r : {sum[63], sum[63:1]} + r;
              end
          endcase
        VADD, VADC, VMADC, VREDSUM, VWREDSUMU, VWREDSUM: unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [ 8:0] sum = opa.w8 [b] + opb.w8 [b] +
                logic'(op_q inside {VADC, VMADC} && mask_q[1*b] & ~vm_q);
                res.w8[b] = (op_q == VMADC) ? {6'b0, 1'b1, sum[8]} : sum[7:0];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sum = opa.w16[b] + opb.w16[b] +
                logic'(op_q inside {VADC, VMADC} && mask_q[2*b] & ~vm_q);
                res.w16[b] = (op_q == VMADC) ? {14'b0, 1'b1, sum[16]} : sum[15:0];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sum = opa.w32[b] + opb.w32[b] +
                logic'(op_q inside {VADC, VMADC} && mask_q[4*b] & ~vm_q);
                res.w32[b] = (op_q == VMADC) ? {30'b0, 1'b1, sum[32]} : sum[31:0];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sum = opa.w64[b] + opb.w64[b] +
                logic'(op_q inside {VADC, VMADC} && mask_q[8*b] & ~vm_q);
                res.w64[b] = (op_q == VMADC) ? {62'b0, 1'b1, sum[64]} : sum[63:0];
              end
          endcase
        VSUB, VSBC, VMSBC: unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [ 8:0] sub = opb.w8 [b] - opa.w8 [b] -
                logic'(op_q inside {VSBC, VMSBC} && mask_q[1*b] & ~vm_q);
                res.w8[b] = (op_q == VMSBC) ? {6'b0, 1'b1, sub[8]} : sub[7:0];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sub = opb.w16[b] - opa.w16[b] -
                logic'(op_q inside {VSBC, VMSBC} && mask_q[2*b] & ~vm_q);
                res.w16[b] = (op_q == VMSBC) ? {14'b0, 1'b1, sub[16]} : sub[15:0];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sub = opb.w32[b] - opa.w32[b] -
                logic'(op_q inside {VSBC, VMSBC} && mask_q[4*b] & ~vm_q);
                res.w32[b] = (op_q == VMSBC) ? {30'b0, 1'b1, sub[32]} : sub[31:0];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sub = opb.w64[b] - opa.w64[b] -
                logic'(op_q inside {VSBC, VMSBC} && mask_q[8*b] & ~vm_q);
                res.w64[b] = (op_q == VMSBC) ? {62'b0, 1'b1, sub[64]} : sub[63:0];
              end
          endcase
        VRSUB: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] = opa.w8 [b] - opb.w8 [b];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] = opa.w16[b] - opb.w16[b];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] = opa.w32[b] - opb.w32[b];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] = opa.w64[b] - opb.w64[b];
          endcase
        VSSUBU: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [8:0] sub = opb.w8 [b] - opa.w8 [b];
                vxsat.w8[b]   = sub[8];
                res.w8[b]     = vxsat.w8[b] ? {8{1'b0}} : sub[7:0];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] sub = opb.w16[b] - opa.w16[b];
                vxsat.w16[b]   = {2{sub[16]}};
                res.w16[b]     = &vxsat.w16[b] ? {16{1'b0}} : sub[15:0];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] sub = opb.w32[b] - opa.w32[b];
                vxsat.w32[b]   = {4{sub[32]}};
                res.w32[b]     = &vxsat.w32[b] ? {32{1'b0}} : sub[31:0];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] sub = opb.w64[b] - opa.w64[b];
                vxsat.w64[b]   = {8{sub[64]}};
                res.w64[b]     = &vxsat.w64[b] ? {64{1'b0}} : sub[63:0];
              end
          endcase
        VSSUB: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
          EW8: for (int b = 0; b < 8; b++) begin
              automatic logic [8:0] sub = opb.w8[b] - opa.w8[b];
              vxsat.w8[b]   = (!opb.w8[b][7] & opa.w8[b][7] & sub[7]) |
                              (opb.w8[b][7] & !opa.w8[b][7] & !sub[7]);
              res.w8[b]     = vxsat.w8[b] ? (opb.w8[b][7] ? 8'h80 : 8'h7F) : sub[7:0];
          end
          EW16: for (int b = 0; b < 4; b++) begin
              automatic logic [16:0] sub = opb.w16[b] - opa.w16[b];
              vxsat.w16[b]   = (!opb.w16[b][15] & opa.w16[b][15] & sub[15]) |
                               (opb.w16[b][15] & !opa.w16[b][15] & !sub[15]);
              res.w16[b]     = vxsat.w16[b] ? (opb.w16[b][15] ? 16'h8000 : 16'h7FFF) : sub[15:0];
          end
          EW32: for (int b = 0; b < 2; b++) begin
              automatic logic [32:0] sub = opb.w32[b] - opa.w32[b];
              vxsat.w32[b]   = (!opb.w32[b][31] & opa.w32[b][31] & sub[31]) |
                               (opb.w32[b][31] & !opa.w32[b][31] & !sub[31]);
              res.w32[b]     = vxsat.w32[b] ? (opb.w32[b][31] ? 32'h80000000 : 32'h7FFFFFFF) : sub[31:0];
          end
          EW64: for (int b = 0; b < 1; b++) begin
              automatic logic [64:0] sub = opb.w64[b] - opa.w64[b];
              vxsat.w64[b]   = (!opb.w64[b][63] & opa.w64[b][63] & sub[63]) |
                               (opb.w64[b][63] & !opa.w64[b][63] & !sub[63]);
              res.w64[b]     = vxsat.w64[b] ? (opb.w64[b][63] ? 64'h8000000000000000 : 64'h7FFFFFFFFFFFFFFF) : sub[63:0];
          end
          endcase
        VASUB, VASUBU: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [ 8:0] sub = opb.w8 [b] - opa.w8 [b];
                unique case (vxrm)
                  2'b00: r = sub[0];
                  2'b01: r = &sub[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sub[1] & (sub[0]!=0);
                endcase
                res.w8[b] = (op_q == VASUBU) ? (sub[7:0] >> 1) + r : ($signed(sub[7:0]) >>> 1) + r;
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [ 16:0] sub = opb.w16[b] - opa.w16[b];
                unique case (vxrm)
                  2'b00: r = sub[0];
                  2'b01: r = &sub[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sub[1] & (sub[0]!=0);
                endcase
                res.w16[b] = (op_q == VASUBU) ? (sub[15:0] >> 1) + r : ($signed(sub[15:0]) >>> 1) + r;
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [ 32:0] sub = opb.w32[b] - opa.w32[b];
                unique case (vxrm)
                  2'b00: r = sub[0];
                  2'b01: r = &sub[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sub[1] & (sub[0]!=0);
                endcase
                res.w32[b] = (op_q == VASUBU) ? (sub[31:0] >> 1) + r : ($signed(sub[31:0]) >>> 1) + r;
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [ 64:0] sub = opb.w64[b] - opa.w64[b];
                unique case (vxrm)
                  2'b00: r = sub[0];
                  2'b01: r = &sub[1:0];
                  2'b10: r = 1'b0;
                  2'b11: r = !sub[1] & (sub[0]!=0);
                endcase
                res.w64[b] = (op_q == VASUBU) ? (sub[63:0] >> 1) + r : ($signed(sub[63:0]) >>> 1) + r;
              end
          endcase

        // Shift instructions
        VSLL: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] = opb.w8 [b] << opa.w8 [b][2:0];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] = opb.w16[b] << opa.w16[b][3:0];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] = opb.w32[b] << opa.w32[b][4:0];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] = opb.w64[b] << opa.w64[b][5:0];
          endcase
        VSRL: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] = opb.w8 [b] >> opa.w8 [b][2:0];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] = opb.w16[b] >> opa.w16[b][3:0];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] = opb.w32[b] >> opa.w32[b][4:0];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] = opb.w64[b] >> opa.w64[b][5:0];
          endcase
        VSRA: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] = $signed(opb.w8 [b]) >>> opa.w8 [b][2:0];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] = $signed(opb.w16[b]) >>> opa.w16[b][3:0];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] = $signed(opb.w32[b]) >>> opa.w32[b][4:0];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] = $signed(opb.w64[b]) >>> opa.w64[b][5:0];
          endcase
        VNSRL: unique case (vew_q)
            EW8 : for (int b = 0; b < 4; b++) res.w8 [2*b + narrowing_select_q] = opb.w16[b] >>
                opa.w16[b][3:0];
            EW16: for (int b = 0; b < 2; b++) res.w16[2*b + narrowing_select_q] = opb.w32[b] >>
                opa.w32[b][4:0];
            EW32: for (int b = 0; b < 1; b++) res.w32[2*b + narrowing_select_q] = opb.w64[b] >>
                opa.w64[b][5:0];
          endcase
        VNSRA: unique case (vew_q)
            EW8 : for (int b = 0; b < 4; b++) res.w8 [2*b + narrowing_select_q] =
                $signed(opb.w16[b]) >>> opa.w16[b][3:0];
            EW16: for (int b = 0; b < 2; b++) res.w16[2*b + narrowing_select_q] =
                $signed(opb.w32[b]) >>> opa.w32[b][4:0];
            EW32: for (int b = 0; b < 1; b++) res.w32[2*b + narrowing_select_q] =
                $signed(opb.w64[b]) >>> opa.w64[b][5:0];
          endcase

        // Fixed point shift instructions
        VSSRA: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [7:0] sra = $signed(opb.w8 [b]) >>> opa.w8 [b][2:0];
                res.w8[b] = sra + rm_q[b];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [15:0] sra = $signed(opb.w16[b]) >>> opa.w16[b][3:0];
                res.w16[b] = sra + rm_q[b];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [31:0] sra = $signed(opb.w32[b]) >>> opa.w32[b][4:0];
                res.w32[b] = sra + rm_q[b];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [63:0] sra = $signed(opb.w64[b]) >>> opa.w64[b][5:0];
                res.w64[b] = sra + rm_q[b];
              end
          endcase
        VSSRL: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8: for (int b = 0; b < 8; b++) begin
                automatic logic [8:0] srl = opb.w8 [b] >> opa.w8 [b];
                res.w8[b] = srl + rm_q[b];
              end
            EW16: for (int b = 0; b < 4; b++) begin
                automatic logic [16:0] srl = opb.w16[b] >> opa.w16[b];
                res.w16[b] = srl + rm_q[b];
              end
            EW32: for (int b = 0; b < 2; b++) begin
                automatic logic [32:0] srl = opb.w32[b] >> opa.w32[b];
                res.w32[b] = srl + rm_q[b];
              end
            EW64: for (int b = 0; b < 1; b++) begin
                automatic logic [64:0] srl = opb.w64[b] >> opa.w64[b];
                res.w64[b] = srl + rm_q[b];
              end
          endcase

        // Fixed point clip instructions
        VNCLIP: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8 : for (int b = 0; b < 4; b++) begin
                automatic logic [15:0] clip = $signed(opb.w16[b]) >>> opa.w16[b][3:0];
                vxsat.w8[b]   = |clip[15:8];
                res.w8 [2*b + narrowing_select_q] = ($signed(opb.w16[b]) >>> opa.w16[b][3:0]) + rm_q[b];
              end
            EW16: for (int b = 0; b < 2; b++) begin
                automatic logic [31:0] clip = $signed(opb.w32[b]) >>> opa.w32[b][4:0];
                vxsat.w8[b]   = |clip[31:16];
                res.w16[2*b + narrowing_select_q] = ($signed(opb.w32[b]) >>> opa.w32[b][4:0]) + rm_q[b];
              end
            EW32: for (int b = 0; b < 1; b++) begin
                automatic logic [63:0] clip = $signed(opb.w64[b]) >>> opa.w64[b][5:0];
                vxsat.w8[b]   = |clip[63:32];
                res.w32[2*b + narrowing_select_q] = ($signed(opb.w64[b]) >>> opa.w64[b][5:0]) + rm_q[b];
              end
          endcase
        VNCLIPU: if (FixPtSupport == FixedPointEnable) unique case (vew_q)
            EW8 : for (int b = 0; b < 4; b++) begin
                automatic logic [15:0] clipu = opb.w16[b] >> opa.w16[b][3:0];
                vxsat.w8[b]   = |clipu[15:8];
                res.w8 [2*b + narrowing_select_q] = (opb.w16[b] >> opa.w16[b][3:0]) + rm_q[b];
              end
            EW16: for (int b = 0; b < 2; b++) begin
                automatic logic [31:0] clipu = opb.w32[b] >> opa.w32[b][4:0];
                vxsat.w8[b]   = |clipu[31:16];
                res.w16[2*b + narrowing_select_q] = (opb.w32[b] >> opa.w32[b][4:0]) + rm_q[b];
              end
            EW32: for (int b = 0; b < 1; b++) begin
                automatic logic [63:0] clipu = opb.w64[b] >> opa.w64[b][5:0];
                vxsat.w8[b]   = |clipu[63:32];
                res.w32[2*b + narrowing_select_q] = (opb.w64[b] >> opa.w64[b][5:0]) + rm_q[b];
              end
          endcase

        // Merge instructions
        VMERGE: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] = mask_q[1*b] ? opa.w8 [b] : opb.w8 [b];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] = mask_q[2*b] ? opa.w16[b] : opb.w16[b];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] = mask_q[4*b] ? opa.w32[b] : opb.w32[b];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] = mask_q[8*b] ? opa.w64[b] : opb.w64[b];
          endcase

        // Scalar move
        VMVSX, VFMVSF: res = opa;

        // Comparison instructions
        VMIN, VMINU, VMAX, VMAXU,
        VREDMINU, VREDMIN, VREDMAXU, VREDMAX: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b] =
                (less[1*b] ^ (op_q == VMAX || op_q == VMAXU || op_q == VREDMAXU || op_q == VREDMAX)) ? opb.w8 [b] : opa.w8 [b];
            EW16: for (int b = 0; b < 4; b++) res.w16[b] =
                (less[2*b] ^ (op_q == VMAX || op_q == VMAXU || op_q == VREDMAXU || op_q == VREDMAX)) ? opb.w16[b] : opa.w16[b];
            EW32: for (int b = 0; b < 2; b++) res.w32[b] =
                (less[4*b] ^ (op_q == VMAX || op_q == VMAXU || op_q == VREDMAXU || op_q == VREDMAX)) ? opb.w32[b] : opa.w32[b];
            EW64: for (int b = 0; b < 1; b++) res.w64[b] =
                (less[8*b] ^ (op_q == VMAX || op_q == VMAXU || op_q == VREDMAXU || op_q == VREDMAX)) ? opb.w64[b] : opa.w64[b];
          endcase
        VMSEQ, VMSNE: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b][1:0] =
                {mask_q[1*b], equal[1*b] ^ (op_q == VMSNE)};
            EW16: for (int b = 0; b < 4; b++) res.w16[b][1:0] =
                {mask_q[2*b], equal[2*b] ^ (op_q == VMSNE)};
            EW32: for (int b = 0; b < 2; b++) res.w32[b][1:0] =
                {mask_q[4*b], equal[4*b] ^ (op_q == VMSNE)};
            EW64: for (int b = 0; b < 1; b++) res.w64[b][1:0] =
                {mask_q[8*b], equal[8*b] ^ (op_q == VMSNE)};
          endcase
        VMSLT, VMSLTU: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b][1:0] = {mask_q[1*b], less[1*b]};
            EW16: for (int b = 0; b < 4; b++) res.w16[b][1:0] = {mask_q[2*b], less[2*b]};
            EW32: for (int b = 0; b < 2; b++) res.w32[b][1:0] = {mask_q[4*b], less[4*b]};
            EW64: for (int b = 0; b < 1; b++) res.w64[b][1:0] = {mask_q[8*b], less[8*b]};
          endcase
        VMSLE, VMSLEU, VMSGT, VMSGTU: unique case (vew_q)
            EW8 : for (int b = 0; b < 8; b++) res.w8 [b][1:0] =
                {mask_q[1*b], (less[1*b] || equal[1*b]) ^ (op_q inside {VMSGT, VMSGTU})};
            EW16: for (int b = 0; b < 4; b++) res.w16[b][1:0] =
                {mask_q[2*b], (less[2*b] || equal[2*b]) ^ (op_q inside {VMSGT, VMSGTU})};
            EW32: for (int b = 0; b < 2; b++) res.w32[b][1:0] =
                {mask_q[4*b], (less[4*b] || equal[4*b]) ^ (op_q inside {VMSGT, VMSGTU})};
            EW64: for (int b = 0; b < 1; b++) res.w64[b][1:0] =
                {mask_q[8*b], (less[8*b] || equal[8*b]) ^ (op_q inside {VMSGT, VMSGTU})};
          endcase

        default:;
      endcase
  end : p_alu

  //////////////////
  //  Assertions  //
  //////////////////

  if (DataWidth != $bits(alu_operand_t))
    $error("[simd_valu] The SIMD vector ALU only works for a datapath 64-bit wide.");

endmodule : simd_alu
