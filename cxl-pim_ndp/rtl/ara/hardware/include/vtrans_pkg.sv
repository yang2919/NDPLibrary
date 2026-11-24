// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0
// Author: Jae Hyung Ko

package vtrans_pkg;
  import vtrans_tables_pkg::*;

  localparam logic [28:0] LOG2E_Q28 = 29'd387270501;

  typedef enum logic [2:0] {
    VT_NORMAL   = 3'd0,
    VT_RET_ONE  = 3'd1,
    VT_RET_INF  = 3'd2,
    VT_RET_ZERO = 3'd3,
    VT_RET_NAN  = 3'd4
  } vt_cls_e;

  typedef struct packed {
    vt_cls_e            cls;
    logic signed [32:0] xfix;
  } vt_exp_s0_t;

  typedef struct packed {
    vt_cls_e            cls;
    logic signed [9:0]  k;
    logic [19:0]        f;
  } vt_exp_s1_t;

  typedef struct packed {
    vt_cls_e            cls;
    logic signed [9:0]  k;
    logic [26:0]        tiv;
    logic [16:0]        to1;
    logic [12:0]        to0;
  } vt_exp_s2_t;

  function automatic vt_exp_s0_t vtrans_exp_s0(logic [31:0] x);
    vt_exp_s0_t s;
    logic        sgn;
    logic [7:0]  ex;
    logic [23:0] mant;
    int          sh;
    sgn  = x[31];
    ex   = x[30:23];
    mant = {1'b1, x[22:0]};
    s.cls  = VT_NORMAL;
    s.xfix = '0;
    if (ex == 8'hFF) begin
      if (x[22:0] != '0)  s.cls = VT_RET_NAN;
      else                s.cls = sgn ? VT_RET_ZERO : VT_RET_INF;
    end else if (ex >= 8'd134) begin
      s.cls = sgn ? VT_RET_ZERO : VT_RET_INF;
    end else if (ex < 8'd100) begin
      s.cls = VT_RET_ONE;
    end else begin

      sh     = 32'd126 - {24'd0, ex};
      s.xfix = signed'({1'b0, (sh >= 0) ? (32'(mant) >> sh) : (32'(mant) << -sh)});
      if (sgn) s.xfix = -s.xfix;
    end
    return s;
  endfunction

  function automatic vt_exp_s1_t vtrans_exp_s1(vt_exp_s0_t s0);
    vt_exp_s1_t s;
    logic signed [62:0] prod;
    logic signed [29:0] p;
    prod  = s0.xfix * signed'({1'b0, LOG2E_Q28});
    p     = 30'(prod >>> 32);
    s.cls = s0.cls;
    s.k   = p[29:20];
    s.f   = p[19:0];
    return s;
  endfunction

  function automatic vt_exp_s2_t vtrans_exp_s2(vt_exp_s1_t s1);
    vt_exp_s2_t s;
    logic [18:0] w;
    logic [9:0]  a;
    w     = s1.f[19:1];
    a     = w[18:9];
    s.cls = s1.cls;
    s.k   = s1.k;
    s.tiv = VtExpTiv[a];
    s.to1 = VtExpTo1[{a[9:3], w[8:5]}];
    s.to0 = VtExpTo0[{a[9:6], w[4:0]}];
    return s;
  endfunction

  function automatic logic [31:0] vtrans_exp_s3(vt_exp_s2_t s2);
    logic signed [27:0] y;
    logic [22:0]        frac;
    logic               rnd, carry;
    logic signed [9:0]  e_out;
    unique case (s2.cls)
      VT_RET_NAN:  return 32'h7FC00000;
      VT_RET_INF:  return 32'h7F80_0000;
      VT_RET_ZERO: return 32'h0000_0000;
      VT_RET_ONE:  return 32'h3F80_0000;
      default: ;
    endcase
    y = signed'({1'b0, s2.tiv}) + 28'(signed'(s2.to1)) + 28'(signed'(s2.to0));

    if (y[26]) begin
      frac  = y[25:3];  rnd = y[2] & (y[1] | y[0] | frac[0]);
      e_out = s2.k + 10'sd127;
    end else begin
      frac  = y[24:2];  rnd = y[1] & (y[0] | frac[0]);
      e_out = s2.k + 10'sd126;
    end
    {carry, frac} = {1'b0, frac} + {23'd0, rnd};
    if (carry) e_out = e_out + 10'sd1;
    if (e_out >= 10'sd255) return 32'h7F80_0000;
    if (e_out <= 10'sd0)   return 32'h0000_0000;
    return {1'b0, e_out[7:0], frac};
  endfunction

  function automatic logic [31:0] vtrans_exp_fp32(logic [31:0] x);
    return vtrans_exp_s3(vtrans_exp_s2(vtrans_exp_s1(vtrans_exp_s0(x))));
  endfunction

  typedef struct packed {
    vt_cls_e     cls;
    logic        sgn;
    logic [7:0]  ex;
    logic [19:0] w;
  } vt_rcp_s0_t;

  typedef struct packed {
    vt_cls_e     cls;
    logic        sgn;
    logic [7:0]  ex;
    logic [25:0] tiv;
    logic [14:0] to1;
    logic [10:0] to0;
  } vt_rcp_s2_t;

  function automatic vt_rcp_s0_t vtrans_rcp_s0(logic [31:0] x);
    vt_rcp_s0_t s;
    s.sgn = x[31];
    s.ex  = x[30:23];
    s.w   = x[22:3];
    s.cls = VT_NORMAL;
    if (s.ex == 8'hFF) s.cls = (x[22:0] != '0) ? VT_RET_NAN : VT_RET_ZERO;
    else if (s.ex == 8'd0) s.cls = VT_RET_INF;
    return s;
  endfunction

  function automatic vt_rcp_s2_t vtrans_rcp_s2(vt_rcp_s0_t s0);
    vt_rcp_s2_t s;
    logic [10:0] a;
    a     = s0.w[19:9];
    s.cls = s0.cls;
    s.sgn = s0.sgn;
    s.ex  = s0.ex;
    s.tiv = VtRcpTiv[a];
    s.to1 = VtRcpTo1[{a[10:4], s0.w[8:5]}];
    s.to0 = VtRcpTo0[{a[10:5], s0.w[4:0]}];
    return s;
  endfunction

  function automatic logic [31:0] vtrans_rcp_s3(vt_rcp_s2_t s2);
    logic signed [27:0] y;
    logic [22:0]        frac;
    logic               rnd, carry;
    logic signed [9:0]  e_out;
    unique case (s2.cls)
      VT_RET_NAN:  return 32'h7FC00000;
      VT_RET_ZERO: return {s2.sgn, 31'd0};
      VT_RET_INF:  return {s2.sgn, 8'hFF, 23'd0};
      default: ;
    endcase
    y = signed'({2'b0, s2.tiv}) + 28'(signed'(s2.to1)) + 28'(signed'(s2.to0));

    if (y[26]) begin
      frac  = y[25:3];  rnd = y[2] & (y[1] | y[0] | frac[0]);
      e_out = 10'sd254 - signed'({2'b0, s2.ex});
    end else if (y[25]) begin
      frac  = y[24:2];  rnd = y[1] & (y[0] | frac[0]);
      e_out = 10'sd253 - signed'({2'b0, s2.ex});
    end else begin
      frac  = y[23:1];  rnd = y[0] & frac[0];
      e_out = 10'sd252 - signed'({2'b0, s2.ex});
    end
    {carry, frac} = {1'b0, frac} + {23'd0, rnd};
    if (carry) e_out = e_out + 10'sd1;
    if (e_out >= 10'sd255) return {s2.sgn, 8'hFF, 23'd0};
    if (e_out <= 10'sd0)   return {s2.sgn, 31'd0};
    return {s2.sgn, e_out[7:0], frac};
  endfunction

  function automatic logic [31:0] vtrans_rcp_fp32(logic [31:0] x);
    return vtrans_rcp_s3(vtrans_rcp_s2(vtrans_rcp_s0(x)));
  endfunction

  function automatic logic [31:0] vtrans_h2s(logic [15:0] h);
    logic sgn; logic [4:0] e5; logic [9:0] m10;
    sgn = h[15]; e5 = h[14:10]; m10 = h[9:0];
    if (e5 == 5'd31) return m10 != '0 ? 32'h7FC00000 : {sgn, 8'hFF, 23'd0};
    if (e5 == 5'd0)  return {sgn, 31'd0};
    return {sgn, 8'(e5) + 8'd112, m10, 13'd0};
  endfunction

  function automatic logic [15:0] vtrans_s2h(logic [31:0] s);
    logic sgn; logic signed [9:0] e16; logic [9:0] frac; logic rnd, carry;
    sgn = s[31];
    if (s[30:23] == 8'hFF) return s[22:0] != '0 ? 16'h7E00 : {sgn, 5'd31, 10'd0};
    e16  = signed'({2'b0, s[30:23]}) - 10'sd112;
    frac = s[22:13];
    rnd  = s[12] & (|s[11:0] | frac[0]);
    {carry, frac} = {1'b0, frac} + {10'd0, rnd};
    e16  = e16 + (carry ? 10'sd1 : 10'sd0);
    if (e16 >= 10'sd31) return {sgn, 5'd31, 10'd0};
    if (e16 <= 10'sd0)  return {sgn, 15'd0};
    return {sgn, e16[4:0], carry ? 10'd0 : frac};
  endfunction

  function automatic logic [15:0] vtrans_exp_fp16(logic [15:0] x);
    return vtrans_s2h(vtrans_exp_fp32(vtrans_h2s(x)));
  endfunction

  function automatic logic [15:0] vtrans_rcp_fp16(logic [15:0] x);
    return vtrans_s2h(vtrans_rcp_fp32(vtrans_h2s(x)));
  endfunction

endpackage
