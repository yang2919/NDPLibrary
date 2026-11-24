// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

module perf_registers #(
    parameter int  unsigned NumTaps         = 3,
    parameter type          axi_lite_req_t  = logic,
    parameter type          axi_lite_resp_t = logic,
    parameter int  unsigned DataWidth       = 64
  ) (
    input  logic                        clk_i,
    input  logic                        rst_ni,
    input  axi_lite_req_t               axi_lite_slave_req_i,
    output axi_lite_resp_t              axi_lite_slave_resp_o,
    input  logic                 [63:0] vinsn_active_cycles_i,
    input  logic [NumTaps-1:0]   [63:0] rd_lat_sum_i,
    input  logic [NumTaps-1:0]   [63:0] rd_count_i,
    input  logic [NumTaps-1:0]   [63:0] rd_outst_max_i,
    input  logic [NumTaps-1:0]   [63:0] wr_lat_sum_i,
    input  logic [NumTaps-1:0]   [63:0] wr_count_i,
    input  logic [NumTaps-1:0]   [63:0] wr_outst_max_i
  );

  localparam logic [63:0] PerfMagic = 64'h5045_5246_0000_0001;

  logic [63:0] rdata;
  logic [11:0] araddr;
  assign araddr = axi_lite_slave_req_i.ar.addr[11:0];

  always_comb begin
    automatic logic [1:0] tap = araddr[7:6] - 2'd1;
    rdata = '0;
    if (araddr < 12'h040) begin
      unique case (araddr[5:3])
        3'd0:    rdata = PerfMagic;
        3'd1:    rdata = vinsn_active_cycles_i;
        default: rdata = '0;
      endcase
    end else if (tap < NumTaps[1:0] || NumTaps == 4) begin
      unique case (araddr[5:3])
        3'd0:    rdata = rd_lat_sum_i  [tap];
        3'd1:    rdata = rd_count_i    [tap];
        3'd2:    rdata = rd_outst_max_i[tap];
        3'd3:    rdata = wr_lat_sum_i  [tap];
        3'd4:    rdata = wr_count_i    [tap];
        3'd5:    rdata = wr_outst_max_i[tap];
        default: rdata = '0;
      endcase
    end
  end

  logic        r_valid_q;
  logic [63:0] r_data_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      r_valid_q <= 1'b0;
      r_data_q  <= '0;
    end else begin
      if (axi_lite_slave_req_i.ar_valid && !r_valid_q) begin
        r_valid_q <= 1'b1;
        r_data_q  <= rdata;
      end else if (r_valid_q && axi_lite_slave_req_i.r_ready) begin
        r_valid_q <= 1'b0;
      end
    end
  end

  assign axi_lite_slave_resp_o.ar_ready = !r_valid_q;
  assign axi_lite_slave_resp_o.r_valid  = r_valid_q;
  assign axi_lite_slave_resp_o.r.data   = r_data_q;
  assign axi_lite_slave_resp_o.r.resp   = axi_pkg::RESP_OKAY;

  logic aw_taken_q, w_taken_q, b_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_taken_q <= 1'b0;
      w_taken_q  <= 1'b0;
      b_valid_q  <= 1'b0;
    end else begin
      if (b_valid_q) begin
        if (axi_lite_slave_req_i.b_ready) begin
          b_valid_q  <= 1'b0;
          aw_taken_q <= 1'b0;
          w_taken_q  <= 1'b0;
        end
      end else begin
        if (axi_lite_slave_req_i.aw_valid && !aw_taken_q) aw_taken_q <= 1'b1;
        if (axi_lite_slave_req_i.w_valid  && !w_taken_q ) w_taken_q  <= 1'b1;
        if ((aw_taken_q || axi_lite_slave_req_i.aw_valid) &&
            (w_taken_q  || axi_lite_slave_req_i.w_valid )) b_valid_q <= 1'b1;
      end
    end
  end

  assign axi_lite_slave_resp_o.aw_ready = !aw_taken_q && !b_valid_q;
  assign axi_lite_slave_resp_o.w_ready  = !w_taken_q  && !b_valid_q;
  assign axi_lite_slave_resp_o.b_valid  = b_valid_q;
  assign axi_lite_slave_resp_o.b.resp   = axi_pkg::RESP_OKAY;

endmodule : perf_registers
