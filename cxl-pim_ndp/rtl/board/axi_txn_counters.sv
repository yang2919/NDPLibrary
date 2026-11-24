// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

module axi_txn_counters #(
    parameter type          axi_req_t   = logic,
    parameter type          axi_resp_t  = logic,

    parameter bit           IdFilterEn  = 1'b0,
    parameter int  unsigned IdFilterBit = 0
  ) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  axi_req_t    axi_req_i,
    input  axi_resp_t   axi_resp_i,
    output logic [63:0] rd_lat_sum_o,
    output logic [63:0] rd_count_o,
    output logic [63:0] rd_outst_max_o,
    output logic [63:0] wr_lat_sum_o,
    output logic [63:0] wr_count_o,
    output logic [63:0] wr_outst_max_o
  );

  logic ar_fire, r_done, aw_fire, b_done;
  assign ar_fire = axi_req_i.ar_valid & axi_resp_i.ar_ready
                 & (!IdFilterEn | axi_req_i.ar.id[IdFilterBit]);
  assign r_done  = axi_resp_i.r_valid & axi_req_i.r_ready & axi_resp_i.r.last
                 & (!IdFilterEn | axi_resp_i.r.id[IdFilterBit]);
  assign aw_fire = axi_req_i.aw_valid & axi_resp_i.aw_ready
                 & (!IdFilterEn | axi_req_i.aw.id[IdFilterBit]);
  assign b_done  = axi_resp_i.b_valid & axi_req_i.b_ready
                 & (!IdFilterEn | axi_resp_i.b.id[IdFilterBit]);

  logic [15:0] rd_outst_q, wr_outst_q;
  logic [15:0] rd_outst_max_q, wr_outst_max_q;
  logic [63:0] rd_lat_sum_q, wr_lat_sum_q;
  logic [31:0] rd_count_q, wr_count_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_outst_q     <= '0; wr_outst_q     <= '0;
      rd_outst_max_q <= '0; wr_outst_max_q <= '0;
      rd_lat_sum_q   <= '0; wr_lat_sum_q   <= '0;
      rd_count_q     <= '0; wr_count_q     <= '0;
    end else begin
      rd_outst_q   <= rd_outst_q + 16'(ar_fire) - 16'(r_done);
      rd_lat_sum_q <= rd_lat_sum_q + 64'(rd_outst_q);
      rd_count_q   <= rd_count_q + 32'(r_done);
      if (rd_outst_q > rd_outst_max_q) rd_outst_max_q <= rd_outst_q;

      wr_outst_q   <= wr_outst_q + 16'(aw_fire) - 16'(b_done);
      wr_lat_sum_q <= wr_lat_sum_q + 64'(wr_outst_q);
      wr_count_q   <= wr_count_q + 32'(b_done);
      if (wr_outst_q > wr_outst_max_q) wr_outst_max_q <= wr_outst_q;
    end
  end

  assign rd_lat_sum_o   = rd_lat_sum_q;
  assign rd_count_o     = 64'(rd_count_q);
  assign rd_outst_max_o = 64'(rd_outst_max_q);
  assign wr_lat_sum_o   = wr_lat_sum_q;
  assign wr_count_o     = 64'(wr_count_q);
  assign wr_outst_max_o = 64'(wr_outst_max_q);

endmodule : axi_txn_counters
