// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

module fpga_top_vhk158 (

    input  logic        sys_clk_p,
    input  logic        sys_clk_n,
    input  logic        cpu_resetn,
    input  logic        uart_rx_i,
    output logic        uart_tx_o,

    input  wire         ddr4_dimm0_sma_clk_clk_p,
    input  wire         ddr4_dimm0_sma_clk_clk_n,

    output wire         ddr4_dimm0_act_n,
    output wire [16:0]  ddr4_dimm0_adr,
    input  wire         ddr4_dimm0_alert_n,
    output wire [1:0]   ddr4_dimm0_ba,
    output wire [1:0]   ddr4_dimm0_bg,
    output wire         ddr4_dimm0_ck_c,
    output wire         ddr4_dimm0_ck_t,
    output wire         ddr4_dimm0_cke,
    output wire         ddr4_dimm0_cs_n,
    inout  wire [71:0]  ddr4_dimm0_dq,
    inout  wire [8:0]   ddr4_dimm0_dqs_c,
    inout  wire [8:0]   ddr4_dimm0_dqs_t,
    output wire         ddr4_dimm0_odt,
    output wire         ddr4_dimm0_par,
    output wire         ddr4_dimm0_reset_n,

    output logic [3:0]  gpio_led
);

  localparam int unsigned NR_LANES        = 2;
  localparam int unsigned VLEN            = 2048;
  localparam int unsigned AXI_ADDR_WIDTH  = 64;

  localparam int unsigned AXI_DATA_WIDTH  = 32 * NR_LANES;
  localparam int unsigned AXI_ID_WIDTH    = 5;
  localparam int unsigned AXI_USER_WIDTH  = 1;

  logic soc_clk;
  logic clkwiz_locked;
  logic soc_rst_n;

  clk_wiz_vhk158 u_clkwiz (
      .clk_in1_p (sys_clk_p),
      .clk_in1_n (sys_clk_n),
      .clk_out1  (soc_clk),
      .locked    (clkwiz_locked),
      .resetn    (1'b1)
  );

  logic btn_meta, btn_sync, cpu_resetn_db;
  logic [19:0] db_cnt;
  always_ff @(posedge soc_clk or negedge clkwiz_locked) begin
    if (!clkwiz_locked) begin
      btn_meta <= 1'b0; btn_sync <= 1'b0; cpu_resetn_db <= 1'b0; db_cnt <= '0;
    end else begin
      btn_meta <= cpu_resetn;
      btn_sync <= btn_meta;
      if (btn_sync != cpu_resetn_db) begin
        db_cnt <= db_cnt + 1'b1;
        if (&db_cnt) cpu_resetn_db <= btn_sync;
      end else begin
        db_cnt <= '0;
      end
    end
  end

  logic [3:0] pmc_gpio_o, pmc_gpio_t, pmc_gpio_i;
  logic       pl_soft_rst;
  assign pl_soft_rst = pmc_gpio_o[0] & ~pmc_gpio_t[0];

  logic rst_n_async, rst_n_meta, rst_n_sync;
  assign rst_n_async = ~cpu_resetn_db & clkwiz_locked & ~pl_soft_rst;

  always_ff @(posedge soc_clk or negedge rst_n_async) begin
    if (!rst_n_async) begin
      rst_n_meta <= 1'b0;
      rst_n_sync <= 1'b0;
    end else begin
      rst_n_meta <= 1'b1;
      rst_n_sync <= rst_n_meta;
    end
  end
  assign soc_rst_n = rst_n_sync;

  logic [25:0] blink_cnt;
  always_ff @(posedge soc_clk) blink_cnt <= blink_cnt + 1'b1;

  assign gpio_led[0] = clkwiz_locked;
  assign gpio_led[1] = ~cpu_resetn;
  assign gpio_led[2] = pl_soft_rst;
  assign gpio_led[3] = blink_cnt[25];

  logic axi_ar_seen;
  always_ff @(posedge soc_clk or negedge soc_rst_n) begin
    if (!soc_rst_n)                            axi_ar_seen <= 1'b0;
    else if (l2_axi.ar_valid && l2_axi.ar_ready) axi_ar_seen <= 1'b1;
  end

  assign pmc_gpio_i = {axi_ar_seen, blink_cnt[25], soc_rst_n, pl_soft_rst};

  AXI_BUS #(
      .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
      .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
      .AXI_ID_WIDTH   (AXI_ID_WIDTH),
      .AXI_USER_WIDTH (AXI_USER_WIDTH)
  ) l2_axi ();

  logic        uart_penable;
  logic        uart_pwrite;
  logic [31:0] uart_paddr;
  logic        uart_psel;
  logic [31:0] uart_pwdata;
  logic [31:0] uart_prdata;
  logic        uart_pready;
  logic        uart_pslverr;

  ara_soc_vhk158 #(
      .NrLanes      (NR_LANES),
      .VLEN         (VLEN),
      .AxiDataWidth (AXI_DATA_WIDTH),
      .AxiAddrWidth (AXI_ADDR_WIDTH),
      .AxiUserWidth (AXI_USER_WIDTH),
      .AxiIdWidth   (AXI_ID_WIDTH)
  ) u_soc (
      .clk_i           (soc_clk),
      .rst_ni          (soc_rst_n),
      .exit_o          (),
      .hw_cnt_en_o     (),
      .scan_enable_i   (1'b0),
      .scan_data_i     (1'b0),
      .scan_data_o     (),
      .uart_penable_o  (uart_penable),
      .uart_pwrite_o   (uart_pwrite),
      .uart_paddr_o    (uart_paddr),
      .uart_psel_o     (uart_psel),
      .uart_pwdata_o   (uart_pwdata),
      .uart_prdata_i   (uart_prdata),
      .uart_pready_i   (uart_pready),
      .uart_pslverr_i  (uart_pslverr),
      .l2_axi_mst      (l2_axi)
  );

  design_1_wrapper u_bd (

      .pl_clk                    (soc_clk),

      .pl_axi_in_awid            (l2_axi.aw_id),
      .pl_axi_in_awaddr          (l2_axi.aw_addr),
      .pl_axi_in_awlen           (l2_axi.aw_len),
      .pl_axi_in_awsize          (l2_axi.aw_size),
      .pl_axi_in_awburst         (l2_axi.aw_burst),
      .pl_axi_in_awlock          (l2_axi.aw_lock),
      .pl_axi_in_awcache         (l2_axi.aw_cache),
      .pl_axi_in_awprot          (l2_axi.aw_prot),
      .pl_axi_in_awqos           (l2_axi.aw_qos),
      .pl_axi_in_awregion        (l2_axi.aw_region),
      .pl_axi_in_awvalid         (l2_axi.aw_valid),
      .pl_axi_in_awready         (l2_axi.aw_ready),
      .pl_axi_in_wdata           (l2_axi.w_data),
      .pl_axi_in_wstrb           (l2_axi.w_strb),
      .pl_axi_in_wlast           (l2_axi.w_last),
      .pl_axi_in_wvalid          (l2_axi.w_valid),
      .pl_axi_in_wready          (l2_axi.w_ready),
      .pl_axi_in_bid             (l2_axi.b_id),
      .pl_axi_in_bresp           (l2_axi.b_resp),
      .pl_axi_in_bvalid          (l2_axi.b_valid),
      .pl_axi_in_bready          (l2_axi.b_ready),
      .pl_axi_in_arid            (l2_axi.ar_id),
      .pl_axi_in_araddr          (l2_axi.ar_addr),
      .pl_axi_in_arlen           (l2_axi.ar_len),
      .pl_axi_in_arsize          (l2_axi.ar_size),
      .pl_axi_in_arburst         (l2_axi.ar_burst),
      .pl_axi_in_arlock          (l2_axi.ar_lock),
      .pl_axi_in_arcache         (l2_axi.ar_cache),
      .pl_axi_in_arprot          (l2_axi.ar_prot),
      .pl_axi_in_arqos           (l2_axi.ar_qos),
      .pl_axi_in_arregion        (l2_axi.ar_region),
      .pl_axi_in_arvalid         (l2_axi.ar_valid),
      .pl_axi_in_arready         (l2_axi.ar_ready),
      .pl_axi_in_rid             (l2_axi.r_id),
      .pl_axi_in_rdata           (l2_axi.r_data),
      .pl_axi_in_rresp           (l2_axi.r_resp),
      .pl_axi_in_rlast           (l2_axi.r_last),
      .pl_axi_in_rvalid          (l2_axi.r_valid),
      .pl_axi_in_rready          (l2_axi.r_ready),

      .ddr4_dimm0_sma_clk_clk_p  (ddr4_dimm0_sma_clk_clk_p),
      .ddr4_dimm0_sma_clk_clk_n  (ddr4_dimm0_sma_clk_clk_n),

      .pmc_gpio_o                (pmc_gpio_o),
      .pmc_gpio_t                (pmc_gpio_t),
      .pmc_gpio_i                (pmc_gpio_i),

      .ddr4_dimm0_act_n          (ddr4_dimm0_act_n),
      .ddr4_dimm0_adr            (ddr4_dimm0_adr),
      .ddr4_dimm0_alert_n        (ddr4_dimm0_alert_n),
      .ddr4_dimm0_ba             (ddr4_dimm0_ba),
      .ddr4_dimm0_bg             (ddr4_dimm0_bg),
      .ddr4_dimm0_ck_c           (ddr4_dimm0_ck_c),
      .ddr4_dimm0_ck_t           (ddr4_dimm0_ck_t),
      .ddr4_dimm0_cke            (ddr4_dimm0_cke),
      .ddr4_dimm0_cs_n           (ddr4_dimm0_cs_n),
      .ddr4_dimm0_dq             (ddr4_dimm0_dq),
      .ddr4_dimm0_dqs_c          (ddr4_dimm0_dqs_c),
      .ddr4_dimm0_dqs_t          (ddr4_dimm0_dqs_t),
      .ddr4_dimm0_odt            (ddr4_dimm0_odt),
      .ddr4_dimm0_par            (ddr4_dimm0_par),
      .ddr4_dimm0_reset_n        (ddr4_dimm0_reset_n)
  );

  apb_uart u_uart (
      .CLK     (soc_clk),
      .RSTN    (soc_rst_n),
      .PSEL    (uart_psel),
      .PENABLE (uart_penable),
      .PWRITE  (uart_pwrite),
      .PADDR   (uart_paddr[4:2]),
      .PWDATA  (uart_pwdata),
      .PRDATA  (uart_prdata),
      .PREADY  (uart_pready),
      .PSLVERR (uart_pslverr),
      .INT     (),
      .OUT1N   (),
      .OUT2N   (),
      .RTSN    (),
      .DTRN    (),
      .CTSN    (1'b0),
      .DSRN    (1'b0),
      .DCDN    (1'b0),
      .RIN     (1'b0),
      .SIN     (uart_rx_i),
      .SOUT    (uart_tx_o)
  );

endmodule
