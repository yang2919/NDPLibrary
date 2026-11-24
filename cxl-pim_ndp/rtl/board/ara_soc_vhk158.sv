// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module ara_soc_vhk158 import axi_pkg::*; import ara_pkg::*; #(

    parameter  int           unsigned NrLanes      = 0,
    parameter  int           unsigned VLEN         = 0,
    parameter  int           unsigned OSSupport    = 1,

    parameter  fpu_support_e          FPUSupport   = FPUSupportByteHalfSingle,

    parameter  bit                    Cva6ScalarD  = 1'b1,

    parameter  fpext_support_e        FPExtSupport = FPExtSupportEnable,

    parameter  vtrans_support_e       VtransSupport = VtransSupportEnable,

    parameter  fixpt_support_e        FixPtSupport = FixedPointEnable,

    parameter  seg_support_e          SegSupport   = SegSupportEnable,

    parameter  int           unsigned AxiDataWidth = 32*NrLanes,
    parameter  int           unsigned AxiAddrWidth = 64,
    parameter  int           unsigned AxiUserWidth = 1,
    parameter  int           unsigned AxiIdWidth   = 5,

    parameter  int           unsigned AxiRespDelay = 200,

    parameter  int           unsigned L2NumWords   = (2**22) / NrLanes,

    localparam type                   axi_data_t   = logic [AxiDataWidth-1:0],
    localparam type                   axi_strb_t   = logic [AxiDataWidth/8-1:0],
    localparam type                   axi_addr_t   = logic [AxiAddrWidth-1:0],
    localparam type                   axi_user_t   = logic [AxiUserWidth-1:0],
    localparam type                   axi_id_t     = logic [AxiIdWidth-1:0]
  ) (
    input  logic        clk_i,
    input  logic        rst_ni,
    output logic [63:0] exit_o,
    output logic [63:0] hw_cnt_en_o,

    input  logic        scan_enable_i,
    input  logic        scan_data_i,
    output logic        scan_data_o,

    output logic        uart_penable_o,
    output logic        uart_pwrite_o,
    output logic [31:0] uart_paddr_o,
    output logic        uart_psel_o,
    output logic [31:0] uart_pwdata_o,
    input  logic [31:0] uart_prdata_i,
    input  logic        uart_pready_i,
    input  logic        uart_pslverr_i,

    AXI_BUS.Master      l2_axi_mst
  );

  `include "axi/assign.svh"
  `include "axi/typedef.svh"
  `include "common_cells/registers.svh"
  `include "apb/typedef.svh"
  `include "ara/intf_typedef.svh"

  localparam NrAXIMasters = 1;

  typedef enum int unsigned {
    L2MEM = 0,
    UART  = 1,
    CTRL  = 2,
    PERF  = 3
  } axi_slaves_e;
  localparam NrAXISlaves = PERF + 1;

  localparam logic [63:0] DRAMLength = 64'h40000000;
  localparam logic [63:0] UARTLength = 64'h1000;
  localparam logic [63:0] CTRLLength = 64'h1000;
  localparam logic [63:0] PERFLength = 64'h1000;

  localparam logic [63:0] SPMLength  = 64'h80000;

  typedef enum logic [63:0] {
    SPMBase  = 64'h3000_0000,
    DRAMBase = 64'h4000_0000,
    UARTBase = 64'hC000_0000,
    CTRLBase = 64'hD000_0000,
    PERFBase = 64'hD001_0000
  } soc_bus_start_e;

  localparam AxiNarrowDataWidth = 64;
  localparam AxiNarrowStrbWidth = AxiNarrowDataWidth / 8;

  localparam AxiWideDataWidth   = AxiDataWidth;
  localparam AXiWideStrbWidth   = AxiWideDataWidth / 8;

  localparam AxiSocIdWidth  = AxiIdWidth - $clog2(NrAXIMasters);
  localparam AxiCoreIdWidth = AxiSocIdWidth - 1;

  typedef logic [AxiNarrowDataWidth-1:0] axi_narrow_data_t;
  typedef logic [AxiNarrowStrbWidth-1:0] axi_narrow_strb_t;
  typedef logic [AxiSocIdWidth-1:0] axi_soc_id_t;
  typedef logic [AxiCoreIdWidth-1:0] axi_core_id_t;

  `AXI_TYPEDEF_ALL(system, axi_addr_t, axi_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_TYPEDEF_ALL(ara_axi, axi_addr_t, axi_core_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_TYPEDEF_ALL(ariane_axi, axi_addr_t, axi_core_id_t, axi_narrow_data_t, axi_narrow_strb_t,
    axi_user_t)
  `AXI_TYPEDEF_ALL(soc_narrow, axi_addr_t, axi_soc_id_t, axi_narrow_data_t, axi_narrow_strb_t,
    axi_user_t)
  `AXI_TYPEDEF_ALL(soc_wide, axi_addr_t, axi_soc_id_t, axi_data_t, axi_strb_t, axi_user_t)
  `AXI_LITE_TYPEDEF_ALL(soc_narrow_lite, axi_addr_t, axi_narrow_data_t, axi_narrow_strb_t)

  system_req_t  system_axi_req_spill;
  system_resp_t system_axi_resp_spill;
  system_resp_t system_axi_resp_spill_del;
  system_req_t  system_axi_req;
  system_resp_t system_axi_resp;

  soc_wide_req_t    [NrAXISlaves-1:0] periph_wide_axi_req;
  soc_wide_resp_t   [NrAXISlaves-1:0] periph_wide_axi_resp;
  soc_narrow_req_t  [NrAXISlaves-1:0] periph_narrow_axi_req;
  soc_narrow_resp_t [NrAXISlaves-1:0] periph_narrow_axi_resp;

  localparam axi_pkg::xbar_cfg_t XBarCfg = '{
    NoSlvPorts        : NrAXIMasters,
    NoMstPorts        : NrAXISlaves,
    MaxMstTrans       : 4,
    MaxSlvTrans       : 4,
    FallThrough       : 1'b0,
    LatencyMode       : axi_pkg::CUT_MST_PORTS,
    PipelineStages    : 0,
    AxiIdWidthSlvPorts: AxiSocIdWidth,
    AxiIdUsedSlvPorts : AxiSocIdWidth,
    UniqueIds         : 1'b0,
    AxiAddrWidth      : AxiAddrWidth,
    AxiDataWidth      : AxiWideDataWidth,
    NoAddrRules       : NrAXISlaves,
    default           : '0
  };

  axi_pkg::xbar_rule_64_t [NrAXISlaves-1:0] routing_rules;
  assign routing_rules = '{
    '{idx: PERF, start_addr: PERFBase, end_addr: PERFBase + PERFLength},
    '{idx: CTRL, start_addr: CTRLBase, end_addr: CTRLBase + CTRLLength},
    '{idx: UART, start_addr: UARTBase, end_addr: UARTBase + UARTLength},
    '{idx: L2MEM, start_addr: DRAMBase, end_addr: DRAMBase + DRAMLength}
  };

  axi_xbar #(
    .Cfg          (XBarCfg                ),
    .slv_aw_chan_t(system_aw_chan_t       ),
    .mst_aw_chan_t(soc_wide_aw_chan_t     ),
    .w_chan_t     (system_w_chan_t        ),
    .slv_b_chan_t (system_b_chan_t        ),
    .mst_b_chan_t (soc_wide_b_chan_t      ),
    .slv_ar_chan_t(system_ar_chan_t       ),
    .mst_ar_chan_t(soc_wide_ar_chan_t     ),
    .slv_r_chan_t (system_r_chan_t        ),
    .mst_r_chan_t (soc_wide_r_chan_t      ),
    .slv_req_t    (system_req_t           ),
    .slv_resp_t   (system_resp_t          ),
    .mst_req_t    (soc_wide_req_t         ),
    .mst_resp_t   (soc_wide_resp_t        ),
    .rule_t       (axi_pkg::xbar_rule_64_t)
  ) i_soc_xbar (
    .clk_i                (clk_i               ),
    .rst_ni               (rst_ni              ),
    .test_i               (1'b0                ),
    .slv_ports_req_i      (system_axi_req      ),
    .slv_ports_resp_o     (system_axi_resp     ),
    .mst_ports_req_o      (periph_wide_axi_req ),
    .mst_ports_resp_i     (periph_wide_axi_resp),
    .addr_map_i           (routing_rules       ),
    .en_default_mst_port_i('0                  ),
    .default_mst_port_i   ('0                  )
  );

  soc_wide_req_t  l2mem_wide_axi_req_wo_atomics;
  soc_wide_resp_t l2mem_wide_axi_resp_wo_atomics;
  axi_atop_filter #(
    .AxiIdWidth     (AxiSocIdWidth  ),
    .AxiMaxWriteTxns(4              ),
    .axi_req_t      (soc_wide_req_t ),
    .axi_resp_t     (soc_wide_resp_t)
  ) i_l2mem_atop_filter (
    .clk_i     (clk_i                         ),
    .rst_ni    (rst_ni                        ),
    .slv_req_i (periph_wide_axi_req[L2MEM]    ),
    .slv_resp_o(periph_wide_axi_resp[L2MEM]   ),
    .mst_req_o (l2mem_wide_axi_req_wo_atomics ),
    .mst_resp_i(l2mem_wide_axi_resp_wo_atomics)
  );

  `AXI_ASSIGN_FROM_REQ(l2_axi_mst, l2mem_wide_axi_req_wo_atomics)
  `AXI_ASSIGN_TO_RESP (l2mem_wide_axi_resp_wo_atomics, l2_axi_mst)

  localparam int unsigned SPMNumWords = SPMLength / (AxiWideDataWidth / 8);

  ara_axi_req_t  ara_spm_axi_req;
  ara_axi_resp_t ara_spm_axi_resp;

  logic ara_idle;

  logic                           spma_req;
  logic                           spma_we;
  logic [AxiAddrWidth-1:0]        spma_addr;
  logic [AxiWideDataWidth/8-1:0]  spma_be;
  logic [AxiWideDataWidth-1:0]    spma_wdata;
  logic [AxiWideDataWidth-1:0]    spma_rdata;
  logic                           spma_rvalid;

  axi_to_mem #(
    .AddrWidth (AxiAddrWidth    ),
    .DataWidth (AxiWideDataWidth),
    .IdWidth   (AxiCoreIdWidth  ),
    .NumBanks  (1               ),
    .BufDepth  (1               ),
    .axi_req_t (ara_axi_req_t   ),
    .axi_resp_t(ara_axi_resp_t  )
  ) i_spm_axi_to_mem_ara (
    .clk_i       (clk_i            ),
    .rst_ni      (rst_ni           ),
    .axi_req_i   (ara_spm_axi_req  ),
    .axi_resp_o  (ara_spm_axi_resp ),
    .mem_req_o   (spma_req         ),
    .mem_gnt_i   (spma_req         ),
    .mem_we_o    (spma_we          ),
    .mem_addr_o  (spma_addr        ),
    .mem_strb_o  (spma_be          ),
    .mem_wdata_o (spma_wdata       ),
    .mem_rdata_i (spma_rdata       ),
    .mem_rvalid_i(spma_rvalid      ),
    .mem_atop_o  (     ),
    .busy_o      (     )
  );

  localparam int unsigned SPMAddrLsb = $clog2(AxiWideDataWidth/8);
  localparam int unsigned SPMAddrMsb = $clog2(SPMNumWords) - 1 + SPMAddrLsb;
  logic [$clog2(SPMNumWords)-1:0] spma_word_addr;
  assign spma_word_addr = spma_addr[SPMAddrMsb:SPMAddrLsb];

  tc_sram #(
    .NumWords (SPMNumWords     ),
    .NumPorts (1               ),
    .DataWidth(AxiWideDataWidth),
    .Latency  (1               ),
    .ImplKey  ("ultra"         )
  ) i_spm (
    .clk_i  (clk_i          ),
    .rst_ni (rst_ni         ),
    .req_i  (spma_req       ),
    .we_i   (spma_we        ),
    .addr_i (spma_word_addr ),
    .wdata_i(spma_wdata     ),
    .be_i   (spma_be        ),
    .rdata_o(spma_rdata     )
  );

  `FF(spma_rvalid, spma_req, 1'b0);

  `AXI_TYPEDEF_ALL(uart_axi, axi_addr_t, axi_soc_id_t, logic [31:0], logic [3:0], axi_user_t)
  `AXI_LITE_TYPEDEF_ALL(uart_lite, axi_addr_t, logic [31:0], logic [3:0])
  `APB_TYPEDEF_ALL(uart_apb, axi_addr_t, logic [31:0], logic [3:0])

  uart_axi_req_t   uart_axi_req;
  uart_axi_resp_t  uart_axi_resp;
  uart_lite_req_t  uart_lite_req;
  uart_lite_resp_t uart_lite_resp;
  uart_apb_req_t   uart_apb_req;
  uart_apb_resp_t  uart_apb_resp;

  assign uart_penable_o = uart_apb_req.penable;
  assign uart_pwrite_o  = uart_apb_req.pwrite;
  assign uart_paddr_o   = uart_apb_req.paddr;
  assign uart_psel_o    = uart_apb_req.psel;
  assign uart_pwdata_o  = uart_apb_req.pwdata;
  assign uart_apb_resp.prdata  = uart_prdata_i;
  assign uart_apb_resp.pready  = uart_pready_i;
  assign uart_apb_resp.pslverr = uart_pslverr_i;

  typedef struct packed {
    int unsigned idx;
    axi_addr_t   start_addr;
    axi_addr_t   end_addr;
  } uart_apb_rule_t;

  uart_apb_rule_t uart_apb_map = '{idx: 0, start_addr: '0, end_addr: '1};

  axi_lite_to_apb #(
    .NoApbSlaves     (32'd1           ),
    .NoRules         (32'd1           ),
    .AddrWidth       (AxiAddrWidth    ),
    .DataWidth       (32'd32          ),
    .PipelineRequest (1'b0            ),
    .PipelineResponse(1'b0            ),
    .axi_lite_req_t  (uart_lite_req_t ),
    .axi_lite_resp_t (uart_lite_resp_t),
    .apb_req_t       (uart_apb_req_t  ),
    .apb_resp_t      (uart_apb_resp_t ),
    .rule_t          (uart_apb_rule_t )
  ) i_axi_lite_to_apb_uart (
    .clk_i          (clk_i         ),
    .rst_ni         (rst_ni        ),
    .axi_lite_req_i (uart_lite_req ),
    .axi_lite_resp_o(uart_lite_resp),
    .apb_req_o      (uart_apb_req  ),
    .apb_resp_i     (uart_apb_resp ),
    .addr_map_i     (uart_apb_map  )
  );

  axi_to_axi_lite #(
    .AxiAddrWidth   (AxiAddrWidth    ),
    .AxiDataWidth   (32'd32          ),
    .AxiIdWidth     (AxiSocIdWidth   ),
    .AxiUserWidth   (AxiUserWidth    ),
    .AxiMaxWriteTxns(32'd1           ),
    .AxiMaxReadTxns (32'd1           ),
    .FallThrough    (1'b1            ),
    .full_req_t     (uart_axi_req_t  ),
    .full_resp_t    (uart_axi_resp_t ),
    .lite_req_t     (uart_lite_req_t ),
    .lite_resp_t    (uart_lite_resp_t)
  ) i_axi_to_axi_lite_uart (
    .clk_i     (clk_i         ),
    .rst_ni    (rst_ni        ),
    .test_i    (1'b0          ),
    .slv_req_i (uart_axi_req  ),
    .slv_resp_o(uart_axi_resp ),
    .mst_req_o (uart_lite_req ),
    .mst_resp_i(uart_lite_resp)
  );

  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth  ),
    .AxiMstPortDataWidth(32                ),
    .AxiAddrWidth       (AxiAddrWidth      ),
    .AxiIdWidth         (AxiSocIdWidth     ),
    .AxiMaxReads        (1                 ),
    .ar_chan_t          (soc_wide_ar_chan_t),
    .mst_r_chan_t       (uart_axi_r_chan_t ),
    .slv_r_chan_t       (soc_wide_r_chan_t ),
    .aw_chan_t          (uart_axi_aw_chan_t),
    .b_chan_t           (soc_wide_b_chan_t ),
    .mst_w_chan_t       (uart_axi_w_chan_t ),
    .slv_w_chan_t       (soc_wide_w_chan_t ),
    .axi_mst_req_t      (uart_axi_req_t    ),
    .axi_mst_resp_t     (uart_axi_resp_t   ),
    .axi_slv_req_t      (soc_wide_req_t    ),
    .axi_slv_resp_t     (soc_wide_resp_t   )
  ) i_axi_slave_uart_dwc (
    .clk_i     (clk_i                     ),
    .rst_ni    (rst_ni                    ),
    .slv_req_i (periph_wide_axi_req[UART] ),
    .slv_resp_o(periph_wide_axi_resp[UART]),
    .mst_req_o (uart_axi_req              ),
    .mst_resp_i(uart_axi_resp             )
  );

  soc_narrow_lite_req_t  axi_lite_ctrl_registers_req;
  soc_narrow_lite_resp_t axi_lite_ctrl_registers_resp;

  logic [63:0] event_trigger;

  axi_to_axi_lite #(
    .AxiAddrWidth   (AxiAddrWidth          ),
    .AxiDataWidth   (AxiNarrowDataWidth    ),
    .AxiIdWidth     (AxiSocIdWidth         ),
    .AxiUserWidth   (AxiUserWidth          ),
    .AxiMaxReadTxns (1                     ),
    .AxiMaxWriteTxns(1                     ),
    .FallThrough    (1'b0                  ),
    .full_req_t     (soc_narrow_req_t      ),
    .full_resp_t    (soc_narrow_resp_t     ),
    .lite_req_t     (soc_narrow_lite_req_t ),
    .lite_resp_t    (soc_narrow_lite_resp_t)
  ) i_axi_to_axi_lite (
    .clk_i     (clk_i                        ),
    .rst_ni    (rst_ni                       ),
    .test_i    (1'b0                         ),
    .slv_req_i (periph_narrow_axi_req[CTRL]  ),
    .slv_resp_o(periph_narrow_axi_resp[CTRL] ),
    .mst_req_o (axi_lite_ctrl_registers_req  ),
    .mst_resp_i(axi_lite_ctrl_registers_resp )
  );

  ctrl_registers #(
    .DRAMBaseAddr   (DRAMBase              ),
    .DRAMLength     (DRAMLength            ),
    .DataWidth      (AxiNarrowDataWidth    ),
    .AddrWidth      (AxiAddrWidth          ),
    .axi_lite_req_t (soc_narrow_lite_req_t ),
    .axi_lite_resp_t(soc_narrow_lite_resp_t)
  ) i_ctrl_registers (
    .clk_i                (clk_i                       ),
    .rst_ni               (rst_ni                      ),
    .axi_lite_slave_req_i (axi_lite_ctrl_registers_req ),
    .axi_lite_slave_resp_o(axi_lite_ctrl_registers_resp),
    .hw_cnt_en_o          (hw_cnt_en_o                 ),
    .dram_base_addr_o     (                ),
    .dram_end_addr_o      (                ),
    .exit_o               (exit_o                      ),
    .event_trigger_o      (event_trigger)
  );

  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiSocIdWidth       ),
    .AxiMaxReads        (2                   ),
    .ar_chan_t          (soc_wide_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    .slv_r_chan_t       (soc_wide_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    .slv_w_chan_t       (soc_wide_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    .axi_slv_req_t      (soc_wide_req_t      ),
    .axi_slv_resp_t     (soc_wide_resp_t     )
  ) i_axi_slave_ctrl_dwc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    .slv_req_i (periph_wide_axi_req[CTRL]   ),
    .slv_resp_o(periph_wide_axi_resp[CTRL]  ),
    .mst_req_o (periph_narrow_axi_req[CTRL] ),
    .mst_resp_i(periph_narrow_axi_resp[CTRL])
  );

  localparam int unsigned NumPerfTaps = 3;
  logic [NumPerfTaps-1:0][63:0] perf_rd_lat_sum, perf_rd_count, perf_rd_outst_max;
  logic [NumPerfTaps-1:0][63:0] perf_wr_lat_sum, perf_wr_count, perf_wr_outst_max;

  axi_txn_counters #(
    .axi_req_t (ara_axi_req_t ),
    .axi_resp_t(ara_axi_resp_t),
    .IdFilterEn(1'b0          )
  ) i_perf_tap_spm (
    .clk_i         (clk_i               ),
    .rst_ni        (rst_ni              ),
    .axi_req_i     (ara_spm_axi_req     ),
    .axi_resp_i    (ara_spm_axi_resp    ),
    .rd_lat_sum_o  (perf_rd_lat_sum  [0]),
    .rd_count_o    (perf_rd_count    [0]),
    .rd_outst_max_o(perf_rd_outst_max[0]),
    .wr_lat_sum_o  (perf_wr_lat_sum  [0]),
    .wr_count_o    (perf_wr_count    [0]),
    .wr_outst_max_o(perf_wr_outst_max[0])
  );

  axi_txn_counters #(
    .axi_req_t  (soc_wide_req_t     ),
    .axi_resp_t (soc_wide_resp_t    ),
    .IdFilterEn (1'b1               ),
    .IdFilterBit(AxiSocIdWidth - 1  )
  ) i_perf_tap_dram_ara (
    .clk_i         (clk_i                      ),
    .rst_ni        (rst_ni                     ),
    .axi_req_i     (periph_wide_axi_req [L2MEM]),
    .axi_resp_i    (periph_wide_axi_resp[L2MEM]),
    .rd_lat_sum_o  (perf_rd_lat_sum  [1]       ),
    .rd_count_o    (perf_rd_count    [1]       ),
    .rd_outst_max_o(perf_rd_outst_max[1]       ),
    .wr_lat_sum_o  (perf_wr_lat_sum  [1]       ),
    .wr_count_o    (perf_wr_count    [1]       ),
    .wr_outst_max_o(perf_wr_outst_max[1]       )
  );

  axi_txn_counters #(
    .axi_req_t (soc_wide_req_t ),
    .axi_resp_t(soc_wide_resp_t),
    .IdFilterEn(1'b0           )
  ) i_perf_tap_dram_all (
    .clk_i         (clk_i                      ),
    .rst_ni        (rst_ni                     ),
    .axi_req_i     (periph_wide_axi_req [L2MEM]),
    .axi_resp_i    (periph_wide_axi_resp[L2MEM]),
    .rd_lat_sum_o  (perf_rd_lat_sum  [2]       ),
    .rd_count_o    (perf_rd_count    [2]       ),
    .rd_outst_max_o(perf_rd_outst_max[2]       ),
    .wr_lat_sum_o  (perf_wr_lat_sum  [2]       ),
    .wr_count_o    (perf_wr_count    [2]       ),
    .wr_outst_max_o(perf_wr_outst_max[2]       )
  );

  logic [63:0] vinsn_active_cycles_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)         vinsn_active_cycles_q <= '0;
    else if (!ara_idle)  vinsn_active_cycles_q <= vinsn_active_cycles_q + 64'd1;
  end

  soc_narrow_lite_req_t  axi_lite_perf_req;
  soc_narrow_lite_resp_t axi_lite_perf_resp;

  axi_to_axi_lite #(
    .AxiAddrWidth   (AxiAddrWidth          ),
    .AxiDataWidth   (AxiNarrowDataWidth    ),
    .AxiIdWidth     (AxiSocIdWidth         ),
    .AxiUserWidth   (AxiUserWidth          ),
    .AxiMaxReadTxns (1                     ),
    .AxiMaxWriteTxns(1                     ),
    .FallThrough    (1'b0                  ),
    .full_req_t     (soc_narrow_req_t      ),
    .full_resp_t    (soc_narrow_resp_t     ),
    .lite_req_t     (soc_narrow_lite_req_t ),
    .lite_resp_t    (soc_narrow_lite_resp_t)
  ) i_axi_to_axi_lite_perf (
    .clk_i     (clk_i                        ),
    .rst_ni    (rst_ni                       ),
    .test_i    (1'b0                         ),
    .slv_req_i (periph_narrow_axi_req[PERF]  ),
    .slv_resp_o(periph_narrow_axi_resp[PERF] ),
    .mst_req_o (axi_lite_perf_req            ),
    .mst_resp_i(axi_lite_perf_resp           )
  );

  perf_registers #(
    .NumTaps        (NumPerfTaps           ),
    .axi_lite_req_t (soc_narrow_lite_req_t ),
    .axi_lite_resp_t(soc_narrow_lite_resp_t),
    .DataWidth      (AxiNarrowDataWidth    )
  ) i_perf_registers (
    .clk_i                (clk_i             ),
    .rst_ni               (rst_ni            ),
    .axi_lite_slave_req_i (axi_lite_perf_req ),
    .axi_lite_slave_resp_o(axi_lite_perf_resp),
    .vinsn_active_cycles_i    (vinsn_active_cycles_q ),
    .rd_lat_sum_i         (perf_rd_lat_sum   ),
    .rd_count_i           (perf_rd_count     ),
    .rd_outst_max_i       (perf_rd_outst_max ),
    .wr_lat_sum_i         (perf_wr_lat_sum   ),
    .wr_count_i           (perf_wr_count     ),
    .wr_outst_max_i       (perf_wr_outst_max )
  );

  axi_dw_converter #(
    .AxiSlvPortDataWidth(AxiWideDataWidth    ),
    .AxiMstPortDataWidth(AxiNarrowDataWidth  ),
    .AxiAddrWidth       (AxiAddrWidth        ),
    .AxiIdWidth         (AxiSocIdWidth       ),
    .AxiMaxReads        (2                   ),
    .ar_chan_t          (soc_wide_ar_chan_t  ),
    .mst_r_chan_t       (soc_narrow_r_chan_t ),
    .slv_r_chan_t       (soc_wide_r_chan_t   ),
    .aw_chan_t          (soc_narrow_aw_chan_t),
    .b_chan_t           (soc_narrow_b_chan_t ),
    .mst_w_chan_t       (soc_narrow_w_chan_t ),
    .slv_w_chan_t       (soc_wide_w_chan_t   ),
    .axi_mst_req_t      (soc_narrow_req_t    ),
    .axi_mst_resp_t     (soc_narrow_resp_t   ),
    .axi_slv_req_t      (soc_wide_req_t      ),
    .axi_slv_resp_t     (soc_wide_resp_t     )
  ) i_axi_slave_perf_dwc (
    .clk_i     (clk_i                       ),
    .rst_ni    (rst_ni                      ),
    .slv_req_i (periph_wide_axi_req[PERF]   ),
    .slv_resp_o(periph_wide_axi_resp[PERF]  ),
    .mst_req_o (periph_narrow_axi_req[PERF] ),
    .mst_resp_i(periph_narrow_axi_resp[PERF])
  );

  logic [2:0] hart_id;

  assign hart_id = '0;

  function automatic config_pkg::cva6_user_cfg_t gen_usr_cva6_config(config_pkg::cva6_user_cfg_t cfg);
    cfg.AxiAddrWidth          = AxiAddrWidth;
    cfg.AxiDataWidth          = AxiNarrowDataWidth;
    cfg.AxiIdWidth            = AxiIdWidth;
    cfg.AxiUserWidth          = AxiUserWidth;
    cfg.XF16                  = FPUSupport[3];
    cfg.RVF                   = FPUSupport[4];
    cfg.RVD                   = Cva6ScalarD;
    cfg.XF16ALT               = FPUSupport[2];
    cfg.XF8                   = FPUSupport[1];

    cfg.NrPMPEntries          = 0;

    cfg.NrNonIdempotentRules  = 2;
    cfg.NonIdempotentAddrBase = {UARTBase, CTRLBase};
    cfg.NonIdempotentLength   = {UARTLength, CTRLLength};
    cfg.NrExecuteRegionRules  = 3;

    cfg.ExecuteRegionAddrBase = {DRAMBase,   64'h1_0000, 64'h0};
    cfg.ExecuteRegionLength   = {DRAMLength, 64'h10000,  64'h1000};

    cfg.NrCachedRegionRules   = 1;
    cfg.CachedRegionAddrBase  = {DRAMBase};
    cfg.CachedRegionLength    = {DRAMLength};

    return cfg;
  endfunction

  localparam config_pkg::cva6_user_cfg_t CVA6AraConfig_user = gen_usr_cva6_config(cva6_config_pkg::cva6_cfg);

  localparam config_pkg::cva6_cfg_t CVA6AraConfig = build_config_pkg::build_config(CVA6AraConfig_user);

  `CVA6_TYPEDEF_EXCEPTION(exception_t, CVA6AraConfig);

  `CVA6_INTF_TYPEDEF_ACC_REQ(accelerator_req_t, CVA6AraConfig, fpnew_pkg::roundmode_e);
  `CVA6_INTF_TYPEDEF_ACC_RESP(accelerator_resp_t, CVA6AraConfig, exception_t);

  `CVA6_INTF_TYPEDEF_MMU_REQ(acc_mmu_req_t, CVA6AraConfig);
  `CVA6_INTF_TYPEDEF_MMU_RESP(acc_mmu_resp_t, CVA6AraConfig, exception_t);

  `CVA6_INTF_TYPEDEF_CVA6_TO_ACC(cva6_to_acc_t, accelerator_req_t, acc_mmu_resp_t);
  `CVA6_INTF_TYPEDEF_ACC_TO_CVA6(acc_to_cva6_t, accelerator_resp_t, acc_mmu_req_t);

`ifndef TARGET_GATESIM
  ara_system #(
    .NrLanes           (NrLanes              ),
    .VLEN              (VLEN                 ),
    .OSSupport         (OSSupport            ),
    .FPUSupport        (FPUSupport           ),
    .FPExtSupport      (FPExtSupport         ),
    .VtransSupport     (VtransSupport        ),
    .FixPtSupport      (FixPtSupport         ),
    .SegSupport        (SegSupport           ),
    .CVA6Cfg           (CVA6AraConfig        ),
    .exception_t       (exception_t          ),
    .accelerator_req_t (accelerator_req_t    ),
    .accelerator_resp_t(accelerator_resp_t   ),
    .acc_mmu_req_t     (acc_mmu_req_t        ),
    .acc_mmu_resp_t    (acc_mmu_resp_t       ),
    .cva6_to_acc_t     (cva6_to_acc_t        ),
    .acc_to_cva6_t     (acc_to_cva6_t        ),
    .AxiAddrWidth      (AxiAddrWidth         ),
    .AxiIdWidth        (AxiCoreIdWidth       ),
    .AxiNarrowDataWidth(AxiNarrowDataWidth   ),
    .AxiWideDataWidth  (AxiDataWidth         ),
    .ara_axi_ar_t      (ara_axi_ar_chan_t    ),
    .ara_axi_aw_t      (ara_axi_aw_chan_t    ),
    .ara_axi_b_t       (ara_axi_b_chan_t     ),
    .ara_axi_r_t       (ara_axi_r_chan_t     ),
    .ara_axi_w_t       (ara_axi_w_chan_t     ),
    .ara_axi_req_t     (ara_axi_req_t        ),
    .ara_axi_resp_t    (ara_axi_resp_t       ),
    .ariane_axi_ar_t   (ariane_axi_ar_chan_t ),
    .ariane_axi_aw_t   (ariane_axi_aw_chan_t ),
    .ariane_axi_b_t    (ariane_axi_b_chan_t  ),
    .ariane_axi_r_t    (ariane_axi_r_chan_t  ),
    .ariane_axi_w_t    (ariane_axi_w_chan_t  ),
    .ariane_axi_req_t  (ariane_axi_req_t     ),
    .ariane_axi_resp_t (ariane_axi_resp_t    ),
    .system_axi_ar_t   (system_ar_chan_t     ),
    .system_axi_aw_t   (system_aw_chan_t     ),
    .system_axi_b_t    (system_b_chan_t      ),
    .system_axi_r_t    (system_r_chan_t      ),
    .system_axi_w_t    (system_w_chan_t      ),
    .system_axi_req_t  (system_req_t         ),
    .system_axi_resp_t (system_resp_t        ),
    .SPMEnable         (1'b1                 ),
    .SPMBase           (SPMBase              ),
    .SPMLength         (SPMLength            ))
`else
  ara_system #(
    .SPMEnable         (1'b1                 ),
    .SPMBase           (SPMBase              ),
    .SPMLength         (SPMLength            ))
`endif
  i_system (
    .clk_i        (clk_i                    ),
    .rst_ni       (rst_ni                   ),
    .boot_addr_i  (DRAMBase                 ),
    .hart_id_i    (hart_id                  ),
    .scan_enable_i(1'b0                     ),
    .scan_data_i  (1'b0                     ),
    .scan_data_o  (        ),
    .spm_axi_req_o (ara_spm_axi_req         ),
    .spm_axi_resp_i(ara_spm_axi_resp        ),
    .ara_idle_o    (ara_idle                ),
`ifndef TARGET_GATESIM
    .axi_req_o    (system_axi_req           ),
    .axi_resp_i   (system_axi_resp          )
  );
`else
    .axi_req_o    (system_axi_req_spill     ),
    .axi_resp_i   (system_axi_resp_spill_del)
  );
`endif

`ifdef TARGET_GATESIM
  assign #(AxiRespDelay*1ps) system_axi_resp_spill_del = system_axi_resp_spill;

  axi_cut #(
    .ar_chan_t   (system_ar_chan_t     ),
    .aw_chan_t   (system_aw_chan_t     ),
    .b_chan_t    (system_b_chan_t      ),
    .r_chan_t    (system_r_chan_t      ),
    .w_chan_t    (system_w_chan_t      ),
    .req_t       (system_req_t         ),
    .resp_t      (system_resp_t        )
  ) i_system_cut (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .slv_req_i   (system_axi_req_spill),
    .slv_resp_o  (system_axi_resp_spill),
    .mst_req_o   (system_axi_req),
    .mst_resp_i  (system_axi_resp)
  );
`endif

  if (NrLanes == 0)
    $error("[ara_soc] Ara needs to have at least one lane.");

  if (AxiDataWidth == 0)
    $error("[ara_soc] The AXI data width must be greater than zero.");

  if (AxiAddrWidth == 0)
    $error("[ara_soc] The AXI address width must be greater than zero.");

  if (AxiUserWidth == 0)
    $error("[ara_soc] The AXI user width must be greater than zero.");

  if (AxiIdWidth == 0)
    $error("[ara_soc] The AXI ID width must be greater than zero.");

  if (RVVD(FPUSupport) && !CVA6AraConfig.RVD)
    $error(
      "[ara] Cannot support double-precision floating-point on Ara if CVA6 does not support it.");

  if (RVVF(FPUSupport) && !CVA6AraConfig.RVF)
    $error(
      "[ara] Cannot support single-precision floating-point on Ara if CVA6 does not support it.");

  if (RVVH(FPUSupport) && !CVA6AraConfig.XF16)
    $error(
      "[ara] Cannot support half-precision floating-point on Ara if CVA6 does not support it.");

  if (RVVHA(FPUSupport) && !CVA6AraConfig.XF16ALT)
    $error(
      "[ara] Cannot support alt-half-precision floating-point on Ara if CVA6 does not support it.");

  if (RVVB(FPUSupport) && !CVA6AraConfig.XF8)
    $error(
      "[ara] Cannot support byte-precision floating-point on Ara if CVA6 does not support it.");

  if (RVVBA(FPUSupport) && !CVA6AraConfig.XF8ALT)
    $error(
      "[ara] Cannot support alt-byte-precision floating-point on Ara if CVA6 does not support it.");

endmodule : ara_soc_vhk158
