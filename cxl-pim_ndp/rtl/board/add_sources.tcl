set ROOT [file normalize [file dirname [info script]]/../ara]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/tech_cells_generic/src/fpga/pad_functional_xilinx.sv \
    $ROOT/hardware/deps/tech_cells_generic/src/fpga/tc_clk_xilinx.sv \
    $ROOT/hardware/deps/tech_cells_generic/src/fpga/tc_sram_xilinx.sv \
    $ROOT/hardware/deps/tech_cells_generic/src/rtl/tc_sram_impl.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/tech_cells_generic/src/deprecated/pulp_clock_gating_async.sv \
    $ROOT/hardware/deps/tech_cells_generic/src/deprecated/cluster_clk_cells.sv \
    $ROOT/hardware/deps/tech_cells_generic/src/deprecated/pulp_clk_cells.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/common_cells/src/binary_to_gray.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/common_cells/src/cb_filter_pkg.sv \
    $ROOT/hardware/deps/common_cells/src/cc_onehot.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_reset_ctrlr_pkg.sv \
    $ROOT/hardware/deps/common_cells/src/cf_math_pkg.sv \
    $ROOT/hardware/deps/common_cells/src/clk_int_div.sv \
    $ROOT/hardware/deps/common_cells/src/credit_counter.sv \
    $ROOT/hardware/deps/common_cells/src/delta_counter.sv \
    $ROOT/hardware/deps/common_cells/src/ecc_pkg.sv \
    $ROOT/hardware/deps/common_cells/src/edge_propagator_tx.sv \
    $ROOT/hardware/deps/common_cells/src/exp_backoff.sv \
    $ROOT/hardware/deps/common_cells/src/fifo_v3.sv \
    $ROOT/hardware/deps/common_cells/src/gray_to_binary.sv \
    $ROOT/hardware/deps/common_cells/src/isochronous_4phase_handshake.sv \
    $ROOT/hardware/deps/common_cells/src/isochronous_spill_register.sv \
    $ROOT/hardware/deps/common_cells/src/lfsr.sv \
    $ROOT/hardware/deps/common_cells/src/lfsr_16bit.sv \
    $ROOT/hardware/deps/common_cells/src/lfsr_8bit.sv \
    $ROOT/hardware/deps/common_cells/src/lossy_valid_to_stream.sv \
    $ROOT/hardware/deps/common_cells/src/mv_filter.sv \
    $ROOT/hardware/deps/common_cells/src/onehot_to_bin.sv \
    $ROOT/hardware/deps/common_cells/src/plru_tree.sv \
    $ROOT/hardware/deps/common_cells/src/passthrough_stream_fifo.sv \
    $ROOT/hardware/deps/common_cells/src/popcount.sv \
    $ROOT/hardware/deps/common_cells/src/rr_arb_tree.sv \
    $ROOT/hardware/deps/common_cells/src/rstgen_bypass.sv \
    $ROOT/hardware/deps/common_cells/src/serial_deglitch.sv \
    $ROOT/hardware/deps/common_cells/src/shift_reg.sv \
    $ROOT/hardware/deps/common_cells/src/shift_reg_gated.sv \
    $ROOT/hardware/deps/common_cells/src/spill_register_flushable.sv \
    $ROOT/hardware/deps/common_cells/src/stream_demux.sv \
    $ROOT/hardware/deps/common_cells/src/stream_filter.sv \
    $ROOT/hardware/deps/common_cells/src/stream_fork.sv \
    $ROOT/hardware/deps/common_cells/src/stream_intf.sv \
    $ROOT/hardware/deps/common_cells/src/stream_join_dynamic.sv \
    $ROOT/hardware/deps/common_cells/src/stream_mux.sv \
    $ROOT/hardware/deps/common_cells/src/stream_throttle.sv \
    $ROOT/hardware/deps/common_cells/src/sub_per_hash.sv \
    $ROOT/hardware/deps/common_cells/src/sync.sv \
    $ROOT/hardware/deps/common_cells/src/sync_wedge.sv \
    $ROOT/hardware/deps/common_cells/src/unread.sv \
    $ROOT/hardware/deps/common_cells/src/read.sv \
    $ROOT/hardware/deps/common_cells/src/addr_decode_dync.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_2phase.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_4phase.sv \
    $ROOT/hardware/deps/common_cells/src/clk_int_div_static.sv \
    $ROOT/hardware/deps/common_cells/src/addr_decode.sv \
    $ROOT/hardware/deps/common_cells/src/addr_decode_napot.sv \
    $ROOT/hardware/deps/common_cells/src/multiaddr_decode.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/common_cells/src/cb_filter.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_fifo_2phase.sv \
    $ROOT/hardware/deps/common_cells/src/clk_mux_glitch_free.sv \
    $ROOT/hardware/deps/common_cells/src/counter.sv \
    $ROOT/hardware/deps/common_cells/src/ecc_decode.sv \
    $ROOT/hardware/deps/common_cells/src/ecc_encode.sv \
    $ROOT/hardware/deps/common_cells/src/edge_detect.sv \
    $ROOT/hardware/deps/common_cells/src/lzc.sv \
    $ROOT/hardware/deps/common_cells/src/max_counter.sv \
    $ROOT/hardware/deps/common_cells/src/rstgen.sv \
    $ROOT/hardware/deps/common_cells/src/spill_register.sv \
    $ROOT/hardware/deps/common_cells/src/stream_delay.sv \
    $ROOT/hardware/deps/common_cells/src/stream_fifo.sv \
    $ROOT/hardware/deps/common_cells/src/stream_fork_dynamic.sv \
    $ROOT/hardware/deps/common_cells/src/stream_join.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_reset_ctrlr.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_fifo_gray.sv \
    $ROOT/hardware/deps/common_cells/src/fall_through_register.sv \
    $ROOT/hardware/deps/common_cells/src/id_queue.sv \
    $ROOT/hardware/deps/common_cells/src/stream_to_mem.sv \
    $ROOT/hardware/deps/common_cells/src/stream_arbiter_flushable.sv \
    $ROOT/hardware/deps/common_cells/src/stream_fifo_optimal_wrap.sv \
    $ROOT/hardware/deps/common_cells/src/stream_register.sv \
    $ROOT/hardware/deps/common_cells/src/stream_xbar.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_fifo_gray_clearable.sv \
    $ROOT/hardware/deps/common_cells/src/cdc_2phase_clearable.sv \
    $ROOT/hardware/deps/common_cells/src/mem_to_banks_detailed.sv \
    $ROOT/hardware/deps/common_cells/src/stream_arbiter.sv \
    $ROOT/hardware/deps/common_cells/src/stream_omega_net.sv \
    $ROOT/hardware/deps/common_cells/src/mem_to_banks.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/common_cells/src/deprecated/clock_divider_counter.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/clk_div.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/find_first_one.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/generic_LFSR_8bit.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/generic_fifo.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/prioarbiter.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/pulp_sync.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/pulp_sync_wedge.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/rrarbiter.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/clock_divider.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/fifo_v2.sv \
    $ROOT/hardware/deps/common_cells/src/deprecated/fifo_v1.sv \
    $ROOT/hardware/deps/common_cells/src/edge_propagator_ack.sv \
    $ROOT/hardware/deps/common_cells/src/edge_propagator.sv \
    $ROOT/hardware/deps/common_cells/src/edge_propagator_rx.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/control_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv \
    $ROOT/hardware/deps/fpu_div_sqrt_mvp/hdl/div_sqrt_mvp_wrapper.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/axi/src/axi_pkg.sv \
    $ROOT/hardware/deps/axi/src/axi_intf.sv \
    $ROOT/hardware/deps/axi/src/axi_atop_filter.sv \
    $ROOT/hardware/deps/axi/src/axi_burst_splitter.sv \
    $ROOT/hardware/deps/axi/src/axi_bus_compare.sv \
    $ROOT/hardware/deps/axi/src/axi_cdc_dst.sv \
    $ROOT/hardware/deps/axi/src/axi_cdc_src.sv \
    $ROOT/hardware/deps/axi/src/axi_cut.sv \
    $ROOT/hardware/deps/axi/src/axi_delayer.sv \
    $ROOT/hardware/deps/axi/src/axi_demux_simple.sv \
    $ROOT/hardware/deps/axi/src/axi_dw_downsizer.sv \
    $ROOT/hardware/deps/axi/src/axi_dw_upsizer.sv \
    $ROOT/hardware/deps/axi/src/axi_fifo.sv \
    $ROOT/hardware/deps/axi/src/axi_id_remap.sv \
    $ROOT/hardware/deps/axi/src/axi_id_prepend.sv \
    $ROOT/hardware/deps/axi/src/axi_isolate.sv \
    $ROOT/hardware/deps/axi/src/axi_join.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_demux.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_dw_converter.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_from_mem.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_join.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_lfsr.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_mailbox.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_mux.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_regs.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_to_apb.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_to_axi.sv \
    $ROOT/hardware/deps/axi/src/axi_modify_address.sv \
    $ROOT/hardware/deps/axi/src/axi_mux.sv \
    $ROOT/hardware/deps/axi/src/axi_rw_join.sv \
    $ROOT/hardware/deps/axi/src/axi_rw_split.sv \
    $ROOT/hardware/deps/axi/src/axi_serializer.sv \
    $ROOT/hardware/deps/axi/src/axi_slave_compare.sv \
    $ROOT/hardware/deps/axi/src/axi_throttle.sv \
    $ROOT/hardware/deps/axi/src/axi_to_detailed_mem.sv \
    $ROOT/hardware/deps/axi/src/axi_cdc.sv \
    $ROOT/hardware/deps/axi/src/axi_demux.sv \
    $ROOT/hardware/deps/axi/src/axi_err_slv.sv \
    $ROOT/hardware/deps/axi/src/axi_dw_converter.sv \
    $ROOT/hardware/deps/axi/src/axi_from_mem.sv \
    $ROOT/hardware/deps/axi/src/axi_id_serialize.sv \
    $ROOT/hardware/deps/axi/src/axi_lfsr.sv \
    $ROOT/hardware/deps/axi/src/axi_multicut.sv \
    $ROOT/hardware/deps/axi/src/axi_to_axi_lite.sv \
    $ROOT/hardware/deps/axi/src/axi_to_mem.sv \
    $ROOT/hardware/deps/axi/src/axi_zero_mem.sv \
    $ROOT/hardware/deps/axi/src/axi_interleaved_xbar.sv \
    $ROOT/hardware/deps/axi/src/axi_iw_converter.sv \
    $ROOT/hardware/deps/axi/src/axi_lite_xbar.sv \
    $ROOT/hardware/deps/axi/src/axi_xbar_unmuxed.sv \
    $ROOT/hardware/deps/axi/src/axi_to_mem_banked.sv \
    $ROOT/hardware/deps/axi/src/axi_to_mem_interleaved.sv \
    $ROOT/hardware/deps/axi/src/axi_to_mem_split.sv \
    $ROOT/hardware/deps/axi/src/axi_xbar.sv \
    $ROOT/hardware/deps/axi/src/axi_xp.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/fpnew/src/fpnew_pkg.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_cast_multi.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_classifier.sv \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v \
    $ROOT/hardware/deps/fpnew/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v \
    $ROOT/hardware/deps/fpnew/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v \
    $ROOT/hardware/deps/fpnew/src/fpnew_divsqrt_th_32.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_divsqrt_th_64_multi.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_divsqrt_multi.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_fma.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_fma_multi.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_sdotp_multi.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_sdotp_multi_wrapper.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_noncomp.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_opgroup_block.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_opgroup_fmt_slice.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_opgroup_multifmt_slice.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_rounding.sv \
    $ROOT/hardware/deps/fpnew/src/lfsr_sr.sv \
    $ROOT/hardware/deps/fpnew/src/fpnew_top.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/apb/src/apb_pkg.sv \
    $ROOT/hardware/deps/apb/src/apb_intf.sv \
    $ROOT/hardware/deps/apb/src/apb_err_slv.sv \
    $ROOT/hardware/deps/apb/src/apb_regs.sv \
    $ROOT/hardware/deps/apb/src/apb_cdc.sv \
    $ROOT/hardware/deps/apb/src/apb_demux.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/cva6/core/include/config_pkg.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/cva6/core/include/cv64a6_imafdcv_sv39_config_pkg.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/cva6/core/include/riscv_pkg.sv \
    $ROOT/hardware/deps/cva6/core/include/ariane_pkg.sv \
    $ROOT/hardware/deps/cva6/core/include/build_config_pkg.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/cva6/core/cva6_mmu/cva6_tlb.sv \
    $ROOT/hardware/deps/cva6/core/cva6_mmu/cva6_shared_tlb.sv \
    $ROOT/hardware/deps/cva6/core/cva6_mmu/cva6_mmu.sv \
    $ROOT/hardware/deps/cva6/core/cva6_mmu/cva6_ptw.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/deps/cva6/core/include/wt_cache_pkg.sv \
    $ROOT/hardware/deps/cva6/core/include/std_cache_pkg.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_example/include/cvxif_instr_pkg.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_fu.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_issue_register_commit_if_driver.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_compressed_if_driver.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_example/cvxif_example_coprocessor.sv \
    $ROOT/hardware/deps/cva6/core/cvxif_example/instr_decoder.sv \
    $ROOT/hardware/deps/cva6/core/cva6_rvfi_probes.sv \
    $ROOT/hardware/deps/cva6/core/cva6_fifo_v3.sv \
    $ROOT/hardware/deps/cva6/core/cva6.sv \
    $ROOT/hardware/deps/cva6/core/alu.sv \
    $ROOT/hardware/deps/cva6/core/fpu_wrap.sv \
    $ROOT/hardware/deps/cva6/core/branch_unit.sv \
    $ROOT/hardware/deps/cva6/core/compressed_decoder.sv \
    $ROOT/hardware/deps/cva6/core/controller.sv \
    $ROOT/hardware/deps/cva6/core/csr_buffer.sv \
    $ROOT/hardware/deps/cva6/core/csr_regfile.sv \
    $ROOT/hardware/deps/cva6/core/decoder.sv \
    $ROOT/hardware/deps/cva6/core/ex_stage.sv \
    $ROOT/hardware/deps/cva6/core/acc_dispatcher.sv \
    $ROOT/hardware/deps/cva6/core/instr_realign.sv \
    $ROOT/hardware/deps/cva6/core/macro_decoder.sv \
    $ROOT/hardware/deps/cva6/core/id_stage.sv \
    $ROOT/hardware/deps/cva6/core/issue_read_operands.sv \
    $ROOT/hardware/deps/cva6/core/issue_stage.sv \
    $ROOT/hardware/deps/cva6/core/load_unit.sv \
    $ROOT/hardware/deps/cva6/core/load_store_unit.sv \
    $ROOT/hardware/deps/cva6/core/lsu_bypass.sv \
    $ROOT/hardware/deps/cva6/core/mult.sv \
    $ROOT/hardware/deps/cva6/core/multiplier.sv \
    $ROOT/hardware/deps/cva6/core/serdiv.sv \
    $ROOT/hardware/deps/cva6/core/perf_counters.sv \
    $ROOT/hardware/deps/cva6/core/ariane_regfile_ff.sv \
    $ROOT/hardware/deps/cva6/core/ariane_regfile_fpga.sv \
    $ROOT/hardware/deps/cva6/core/scoreboard.sv \
    $ROOT/hardware/deps/cva6/core/store_buffer.sv \
    $ROOT/hardware/deps/cva6/core/amo_buffer.sv \
    $ROOT/hardware/deps/cva6/core/store_unit.sv \
    $ROOT/hardware/deps/cva6/core/commit_stage.sv \
    $ROOT/hardware/deps/cva6/core/axi_shim.sv \
    $ROOT/hardware/deps/cva6/core/frontend/btb.sv \
    $ROOT/hardware/deps/cva6/core/frontend/bht.sv \
    $ROOT/hardware/deps/cva6/core/frontend/ras.sv \
    $ROOT/hardware/deps/cva6/core/frontend/instr_scan.sv \
    $ROOT/hardware/deps/cva6/core/frontend/instr_queue.sv \
    $ROOT/hardware/deps/cva6/core/frontend/frontend.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_dcache_ctrl.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_dcache_mem.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_dcache_missunit.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_dcache_wbuffer.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_dcache.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_cache_subsystem.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/wt_axi_adapter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_icache.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/tag_cmp.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cache_ctrl.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/amo_alu.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/axi_adapter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/miss_handler.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/std_nbdcache.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_icache_axi_wrapper.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/std_cache_subsystem.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_pkg.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_resp_demux.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_to_axi_read.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_to_axi_write.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_req_read_arbiter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/utils/hpdcache_mem_req_write_arbiter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_demux.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_lfsr.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sync_buffer.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fifo_reg.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fifo_reg_initialized.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_fxarb.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_rrarb.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_mux.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_decoder.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_1hot_to_binary.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_prio_1hot_encoder.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram_wbyteenable.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_sram_wmask.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_regbank_wbyteenable_1rw.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_regbank_wmask_1rw.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_downsize.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_upsize.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/common/hpdcache_data_resize.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_pkg.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_arb.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hwpf_stride/hwpf_stride_wrapper.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_amo.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_cmo.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_core_arbiter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_ctrl.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_ctrl_pe.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_memctrl.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_miss_handler.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_mshr.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_rtab.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_uncached.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_plru.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_random.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_victim_sel.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_wbuf.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/src/hpdcache_flush.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_hpdcache_if_adapter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_hpdcache_subsystem_axi_arbiter.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_hpdcache_subsystem.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/cva6_hpdcache_wrapper.sv \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache_tc_sram.sv \
    $ROOT/hardware/deps/cva6/core/pmp/src/pmp.sv \
    $ROOT/hardware/deps/cva6/core/pmp/src/pmp_entry.sv \
    $ROOT/hardware/deps/cva6/core/pmp/src/pmp_data_if.sv \
    $ROOT/hardware/deps/cva6/vendor/pulp-platform/fpga-support/fpga-support-stubs.sv \
    $ROOT/hardware/deps/cva6/common/local/util/tc_sram_wrapper.sv \
    $ROOT/hardware/deps/cva6/common/local/util/tc_sram_wrapper_cache_techno.sv \
    $ROOT/hardware/deps/cva6/common/local/util/sram_pulp.sv \
    $ROOT/hardware/deps/cva6/common/local/util/sram_cache.sv \
]

add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hardware/include/rvv_pkg.sv \
    $ROOT/hardware/include/vtrans_tables_pkg.sv \
    $ROOT/hardware/include/vtrans_pkg.sv \
    $ROOT/hardware/include/ara_pkg.sv \
    $ROOT/hardware/src/segment_sequencer.sv \
    $ROOT/hardware/src/ctrl_registers.sv \
    $ROOT/hardware/src/cva6_accel_first_pass_decoder.sv \
    $ROOT/hardware/src/ara_dispatcher.sv \
    $ROOT/hardware/src/ara_sequencer.sv \
    $ROOT/hardware/src/axi_inval_filter.sv \
    $ROOT/hardware/src/lane/lane_sequencer.sv \
    $ROOT/hardware/src/lane/operand_queue.sv \
    $ROOT/hardware/src/lane/operand_requester.sv \
    $ROOT/hardware/src/lane/simd_alu.sv \
    $ROOT/hardware/src/lane/simd_div.sv \
    $ROOT/hardware/src/lane/simd_mul.sv \
    $ROOT/hardware/src/lane/vector_regfile.sv \
    $ROOT/hardware/src/lane/power_gating_generic.sv \
    $ROOT/hardware/src/masku/masku_operands.sv \
    $ROOT/hardware/src/sldu/p2_stride_gen.sv \
    $ROOT/hardware/src/sldu/sldu_op_dp.sv \
    $ROOT/hardware/src/sldu/sldu.sv \
    $ROOT/hardware/src/vlsu/addrgen.sv \
    $ROOT/hardware/src/vlsu/vldu.sv \
    $ROOT/hardware/src/vlsu/vstu.sv \
    $ROOT/hardware/src/lane/operand_queues_stage.sv \
    $ROOT/hardware/src/lane/valu.sv \
    $ROOT/hardware/src/lane/vtrans_unit.sv \
    $ROOT/hardware/src/lane/vmfpu.sv \
    $ROOT/hardware/src/lane/fixed_p_rounding.sv \
    $ROOT/hardware/src/vlsu/vlsu.sv \
    $ROOT/hardware/src/masku/masku.sv \
    $ROOT/hardware/src/lane/vector_fus_stage.sv \
    $ROOT/hardware/src/lane/lane.sv \
    $ROOT/hardware/src/ara.sv \
    $ROOT/hardware/src/ara_system.sv \
    $ROOT/hardware/src/ara_soc.sv \
]

set_property include_dirs [list \
    $ROOT/hardware/deps/apb/include \
    $ROOT/hardware/deps/axi/include \
    $ROOT/hardware/deps/common_cells/include \
    $ROOT/hardware/deps/cva6/common/local/util \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/include \
    $ROOT/hardware/deps/cva6/core/include \
    $ROOT/hardware/include \
] [current_fileset]

set_property include_dirs [list \
    $ROOT/hardware/deps/apb/include \
    $ROOT/hardware/deps/axi/include \
    $ROOT/hardware/deps/common_cells/include \
    $ROOT/hardware/deps/cva6/common/local/util \
    $ROOT/hardware/deps/cva6/core/cache_subsystem/hpdcache/rtl/include \
    $ROOT/hardware/deps/cva6/core/include \
    $ROOT/hardware/include \
] [current_fileset -simset]

set_property verilog_define [list \
    TARGET_CV64A6_IMAFDCV_SV39 \
    TARGET_CVA6 \
    TARGET_EXCLUDE_FIRST_PASS_DECODER \
    TARGET_FPGA \
    TARGET_RTL \
    TARGET_SYNTHESIS \
    TARGET_VIVADO \
    TARGET_XILINX \
    ARA \
    NR_LANES=8 \
    VLEN=8192 \
] [current_fileset]

set_property verilog_define [list \
    TARGET_CV64A6_IMAFDCV_SV39 \
    TARGET_CVA6 \
    TARGET_EXCLUDE_FIRST_PASS_DECODER \
    TARGET_FPGA \
    TARGET_RTL \
    TARGET_SYNTHESIS \
    TARGET_VIVADO \
    TARGET_XILINX \
    ARA \
    NR_LANES=8 \
    VLEN=8192 \
] [current_fileset -simset]
