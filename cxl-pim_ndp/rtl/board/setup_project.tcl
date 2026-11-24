# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

source [file dirname [info script]]/common.tcl

file mkdir $work_dir
create_project -force $proj_name $work_dir -part $part
set_property board_part xilinx.com:vhk158:part0:1.2 [current_project]

set_property ip_repo_paths {} [current_project]
config_ip_cache -import_from_project -use_cache_location $work_dir/ipcache

if {![file exists $proj_root/add_sources.tcl]} {
    puts "ERROR: add_sources.tcl not found."
    exit 1
}
source $proj_root/add_sources.tcl

foreach f [lsort [glob -nocomplain $proj_root/*.sv]] {
    add_files -norecurse $f
}

set apb_uart_src $proj_root/../ara/hardware/deps/cva6/corev_apu/fpga/src/apb_uart/src
foreach f {
    slib_clock_div.sv slib_counter.sv slib_edge_detect.sv slib_fifo.sv
    slib_input_filter.sv slib_input_sync.sv slib_mv_filter.sv
    uart_baudgen.sv uart_interrupt.sv uart_receiver.sv uart_transmitter.sv
    apb_uart.sv
} {
    add_files -norecurse $apb_uart_src/$f
}

if {[llength [get_ips -quiet clk_wiz_vhk158]] == 0} {
    create_ip                                            \
        -name    clk_wizard                              \
        -vendor  xilinx.com                              \
        -library ip                                      \
        -version 1.0                                     \
        -module_name clk_wiz_vhk158
}

set_property -dict [list                                                  \
    CONFIG.PRIM_SOURCE                {Differential_clock_capable_pin}    \
    CONFIG.PRIM_IN_FREQ               {200.000}                           \
    CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {120.000}                       \
    CONFIG.USE_RESET                  {true}                              \
    CONFIG.RESET_TYPE                 {ACTIVE_LOW}                        \
    CONFIG.RESET_PORT                 {resetn}                            \
    CONFIG.USE_LOCKED                 {true}                              \
] [get_ips clk_wiz_vhk158]
generate_target {synthesis instantiation_template} \
    [get_files [get_property IP_FILE [get_ips clk_wiz_vhk158]]]

set_property verilog_define { ARA NR_LANES=2 VLEN=2048 } [current_fileset]

source $proj_root/bd/create_noc_ddrmc_bd.tcl

add_files -fileset constrs_1 -norecurse $proj_root/constraints/vhk158.xdc

set_property top fpga_top_vhk158 [current_fileset]
update_compile_order -fileset sources_1

puts ""
puts "==== Project ready: $xpr"
puts "==== Next: run_synth.tcl, then run_impl.tcl"
