# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set proj_name ara_vhk158_bdgen
set proj_root [file normalize [file dirname [info script]]]
set work_dir  $proj_root/build/vivado_bdgen
set part      xcvh1582-vsva3697-2MP-e-S

file mkdir $work_dir
create_project -force $proj_name $work_dir -part $part
set_property board_part xilinx.com:vhk158:part0:1.2 [current_project]

source $proj_root/bd/create_noc_ddrmc_bd.tcl

puts ""
puts "==== BD generated. Wrapper file:"
puts "  $work_dir/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v"
puts ""
puts "==== Open in editor to confirm DDR4 + pmc_gpio port names/widths, then update"
puts "==== fpga_top_vhk158.sv accordingly."
