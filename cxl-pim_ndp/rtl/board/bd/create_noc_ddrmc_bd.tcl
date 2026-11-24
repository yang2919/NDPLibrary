# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set_property board_part xilinx.com:vhk158:part0:1.2 [current_project]

create_bd_design design_1
current_bd_design design_1

set CIPS_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:* CIPS_0]
apply_board_connection -board_interface "ps_pmc_fixed_io" \
    -ip_intf "/CIPS_0/FIXED_IO" -diagram design_1

set_property -dict [list                                      \
    CONFIG.CLOCK_MODE              {Custom}                   \
    CONFIG.DDR_MEMORY_MODE         {Custom}                   \
    CONFIG.PS_PL_CONNECTIVITY_MODE {Custom}                   \
    CONFIG.PS_PMC_CONFIG           {                          \
        CLOCK_MODE              Custom                        \
        DDR_MEMORY_MODE         Custom                        \
        PS_PL_CONNECTIVITY_MODE Custom                        \
        PMC_USE_PMC_NOC_AXI0    1                             \
        PMC_GPIO_EMIO_PERIPHERAL_ENABLE 1                     \
        PMC_GPIO_EMIO_WIDTH             4                     \
    }                                                          \
] [get_bd_cells CIPS_0]

foreach p {PMC_GPIO_o PMC_GPIO_t PMC_GPIO_i} {
    make_bd_pins_external -name pmc_gpio_[string tolower [string index $p 9]] \
        [get_bd_pins /CIPS_0/$p]
}

set cips_noc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 cips_noc]
set_property -dict [list      \
    CONFIG.NUM_SI   {1}       \
    CONFIG.NUM_MI   {0}       \
    CONFIG.NUM_NSI  {0}       \
    CONFIG.NUM_NMI  {1}       \
    CONFIG.NUM_CLKS {1}       \
] $cips_noc

set_property -dict [list \
    CONFIG.CONNECTIONS {M00_INI {read_bw {500} write_bw {500}}} \
    CONFIG.CATEGORY    {ps_pmc}                                  \
] [get_bd_intf_pins /cips_noc/S00_AXI]

set_property CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
    [get_bd_pins /cips_noc/aclk0]

set axi_noc_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 axi_noc_0]

apply_board_connection -board_interface "ddr4_dimm0" \
    -ip_intf "/axi_noc_0/CH0_DDR4_0" -diagram design_1
apply_board_connection -board_interface "ddr4_dimm0_sma_clk" \
    -ip_intf "/axi_noc_0/sys_clk0" -diagram design_1

set_property -dict [list                                                   \
    CONFIG.DATA_WIDTH                    {64}                              \
    CONFIG.ID_WIDTH                      {5}                               \
    CONFIG.CONNECTIONS {MC_0 {read_bw {6400} write_bw {6400} \
                              read_avg_burst {8} write_avg_burst {8}}}     \
    CONFIG.CATEGORY    {pl}                                                \
] [get_bd_intf_pins /axi_noc_0/S00_AXI]

set_property CONFIG.NUM_NSI {1} [get_bd_cells axi_noc_0]
set_property -dict [list \
    CONFIG.CONNECTIONS {MC_0 {read_bw {500} write_bw {500} \
                              read_avg_burst {4} write_avg_burst {4}}} \
] [get_bd_intf_pins /axi_noc_0/S00_INI]

connect_bd_intf_net [get_bd_intf_pins /CIPS_0/PMC_NOC_AXI_0] \
                    [get_bd_intf_pins /cips_noc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins /cips_noc/M00_INI] \
                    [get_bd_intf_pins /axi_noc_0/S00_INI]

set pl_clk_port [create_bd_port -dir I -type clk -freq_hz 120000000 pl_clk]
foreach clk_pin [get_bd_pins -of_objects $axi_noc_0 -filter {NAME =~ "aclk*"}] {
    connect_bd_net $pl_clk_port $clk_pin
}
connect_bd_net [get_bd_pins /CIPS_0/pmc_axi_noc_axi0_clk] \
               [get_bd_pins /cips_noc/aclk0]

make_bd_intf_pins_external -name pl_axi_in \
    [get_bd_intf_pins /axi_noc_0/S00_AXI]

assign_bd_address

puts ""
puts "==== Address assignments ===="
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces /pl_axi_in]] {
    set off [get_property OFFSET $seg]
    set rng [get_property RANGE  $seg]
    puts [format "  %-60s offset=%s range=%s" $seg $off $rng]
}

validate_bd_design
save_bd_design
regenerate_bd_layout

generate_target {synthesis simulation implementation} \
    [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top -import

puts ""
puts "==== BD external ports (fpga_top must match these exactly) ===="
foreach pt [lsort [get_bd_ports]] {
    puts [format "  %-34s dir=%-3s %s" $pt [get_property DIR $pt] \
          [expr {[get_property LEFT $pt] ne "" ? "\[[get_property LEFT $pt]:[get_property RIGHT $pt]\]" : ""}]]
}
foreach pt [lsort [get_bd_intf_ports]] { puts "  $pt  (intf [get_property VLNV $pt])" }
