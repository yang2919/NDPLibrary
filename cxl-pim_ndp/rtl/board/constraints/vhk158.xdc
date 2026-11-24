# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set_property PACKAGE_PIN BT10          [get_ports sys_clk_p]
set_property PACKAGE_PIN BU10          [get_ports sys_clk_n]
set_property IOSTANDARD  LVDS15        [get_ports {sys_clk_p sys_clk_n}]

set_property PACKAGE_PIN BT29          [get_ports cpu_resetn]
set_property IOSTANDARD  LVCMOS15      [get_ports cpu_resetn]

set_property PACKAGE_PIN BG28          [get_ports uart_tx_o]
set_property PACKAGE_PIN BG29          [get_ports uart_rx_i]
set_property IOSTANDARD  LVCMOS15      [get_ports {uart_rx_i uart_tx_o}]

set_property PACKAGE_PIN BK26          [get_ports {gpio_led[0]}]
set_property PACKAGE_PIN BK25          [get_ports {gpio_led[1]}]
set_property PACKAGE_PIN BJ27          [get_ports {gpio_led[2]}]
set_property PACKAGE_PIN BJ26          [get_ports {gpio_led[3]}]
set_property IOSTANDARD  LVCMOS15      [get_ports {gpio_led[*]}]

create_pblock pblock_compute_slr0
resize_pblock  pblock_compute_slr0 -add SLR0
add_cells_to_pblock pblock_compute_slr0 \
    [get_cells [list u_soc/i_system/i_ariane \
                     u_soc/i_system/i_ara]]
