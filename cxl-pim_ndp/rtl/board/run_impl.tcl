# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

source [file dirname [info script]]/common.tcl
require_project

if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    puts "ERROR: synth_1 is not complete. Run run_synth.tcl first."
    exit 1
}
if {[get_property STATUS [get_runs impl_1]] ne "Not started"} {
    reset_run impl_1
}
set_property -dict [list \
    STEPS.OPT_DESIGN.ARGS.DIRECTIVE                 Explore            \
    STEPS.PLACE_DESIGN.ARGS.DIRECTIVE               AggressiveExplore  \
    STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE            AggressiveExplore  \
    STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE               NoTimingRelaxation \
    STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED     true               \
    STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore  \
] [get_runs impl_1]
set_property RQS_FILES [list $proj_root/constraints/qor_2026-09-10.rqs] [get_runs impl_1]

launch_runs impl_1 -to_step write_device_image -jobs 8
wait_on_run impl_1
set status [get_property STATUS [get_runs impl_1]]
puts "==== impl_1: $status"
if {$status ne "write_device_image Complete!"} {
    puts "ERROR: implementation or device image failed"
    exit 1
}
open_run impl_1
report_timing_summary -max_paths 10 -file $proj_root/build/timing_summary.rpt
puts "==== WNS [get_property STATS.WNS [get_runs impl_1]] ns, TNS [get_property STATS.TNS [get_runs impl_1]] ns, WHS [get_property STATS.WHS [get_runs impl_1]] ns"
puts "==== PDI: $work_dir/$proj_name.runs/impl_1/fpga_top_vhk158.pdi"
