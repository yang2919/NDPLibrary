# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

source [file dirname [info script]]/common.tcl
require_project

if {[get_property STATUS [get_runs synth_1]] ne "Not started"} {
    reset_run synth_1
}
set_property -dict [list \
    STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE      AlternateRoutability \
    {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} {}                   \
] [get_runs synth_1]

launch_runs synth_1 -jobs 8
wait_on_run synth_1
set status [get_property STATUS [get_runs synth_1]]
puts "==== synth_1: $status"
if {$status ne "synth_design Complete!"} {
    puts "ERROR: synthesis failed"
    exit 1
}
puts "==== Next: run_impl.tcl"
