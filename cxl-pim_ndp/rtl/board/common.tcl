# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set proj_name ara_vhk158
set proj_root [file normalize [file dirname [info script]]]
set work_dir  $proj_root/build/vivado
set part      xcvh1582-vsva3697-2MP-e-S
set xpr       $work_dir/$proj_name.xpr

set_param general.maxThreads 24

proc require_project {} {
    global xpr work_dir proj_name
    if {![file exists $xpr]} {
        puts "ERROR: project not found at $xpr"
        puts "       Run setup_project.tcl first."
        exit 1
    }
    open_project $xpr
}
