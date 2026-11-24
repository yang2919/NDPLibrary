# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set REL_ROOT [file normalize [file dirname [info script]]/..]

if {![info exists HW_HOST]} { set HW_HOST localhost }
if {![info exists PDI]} {
    set cand [glob -nocomplain $REL_ROOT/pdi/*.pdi]
    if {[llength $cand] == 1} {
        set PDI [lindex $cand 0]
    } elseif {[llength $cand] > 1} {
        puts "!! more than one PDI in $REL_ROOT/pdi — set PDI explicitly:"
        foreach p $cand { puts "     $p" }
        return
    } else {
        puts "!! no PDI found in $REL_ROOT/pdi"
        return
    }
}

proc check_ddr {} {
    targets -set -filter {name == "DPC"}
    mwr 0x40000000 0xDEADBEEF
    set rb [mrd -value 0x40000000]

    if {$rb == 0xDEADBEEF} {
        puts [format "==== DDR OK (0x40000000 = 0x%08X)" $rb]
        return 1
    }
    puts [format "!! DDR sanity FAIL: got 0x%08X, expected DEADBEEF" $rb]
    return 0
}

proc reprogram {{pdi ""}} {
    global PDI
    if {$pdi eq ""} { set pdi $PDI }
    catch { targets -set -filter {name == "PMC"}; rst -system }
    after 500
    targets -set -filter {name == "Versal xcvh1582"}
    puts "==== programming $pdi ..."
    device program $pdi
    check_ddr
}

puts "==== board_up: host=$HW_HOST"
puts "==== PDI=$PDI"
if {![file exists $PDI]} { puts "!! PDI missing: $PDI"; return }

connect -url ${HW_HOST}:3121
jtag targets 3
jtag frequency 30000000

catch { targets -set -filter {name == "PMC"}; rst -system }
after 500

targets -set -filter {name == "Versal xcvh1582"}
puts "==== programming PDI (about 25 s at 30 MHz TCK)..."
device program $PDI
catch { plm log }

if {[check_ddr]} {
    catch { source $REL_ROOT/scripts/kernel_helpers.tcl }
    catch { source $REL_ROOT/scripts/soft_reset.tcl }
    catch { source $REL_ROOT/scripts/read_trap.tcl }
    catch { source $REL_ROOT/scripts/gather_bench.tcl }
    catch { source $REL_ROOT/scripts/roofline_sweep.tcl }
    puts ""
    puts "==== READY"
    puts "====   mem_target ; emio_init                          once per power cycle"
    puts "====   load_kernel <name> ; cva6_reset ; read_result   run a kernel"
    puts "====   list_kernels                                    list the kernels"
    puts "====   read_trap                                       decode a recorded exception"
    puts "====   read_gather_bench ; read_gather_perf            memory characterization"
    puts "====   roofline_run fp32 ; roofline_report             performance sweep"
    puts "====   roofline_run spm_fp32                           sweep in DRAM and in the scratchpad"
    puts "====   read_lsu_bench                                  lsu_bench records"
    puts "====   reprogram                                       reset and reprogram the PDI"
}
