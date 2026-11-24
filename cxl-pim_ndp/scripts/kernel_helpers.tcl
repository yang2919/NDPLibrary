# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set REL_ROOT [file normalize [file dirname [info script]]/..]
set BIN_DIR  $REL_ROOT/sw/bin

proc list_kernels {} {
    global BIN_DIR
    set ns {}
    foreach f [lsort [glob -nocomplain -directory $BIN_DIR *.bin]] {
        lappend ns [file rootname [file tail $f]]
    }
    puts "[llength $ns] images in $BIN_DIR:"
    set line "  "
    foreach n $ns {
        if {[string length "$line$n "] > 78} { puts $line; set line "  " }
        append line "$n "
    }
    if {[string trim $line] ne ""} { puts $line }
}

proc load_kernel {name} {
    global BIN_DIR
    set bin $BIN_DIR/$name.bin
    if {![file exists $bin]} { puts "unknown kernel: $name"; list_kernels; return }
    set words [expr {([file size $bin] + 3) / 4}]
    targets -set -filter {name == "DPC"}
    mwr 0x40100000 0            ;# heartbeat
    mwr 0x40100100 0xdeadbeef   ;# verify magic
    mwr 0x40100300 0            ;# trap record
    mwr -bin -file $bin 0x40000000 $words
    puts "==== loaded $name ($words words, [file size $bin] bytes)"
    puts "==== now: cva6_reset ; after 4000 ; read_result"
}

proc rd32 {addr} {
    set v [mrd -value $addr]
    if {[regexp {[a-fA-F]} $v] && ![string match "0x*" $v]} { set v "0x$v" }
    return [expr {$v & 0xffffffff}]
}

proc read_result {} {
    targets -set -filter {name == "DPC"}
    set hb [rd32 0x40100000]
    set mg [rd32 0x40100100]
    puts [format "heartbeat = 0x%08x     magic = 0x%08x" $hb $mg]
    if {$mg != 0xA1A1B1B1} {
        puts "VERIFY MAGIC NOT SET — the kernel did not finish."
        puts "  heartbeat: CAFE0001 entered main / 0002 uart up / 000F finished"
        puts "  Run read_trap to see whether an exception was recorded."
        return
    }
    set flag   [rd32 0x40100104]
    set cycles [rd32 0x40100108]
    puts [format "%s   cycles = 0x%08x (%d)" \
        [expr {$flag == 1 ? {==== PASS} : {==== FAIL}}] $cycles $cycles]
    puts [format "verify\[3..7\] = 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x" \
        [rd32 0x4010010c] [rd32 0x40100110] [rd32 0x40100114] \
        [rd32 0x40100118] [rd32 0x4010011c]]
}

proc read_lsu_bench {} {
    targets -set -filter {name == "DPC"}
    set base 0x40100200
    set mg [rd32 $base]
    if {$mg != 0x15B00001} { puts [format "lsu_bench magic not set (got 0x%08x): it did not finish" $mg]; return }
    set mask [rd32 [expr {$base + 4 * 14}]]
    puts "record         cycles        ops  cycles/op  check"
    set k 0
    foreach n {chase_l1 chase_dram ld_indep st_indep crc32 alu_ctrl} {
        set c [rd32 [expr {$base + 4 * (2 + $k)}]]
        set o [rd32 [expr {$base + 4 * (8 + $k)}]]
        puts [format "%-10s %10d %10d %10.3f  %s" $n $c $o [expr {$o ? double($c) / $o : 0.0}] \
                     [expr {($mask >> $k) & 1 ? "ok" : "FAILED"}]]
        incr k
    }
}

puts "kernel helper loaded: list_kernels, load_kernel <name>, read_result, read_lsu_bench"
