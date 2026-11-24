# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

set ::RL_ROOT   [file normalize [file dirname [info script]]/..]
set ::RES_BASE  0x40100200
set ::RES_MAGIC 0x2F00F11E
set ::RES_SPM_BASE  0x40100400
set ::RES_SPM_MAGIC 0x5B00F11E

set ::RL_FREQ_MHZ 120

array set ::PEAK   {sp 8  sw_sp 8  hw_sp 8  hwsw_sp 8   rec7_sp 8  rec7sw_sp 8 \
                    hp 16  sw_hp 16  hw_hp 16  hwsw_hp 16  rec7_hp 16  rec7sw_hp 16   i8 32}
array set ::EBYTES {sp 4  sw_sp 4  hw_sp 4  hwsw_sp 4   rec7_sp 4  rec7sw_sp 4 \
                    hp 2   sw_hp 2   hw_hp 2   hwsw_hp 2   rec7_hp 2   rec7sw_hp 2    i8 1}

set ::SWEEP {}
foreach n {16 32 64 128 256}     { lappend ::SWEEP [list mm flop $n] }
foreach n {256 1024 4096 16384} {
    lappend ::SWEEP [list dp   flop $n]
    lappend ::SWEEP [list red  elem $n]
    lappend ::SWEEP [list exp  elem $n]
    lappend ::SWEEP [list gelu elem $n]
    lappend ::SWEEP [list sfx  elem $n]
    lappend ::SWEEP [list lyn  elem $n]
}
foreach n {64 128}               { lappend ::SWEEP [list attn flop $n] }

set ::SWEEP_I8 {}
foreach n {16 32 64 128 256}     { lappend ::SWEEP_I8 [list mm   flop $n] }
foreach n {16 32 64 128 256}     { lappend ::SWEEP_I8 [list mm16 flop $n] }
foreach n {256 1024 4096 16384} {
    lappend ::SWEEP_I8 [list dp  flop $n]
    lappend ::SWEEP_I8 [list red elem $n]
}

proc _rl_sweep {dt} { if {$dt eq "i8"} { return $::SWEEP_I8 } ; return $::SWEEP }

set ::RL_ROWS {}

proc rd32 {addr} {
    set v [mrd -value $addr]
    if {[regexp {[a-fA-F]} $v] && ![string match "0x*" $v]} { set v "0x$v" }
    return [expr {$v & 0xffffffff}]
}

proc _rl_flop {kern N} {
    switch -- $kern {
        mm   -
        mm16 { return [expr {2.0*$N*$N*$N}] }
        dp   { return [expr {2.0*$N}] }
        attn { return [expr {4.0*$N*$N*64}] }
        default { return 0.0 }
    }
}

set MEM_BPC 8.0

proc _rl_ideal {dt kern} {
    set E     $::EBYTES($dt)

    set FU    [expr {$dt eq "i8" ? "OP/cycle" : "FLOP/cycle"}]
    set lane  [expr {$::PEAK($dt)/2}]                                   ;# elements per op per cycle
    set vt    [string match "hw*" $dt]                                  ;# vtrans FU build
    set fused [expr {[string match "sw_*" $dt] || [string match "hwsw_*" $dt] \
                     || [string match "rec7sw_*" $dt]}]
    set expops [expr {$vt ? 1 : 21}]                                    ;# vfexp vs Cephes polynomial

    set divops [expr {[string match "rec7*" $dt]
                      ? ([string match "*_hp" $dt] ? 4 : 6) : 1}]
    switch -- $kern {
        mm   -
        mm16 -
        attn { return [list compute [expr {double($::PEAK($dt))}] $FU] }
        dp   { set c [expr {double($::PEAK($dt))}]; set m [expr {$::MEM_BPC/(2*$E)*2}]
               return [list [expr {$c < $m ? "compute" : "memory"}] [expr {min($c,$m)}] $FU] }
        red  { set K 1;             set B [expr {1*$E}] }
        exp  { set K $expops;       set B [expr {2*$E}] }
        lyn  { set K 5;             set B [expr {9*$E}] }
        gelu { set K [expr {$expops + ($fused ? 4 : 3) + ($divops - 1)}]
               set B [expr {($fused ? 2 : 9)*$E}] }
        sfx  { set K [expr {$expops + 4}]
               set B [expr {($fused ? 5 : 8)*$E}] }
        default { return [list - 0.0 -] }
    }
    set c [expr {double($lane)/$K}]
    set m [expr {$::MEM_BPC/$B}]
    return [list [expr {$c < $m ? "compute" : "memory"}] [expr {min($c,$m)}] elem/cycle]
}

proc roofline_load {dt} {
    set bin $::RL_ROOT/sw/bin/roofline_$dt.bin
    if {![file exists $bin]} {
        puts "missing bin: $bin  (build: cd sw/src && ./build_all_kernels.sh)"
        return
    }
    set words [expr {([file size $bin] + 3) / 4}]
    targets -set -filter {name == "DPC"}
    mwr 0x40100000 0
    mwr $::RES_BASE 0
    mwr $::RES_SPM_BASE 0
    mwr -bin -file $bin 0x40000000 $words
    puts "==== injected roofline_$dt  ($words words, [file size $bin] bytes)"
    set base [expr {[string match "spm_*" $dt] ? [string range $dt 4 end] : $dt}]
    set rd [expr {$base eq $dt ? "roofline_read $dt" : "roofline_read $base ; roofline_read_spm $base"}]
    puts "==== now: cva6_reset ; watch UART for 'roofline done' ; then $rd"
}

proc roofline_read {dt} {
    targets -set -filter {name == "DPC"}
    set hb [rd32 0x40100000]
    set mg [rd32 $::RES_BASE]
    if {$mg != $::RES_MAGIC} {
        puts [format "RES magic not set (got 0x%08x, hb 0x%08x) — harness did not finish (reset? trap?)" $mg $hb]
        return
    }
    set cnt [rd32 [expr {$::RES_BASE + 4}]]
    set sweep [_rl_sweep $dt]
    set exp [llength $sweep]
    if {$cnt != $exp} { puts "WARNING: board record count $cnt != descriptor $exp — main.c/tcl sweep drift" }
    set peak $::PEAK($dt)

    set kept {}
    foreach r $::RL_ROWS { if {[lindex $r 0] ne $dt} { lappend kept $r } }
    set ::RL_ROWS $kept
    puts ""
    puts [format "==== roofline %s  (%d records)" $dt $cnt]
    puts [format "%-6s %-7s %-11s %-10s %-11s %-9s %-8s %s" \
                 kernel N cycles throughput unit ideal "% of ideal" "ideal is"]
    puts "------ ------- ----------- ---------- ----------- --------- ---------- ----------"
    for {set i 0} {$i < $cnt} {incr i} {
        set cyc [rd32 [expr {$::RES_BASE + 8 + $i*4}]]
        lassign [lindex $sweep $i] kern metric N
        set flop [_rl_flop $kern $N]
        if {$cyc == 0} { set rate 0.0 } elseif {$metric eq "flop"} {
            set rate [expr {$flop/$cyc}]
        } else { set rate [expr {double($N)/$cyc}] }
        lassign [_rl_ideal $dt $kern] bound ideal unit
        set pct [expr {$ideal > 0 ? 100.0*$rate/$ideal : 0.0}]
        puts [format "%-6s %-7d %-11d %-10.3f %-11s %-9.3f %8.1f%%   %s" \
                     $kern $N $cyc $rate $unit $ideal $pct $bound]
        lappend ::RL_ROWS [list $dt $kern $metric $N $flop $cyc $rate $pct]
    }
    set pm [rd32 [expr {$::RES_BASE + 8 + $cnt*4}]]
    puts [format "pass bitmask = 0x%08x (bit i set = record i finite)" $pm]
}

proc roofline_report {{tag ""}} {
    if {![llength $::RL_ROWS]} { puts "no rows read yet — run roofline_load/roofline_read first"; return }
    set date [clock format [clock seconds] -format %Y-%m-%d]
    set suf [expr {$tag eq "" ? "" : "_$tag"}]
    set out $::RL_ROOT/results/results_${date}_roofline_2lane${suf}.md
    set f [open $out w]

    puts $f "# 2-lane roofline baseline ($date, $::RL_FREQ_MHZ MHz board)"
    puts $f ""
    puts $f "Full kernel analysis, measured on the $::RL_FREQ_MHZ MHz board without using the scratchpad."
    puts $f "The 2-lane arithmetic peak (fp32 8, fp16 16 FLOP/cycle = 2 lanes x ELEN/SEW x 2 FLOP) for mm and attn, and for dp the memory limit of the 64-bit port (8 B/cycle, two operands per element: fp32 2, fp16 4 FLOP/cycle)."
    puts $f "Element-wise/reduction kernels (red/exp/gelu/sfx/lyn) report elem/cycle."
    puts $f "For i8 the rate is integer OP/cycle, with an arithmetic peak of 32 OP/cycle."
    puts $f ""
    puts $f "| dtype | kernel | N | cycles | rate/cycle | %peak | elem/cycle |"
    puts $f "|---|---|---|---|---|---|---|"
    foreach row $::RL_ROWS {
        lassign $row dt kern metric N flop cyc rate pct
        if {$metric eq "flop"} {
            puts $f [format "| %s | %s | %d | %d | %.3f | %.1f%% | - |" $dt $kern $N $cyc $rate $pct]
        } else {
            puts $f [format "| %s | %s | %d | %d | - | - | %.3f |" $dt $kern $N $cyc $rate]
        }
    }
    close $f
    puts "wrote $out ([llength $::RL_ROWS] rows)"
}

proc roofline_read_spm {dt} {
    targets -set -filter {name == "DPC"}
    set mg [rd32 $::RES_SPM_BASE]
    if {$mg != $::RES_SPM_MAGIC} {
        puts [format "RES_SPM magic not set (got 0x%08x) -- not a roofline_spm image, or it did not finish" $mg]
        return
    }
    set cnt [rd32 [expr {$::RES_SPM_BASE + 4}]]
    set sweep [_rl_sweep $dt]
    puts ""
    puts [format "==== roofline %s, DRAM vs scratchpad  (%d records)" $dt $cnt]
    puts [format "%-6s %-7s %-10s %-10s %-9s %-10s %-10s %s" \
                 kernel N DRAM SPM "SPM/DRAM" to-SPM to-DRAM bit-exact]
    for {set i 0} {$i < $cnt} {incr i} {
        set b [expr {$::RES_SPM_BASE + 8 + $i * 28}]
        set v {}
        for {set k 0} {$k < 7} {incr k} { lappend v [rd32 [expr {$b + $k * 4}]] }
        lassign $v ctrl ctrl_sync dram spm stage copy eq
        lassign [lindex $sweep $i] kern metric N
        set r [expr {$dram ? double($spm) / $dram : 0.0}]
        puts [format "%-6s %-7d %-10d %-10d %-9.3f %-10d %-10d %s" \
                     $kern $N $dram $spm $r $stage $copy [expr {$eq == 1 ? "yes" : "NO"}]]
    }
}

proc roofline_run {dt} {
    set spm [string match "spm_*" $dt]
    set base [expr {$spm ? [string range $dt 4 end] : $dt}]
    set d [expr {$base eq "fp32" ? "sp" : ($base eq "fp16" ? "hp" : $base)}]
    roofline_load [expr {$spm ? "spm_$d" : $d}]
    cva6_reset
    for {set t 0} {$t < 60} {incr t} {
        after 2000
        targets -set -filter {name == "DPC"}
        if {[rd32 0x40100000] == 0xCAFE000F} break
    }
    roofline_read $d
    if {$spm} { roofline_read_spm $d }
}

puts "roofline sweep loaded.  roofline_run fp32 | fp16 | spm_fp32 | spm_fp16 ; roofline_report"
