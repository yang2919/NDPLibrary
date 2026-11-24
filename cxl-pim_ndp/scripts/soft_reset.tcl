set ::PMC_GPIO_BASE   0xF1020000
set ::PMC_DIRM_3      0xF10202C4
set ::PMC_OEN_3       0xF10202C8
set ::PMC_DATA_3      0xF102004C
set ::PMC_DATARO_3    0xF102006C
set ::PMC_MASKLSW_3   0xF1020018

set ::SOFTRST_ASSERT  0xFFFE0001
set ::SOFTRST_RELEASE 0xFFFE0000

proc _rd {a} { return [lindex [mrd -force -value $a 1] 0] }

proc mem_target {} {
    foreach n {DPC "MicroBlaze PPU" PPU PSM} {
        if {[catch {targets -set -filter "name == \"$n\""}]} { continue }
        if {![catch {_rd $::PMC_DIRM_3}]} { puts "mem target: $n"; return $n }
    }
    puts "no memory-capable target found"
    return ""
}

proc emio_init {} {
    mwr -force $::PMC_DIRM_3 [expr {[_rd $::PMC_DIRM_3] | 1}]
    mwr -force $::PMC_OEN_3  [expr {[_rd $::PMC_OEN_3]  | 1}]
    puts [format "emio_init: DIRM_3=0x%08X OEN_3=0x%08X" \
          [_rd $::PMC_DIRM_3] [_rd $::PMC_OEN_3]]
}

proc cva6_reset {{hold_ms 2}} {
    mwr -force $::PMC_MASKLSW_3 $::SOFTRST_ASSERT
    after $hold_ms
    mwr -force $::PMC_MASKLSW_3 $::SOFTRST_RELEASE
    puts "cva6_reset: pulsed pl_soft_rst (DATA_3 now [format 0x%08X [_rd $::PMC_DATA_3]])"
}

proc cva6_hold  {} { mwr -force $::PMC_MASKLSW_3 $::SOFTRST_ASSERT  ; puts "pl_soft_rst ASSERTED (cva6 held)" }
proc cva6_run   {} { mwr -force $::PMC_MASKLSW_3 $::SOFTRST_RELEASE ; puts "pl_soft_rst released (cva6 running)" }

proc emio_selftest {{n 5}} {
    set bad 0
    for {set i 0} {$i < $n} {incr i} {
        set v [expr {$i % 2}]
        mwr -force $::PMC_MASKLSW_3 [expr {$v ? $::SOFTRST_ASSERT : $::SOFTRST_RELEASE}]
        after 50
        set ro [expr {[_rd $::PMC_DATARO_3] & 1}]
        if {$ro != $v} { incr bad }
        puts [format "  drive=%d  fabric=%d  %s" $v $ro [expr {$ro == $v ? "MATCH" : "*** MISMATCH ***"}]]
    }
    mwr -force $::PMC_MASKLSW_3 $::SOFTRST_RELEASE
    puts "emio_selftest: [expr {$n - $bad}]/$n match[expr {$bad ? "  -- FAIL" : ""}]"
    return [expr {$bad == 0}]
}

proc program_pdi {pdi} {
    targets -set -filter {name =~ "Versal*"}
    catch { rst -system }
    after 3000
    if {[catch {device program $pdi} e]} { puts "PROGRAM FAIL: $e"; return 0 }
    after 2000
    puts "programmed: $pdi"
    mem_target
    return 1
}

puts "soft_reset.tcl loaded. procs: mem_target, emio_init, emio_selftest,"
puts "                             cva6_reset, cva6_hold, cva6_run, program_pdi"
