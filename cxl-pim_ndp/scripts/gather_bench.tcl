set ::GB_HB   0x40100000
set ::GB_RES  0x40100200

set ::GB_RECS {
    {spm_vrt  4     1024}
    {spm_full 512   131072}
    {spm_be   16    4096}
    {spm_strm 512   131072}
    {spm_gr   8     16384}
    {spm_gr   64    16384}
    {spm_gr   512   16384}
    {spm_g+ix 512   16384}
    {spm_s64B 512   8192}
    {s64cold 64    16384}
    {s64post 64    16384}
    {stream  64    16384}
    {stream  1024  262144}
    {stream  8192  2097152}
    {gseq    1     65536}
    {grand   8     65536}
    {grand   64    65536}
    {grand   512   65536}
    {grand   4096  65536}
    {grand   32768 65536}
    {g+ix    512   65536}
    {s64B    4096  65536}
    {s4096B  32768 8192}
    {2s_dram 1     16384}
    {2s_spm  1     16384}
    {1s_dram 1     16384}
    {1s_spm  1     16384}
    {2s64_dr 1     16384}
    {2s64_sp 1     16384}
    {2s16_dr 1     16384}
    {2s16_sp 1     16384}
}

proc _gb_word {addr} { return [lindex [mrd -value $addr 1] 0] }

proc read_gather_bench {} {
    targets -set -filter {name == "DPC"}
    set hb [_gb_word $::GB_HB]
    set nrec [llength $::GB_RECS]
    if {$hb != 0xCAFE000F} {
        puts [format "NOT DONE (heartbeat=0x%08X) — printing records completed so far" $hb]
    }
    if {[_gb_word $::GB_RES] != 0x6A7B0001} { puts "no RES magic — harness didn't start"; return }
    set n [_gb_word [expr {$::GB_RES + 4}]]
    set mask [_gb_word [expr {$::GB_RES + 80*4}]]   ;# GB_MASK_SLOT in main.c
    puts "record   spanKiB  elems    cycles      elem/cyc  inferL   check"
    for {set k 0} {$k < $n} {incr k} {
        lassign [lindex $::GB_RECS $k] tag span elems
        set cyc [_gb_word [expr {$::GB_RES + 8 + $k*4}]]
        set ok  [expr {($mask >> $k) & 1}]
        if {$cyc > 0} {
            set rate [expr {double($elems) / $cyc}]
            set infl [expr {$rate < 1.0 ? 4.0 / $rate : 0.0}]
        } else { set rate 0.0; set infl 0.0 }
        puts [format "%-8s %-8d %-8d %-11d %-9.3f %-8.1f %s" \
            $tag $span $elems $cyc $rate $infl [expr {$ok ? "PASS" : "FAIL"}]]
    }
    puts [format "records %d/%d, pass mask = 0x%04X (all-pass = 0x%04X)" \
        $n $nrec $mask [expr {(1 << $nrec) - 1}]]
}

proc _gb_dword {s} {
    set lo [_gb_word [expr {$::GB_RES + $s*4}]]
    set hi [_gb_word [expr {$::GB_RES + ($s+1)*4}]]
    return [expr {($hi << 32) | ($lo & 0xffffffff)}]
}

proc read_gather_perf {} {
    targets -set -filter {name == "DPC"}
    if {[_gb_word $::GB_RES] != 0x6A7B0001} { puts "no RES magic — harness didn't start"; return }
    set mg [_gb_word [expr {$::GB_RES + 150*4}]]
    if {$mg != 0x50455246} {
        puts [format "!! PERF magic hi32 = 0x%08X, expected 0x50455246 ('PERF')" $mg]
        puts "   counters not readable — check the 0xD0010000 slave decode"
        return
    }
    set TAPN {spm dram_ara dram_all}
    puts ""
    puts "window (record delta)    rd_n      lat_sum      meanL   outst_max  cycles"
    foreach {w name} {0 "spm_gr  512KiB" 1 "grand   512KiB"} {
        set s [expr {96 + $w*8}]
        set n   [_gb_dword $s]
        set ls  [_gb_dword [expr {$s+2}]]
        set om  [_gb_word  [expr {$::GB_RES + ($s+4)*4}]]
        set cyc [_gb_dword [expr {$s+5}]]
        set ml  [expr {$n ? double($ls)/$n : 0.0}]
        puts [format "%-22s %-9d %-12d %-7.2f %-10d %d" $name $n $ls $ml $om $cyc]
    }
    puts ""
    puts "tap        rd_n      rd_meanL  rd_omax   wr_n      wr_meanL  wr_omax"
    for {set t 0} {$t < 3} {incr t} {
        set s [expr {112 + $t*12}]
        set rn [_gb_dword $s]
        set rl [_gb_dword [expr {$s+2}]]
        set ro [_gb_word  [expr {$::GB_RES + ($s+4)*4}]]
        set wn [_gb_dword [expr {$s+5}]]
        set wl [_gb_dword [expr {$s+7}]]
        set wo [_gb_word  [expr {$::GB_RES + ($s+9)*4}]]
        puts [format "%-10s %-9d %-9.2f %-9d %-9d %-9.2f %d" [lindex $TAPN $t] \
              $rn [expr {$rn ? double($rl)/$rn : 0.0}] $ro \
              $wn [expr {$wn ? double($wl)/$wn : 0.0}] $wo]
    }
    puts [format "\nvpu_busy = %d cycles" [_gb_dword 148]]
}
puts "gather_bench readout loaded: read_gather_bench ; read_gather_perf"
