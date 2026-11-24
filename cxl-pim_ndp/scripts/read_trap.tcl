proc _t32 {a} { return [lindex [mrd -value $a 1] 0] }

proc read_trap {} {
    targets -set -filter {name == "DPC"}
    if {[_t32 0x40100300] != 0x7A9B0001} { puts "no trap recorded"; return }
    set cause [_t32 0x40100304]
    set epc   [_t32 0x4010030c]
    set tval  [_t32 0x40100314]
    array set NM {0 "insn addr misaligned" 1 "insn access fault" 2 "illegal instruction" \
                  3 "breakpoint" 4 "load addr misaligned" 5 "load access fault" \
                  6 "store addr misaligned" 7 "store access fault" 11 "ecall-M"}
    set nm [expr {[info exists NM($cause)] ? $NM($cause) : "cause $cause"}]
    puts [format "TRAP  mcause=%d (%s)  mepc=0x%08X  mtval=0x%08X  mstatus=0x%08X" \
        $cause $nm $epc $tval [_t32 0x4010031c]]
    set vt [_t32 0x40100320]
    puts [format "  vtype=0x%08X (vill=%d sew=e%d lmul=%s)  vl=%d" $vt \
        [expr {($vt >> 31) & 1}] [expr {8 << (($vt >> 3) & 7)}] \
        [lindex {m1 m2 m4 m8 - mf8 mf4 mf2} [expr {$vt & 7}]] [_t32 0x40100324]]
    puts [format "  vstart=%d  vcsr=0x%X" [_t32 0x40100328] [_t32 0x4010032c]]
    puts "  NOTE: vtype/vl above are read back from Ara over the accelerator"
    puts "        interface and are UNRELIABLE after an exception (corner_sp"
    puts "        reported e8/m1/vl=0 where the disassembly proves e32,m2)."
    puts "        mepc and mtval are authoritative; confirm vtype from the code."
    puts [format "  locate with: riscv64-unknown-elf-objdump -d <elf> | grep -B2 -A2 ' %x:'" $epc]
}
puts "read_trap loaded"
