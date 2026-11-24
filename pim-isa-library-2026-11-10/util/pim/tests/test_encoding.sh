#!/bin/sh
# L2: the machine words of pim_isa.h against P2 section 1.2.
#
# Usage: test_encoding.sh OBJDUMP encoding_c.o encoding_s.o
#
# - encoding_s.o (the assembler macros, P2's registers: scalar rs2 = a2,
#   imm = 0, rs1 = a1; vector v8, rs1 = a1, e64) must hold exactly the
#   words of P2 section 1.2, plus those of section 1.5.
# - encoding_c.o (the C wrappers, address in a1, count or descriptor in
#   a2) must hold the same words with the C wrappers' width (e8) and vs3
#   (v0 for the forms without data, v8 for the memory forms, a group of
#   eight for the vector-typed forms), each vector word preceded by its
#   own vsetvli at e8, m8 in the same function. cbo.flush (a0) and
#   fence rw,rw must be 0x0025200f and 0x0330000f (P-LIB section 12).

set -u
OBJDUMP=$1
COBJ=$2
SOBJ=$3

# P2 section 1.2: name word.
TABLE="pim_sb_ewadd 0x00c5802b
pim_sb_ewmul 0x00c5902b
pim_sb_ewadd_gbuf 0x00c5a02b
pim_sb_ewmul_gbuf 0x00c5b02b
pim_sb_mac 0x00c5c02b
pim_sb_mov_bkgbuf 0x0205f45b
pim_sb_mov_gbufbk 0x0215f45b
pim_ab_ewadd 0x0225f45b
pim_ab_ewmul 0x0235f45b
pim_ab_ewadd_gbuf 0x0245f45b
pim_ab_ewmul_gbuf 0x0255f45b
pim_ab_mac 0x0265f45b
pim_ab_mov_regbk 0x0275f45b
pim_ab_mov_bkreg 0x0285f45b
pim_rd_gb 0x0205f40b
pim_wr_gb 0x0295f45b
pim_rd_reg 0x0215f40b
pim_wr_reg 0x02a5f45b"

# "function word mnemonic" for every instruction of an object.
dump() {
    "$OBJDUMP" -d "$1" | awk '
        /^[0-9a-f]+ <.*>:$/ { f = $2; gsub(/[<>:]/, "", f); next }
        /^ +[0-9a-f]+:\t/ {
            split($0, p, "\t");
            w = p[2]; gsub(/ /, "", w);
            print f, w, p[3] "_" p[4] }'
}

fail=0
checks=0
check() { # description expected actual
    checks=$((checks + 1))
    if [ "$2" != "$3" ]; then
        echo "FAIL $1: expected $2, got $3"
        fail=$((fail + 1))
    fi
}

SDUMP=$(dump "$SOBJ")
CDUMP=$(dump "$COBJ")

# The PIM word of one function (major opcode 0x0b, 0x2b or 0x5b).
pimword() { # dump function
    printf '%s\n' "$1" | awk -v f="$2" '
        $1 == f && length($2) == 8 {
            op = substr($2, 7, 2);
            if (op == "0b" || op == "2b" || op == "5b") print "0x" $2 }'
}

OLDIFS=$IFS
IFS='
'
for line in $TABLE; do
    IFS=$OLDIFS
    set -- $line
    name=$1
    word=$2
    # Assembler macros: exactly P2's word.
    check "macro $name" "$word" "$(pimword "$SDUMP" "s_$name")"
    # C wrappers.
    w=$((word))
    case $name in
      pim_sb_*ewadd*|pim_sb_*ewmul*|pim_sb_mac)
        expect=$word ; cfun=t_$name ;;
      pim_wr_gb|pim_wr_reg|pim_rd_gb|pim_rd_reg)
        # The memory forms use v8; the vector-typed forms any group.
        expect=$(printf '0x%08x' $(( (w & ~0x7f80) | (8 << 7) )))
        check "C ${name}_mem" "$expect" "$(pimword "$CDUMP" "t_${name}_mem")"
        got=$(pimword "$CDUMP" "t_$name")
        vs3=$(( (got >> 7) & 31 ))
        check "C $name vs3 a group of 8" 0 $((vs3 % 8))
        expect=$(printf '0x%08x' $(( (w & ~0x7f80) | (vs3 << 7) )))
        cfun=t_$name ;;
      *)
        expect=$(printf '0x%08x' $(( w & ~0x7f80 ))) ; cfun=t_$name ;;
    esac
    check "C $name" "$expect" "$(pimword "$CDUMP" "$cfun")"
    # Every C vector form has its own vsetvli e8,m8 before the PIM word.
    case $name in
      pim_sb_ewadd|pim_sb_ewmul|pim_sb_ewadd_gbuf|pim_sb_ewmul_gbuf|pim_sb_mac)
        ;;
      *)
        funcs=$cfun
        case $name in
          pim_wr_gb|pim_wr_reg|pim_rd_gb|pim_rd_reg)
            funcs="$cfun ${cfun}_mem" ;;
        esac
        for f in $funcs; do
            n=$(printf '%s\n' "$CDUMP" | awk -v f="$f" '
                $1 == f && $3 ~ /^vsetvli/ && $3 ~ /e8,m8/ { v = 1 }
                $1 == f && length($2) == 8 && v &&
                    (substr($2,7,2) == "0b" || substr($2,7,2) == "5b") {
                    print "ok"; exit }')
            check "C $f vsetvli e8,m8 before the PIM word" ok "$n"
        done ;;
    esac
    IFS='
'
done
IFS=$OLDIFS

# P2 section 1.5: imm = -32 and the e8, e16 and e32 widths.
extra=$(printf '%s\n' "$SDUMP" | awk '$1 == "s_extra" && length($2) == 8 \
    { printf "%s ", $2 }')
check "macro imm -32 and widths" "fec5c02b 0295845b 0295d45b 0295e45b " \
    "$extra"
# P-LIB section 12, probe 2.
check "cbo.flush (a0)" 0x0025200f "$(printf '%s\n' "$CDUMP" | awk \
    '$1 == "t_pim_cbo_flush" && length($2) == 8 { print "0x" $2; exit }')"
check "fence rw,rw" 0x0330000f "$(printf '%s\n' "$CDUMP" | awk \
    '$1 == "t_pim_fence" && length($2) == 8 { print "0x" $2; exit }')"

echo "test_encoding: $checks checks, $fail failures"
[ "$fail" -eq 0 ]
