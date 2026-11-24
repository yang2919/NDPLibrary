// Freestanding RV64/RVV FP16 kernel. a0 points to the N0 argument block.
.attribute arch, "rv64imafdv_zvfh"
.option norvc
.section .text.entry,"ax",@progbits
.globl pnm_vector_add
.type pnm_vector_add,@function
pnm_vector_add:
    li t0, 40
    bne a1, t0, .Linvalid
    lwu t1, 4(a0)
    bne t1, t0, .Linvalid
    lwu t1, 0(a0)
    li t0, 1
    bne t1, t0, .Linvalid
    lwu t1, 12(a0)
    bnez t1, .Linvalid
    lwu t0, 8(a0)       // remaining FP16 elements
    ld a2, 16(a0)
    ld a3, 24(a0)
    ld a4, 32(a0)
    fsrmi t3, 0          // save frm, select round-to-nearest-even
    beqz t0, .Ldone
.Lloop:
    vsetvli t1, t0, e16, m1, ta, ma
    vle16.v v8, (a2)
    vle16.v v9, (a3)
    vfadd.vv v10, v8, v9
    vse16.v v10, (a4)
    slli t2, t1, 1
    add a2, a2, t2
    add a3, a3, t2
    add a4, a4, t2
    sub t0, t0, t1
    bnez t0, .Lloop
.Ldone:
    fsrm t3
    li a0, 0
    ret
.Linvalid:
    li a0, 1
    ret
.size pnm_vector_add, .-pnm_vector_add
