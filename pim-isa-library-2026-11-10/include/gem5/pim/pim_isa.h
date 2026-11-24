/*
 * Processing-in-Memory (PIM) instruction wrappers.
 *
 * One function per PIM instruction of docs/pim/02-isa-plan.md section 1.2,
 * named after its assembly name with '_' for '.', plus pim_fence() and
 * the cache helpers. Each wrapper is one inline assembly statement, so it
 * becomes exactly one PIM instruction (vector forms: a vsetvli and the
 * instruction; memory forms: a vsetvli, a vector load or store, and the
 * instruction). The wrappers hold no policy and no memory management.
 *
 * Forms (P-LIB section 14, DP-C (a)):
 * - scalar forms take the PIM address and a descriptor, PIM_DESC(ncols);
 *   the address must be 8-byte aligned;
 * - the nine vector forms that carry no data take (addr, nbytes); vs3 is
 *   v0, whose content the instruction ignores;
 * - pim_wr_gb_mem(), pim_rd_gb_mem(), pim_wr_reg_mem() and
 *   pim_rd_reg_mem() move data between host memory and the GB or the
 *   registers;
 * - with the RISC-V vector intrinsics, pim_wr_gb(), pim_rd_gb(),
 *   pim_wr_reg() and pim_rd_reg() take or return a vuint8m8_t.
 * Every vector form sets vl = min(nbytes, VLMAX) at e8, m8 and returns vl,
 * the bytes covered. It never crosses a 4 KiB page or a contiguous piece
 * (PIM_LAYOUT_CONTIG_BYTES) only if the caller keeps nbytes inside them.
 *
 * Built with PIM_BACKEND_REF (the host reference backend), each wrapper
 * calls the C model of util/pim/src/ref instead. Built with PIM_DEBUG,
 * the wrappers check alignment, descriptors and 4 KiB pages, and abort
 * with a message on misuse.
 *
 * Under __ASSEMBLER__, the same instructions are assembler macros.
 */

#ifndef __GEM5_PIM_PIM_ISA_H__
#define __GEM5_PIM_PIM_ISA_H__

/* Element width encodings (the width field of the vector forms). */
#define PIM_E8 0
#define PIM_E16 5
#define PIM_E32 6
#define PIM_E64 7

/* The scalar descriptor (P2 section 1.4): ncols in bits 15:0. */
#define PIM_DESC_NCOLS_SHIFT 0
#define PIM_DESC_NCOLS_MASK 0xffffULL
#define PIM_DESC(ncols)                                                       \
    ((((unsigned long long)(ncols)) & PIM_DESC_NCOLS_MASK)                    \
     << PIM_DESC_NCOLS_SHIFT)

/* Cache line size of the coherence helpers. */
#define PIM_CACHE_LINE 64

#ifdef __ASSEMBLER__

/* clang-format off */
/* .insn r wants x<N> where a vector form has v<N>. */
.macro pim_vinsn opcode, width, vreg, rs1, op
.irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, \
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
.ifc \vreg, v\n
.insn r \opcode, \width, 1, x\n, \rs1, x\op
.endif
.endr
.endm

/* Scalar forms: pim_<name> rs2, off, rs1 is <name> rs2, off(rs1). */
.macro pim_sb_ewadd rs2, off, rs1
.insn s 0x2b, 0, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewmul rs2, off, rs1
.insn s 0x2b, 1, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewadd_gbuf rs2, off, rs1
.insn s 0x2b, 2, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewmul_gbuf rs2, off, rs1
.insn s 0x2b, 3, \rs2, \off(\rs1)
.endm
.macro pim_sb_mac rs2, off, rs1
.insn s 0x2b, 4, \rs2, \off(\rs1)
.endm

/* Vector forms: pim_<name> vreg, rs1[, eew] is <name> vreg, (rs1); eew
 * is PIM_E8 (default), PIM_E16, PIM_E32 or PIM_E64. */
.macro pim_sb_mov_bkgbuf vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 0
.endm
.macro pim_sb_mov_gbufbk vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 1
.endm
.macro pim_ab_ewadd vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 2
.endm
.macro pim_ab_ewmul vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 3
.endm
.macro pim_ab_ewadd_gbuf vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 4
.endm
.macro pim_ab_ewmul_gbuf vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 5
.endm
.macro pim_ab_mac vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 6
.endm
.macro pim_ab_mov_regbk vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 7
.endm
.macro pim_ab_mov_bkreg vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 8
.endm
.macro pim_wr_gb vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 9
.endm
.macro pim_wr_reg vs3, rs1, eew=0
pim_vinsn 0x5b, \eew, \vs3, \rs1, 10
.endm
.macro pim_rd_gb vd, rs1, eew=0
pim_vinsn 0x0b, \eew, \vd, \rs1, 0
.endm
.macro pim_rd_reg vd, rs1, eew=0
pim_vinsn 0x0b, \eew, \vd, \rs1, 1
.endm
/* clang-format on */

#else // !__ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PimOp numbers (P2 section 1.2), as the reference backend sees them. */
enum pim_op
{
    PIM_OP_SB_EWADD = 1,
    PIM_OP_SB_EWMUL = 2,
    PIM_OP_SB_EWADD_GBUF = 3,
    PIM_OP_SB_EWMUL_GBUF = 4,
    PIM_OP_SB_MAC = 5,
    PIM_OP_SB_MOV_BKGBUF = 6,
    PIM_OP_SB_MOV_GBUFBK = 7,
    PIM_OP_AB_EWADD = 8,
    PIM_OP_AB_EWMUL = 9,
    PIM_OP_AB_EWADD_GBUF = 10,
    PIM_OP_AB_EWMUL_GBUF = 11,
    PIM_OP_AB_MAC = 12,
    PIM_OP_AB_MOV_REGBK = 13,
    PIM_OP_AB_MOV_BKREG = 14,
    PIM_OP_RD_GB = 15,
    PIM_OP_WR_GB = 16,
    PIM_OP_RD_REG = 17,
    PIM_OP_WR_REG = 18,
};

/* Kinds of the runtime's PIM_DEBUG range check. */
#define PIM_DEBUG_KIND_BANK 1
#define PIM_DEBUG_KIND_GB 2
#define PIM_DEBUG_KIND_REG 3

#ifdef PIM_DEBUG
#include <stdio.h>
#include <stdlib.h>

/*
 * The runtime (libpim) defines this check of buffers, the register window
 * and contiguous pieces; without the runtime it is absent (a weak
 * reference) and only the checks below run.
 */
void pim__debug_range(const char *name, const void *addr, size_t n, int kind)
    __attribute__((weak));

static inline void
pim__debug_hook(const char *name, const void *addr, size_t n, int kind)
{
    if (pim__debug_range) {
        pim__debug_range(name, addr, n, kind);
    }
}

static inline void
pim__debug_scalar(const char *name, const void *addr, uint64_t desc)
{
    if ((uintptr_t)addr & 7) {
        fprintf(stderr, "%s: address %p is not 8-byte aligned\n", name, addr);
        abort();
    }
    if (desc & ~(PIM_DESC_NCOLS_MASK << PIM_DESC_NCOLS_SHIFT)) {
        fprintf(stderr, "%s: reserved descriptor bits set in %#llx\n", name,
                (unsigned long long)desc);
        abort();
    }
    pim__debug_hook(name, addr, 8, PIM_DEBUG_KIND_BANK);
}

static inline void
pim__debug_vector(const char *name, const void *addr, size_t vl, int kind)
{
    const uintptr_t a = (uintptr_t)addr;
    if (vl && (a >> 12) != ((a + vl - 1) >> 12)) {
        fprintf(stderr, "%s: range %p + %zu crosses a 4 KiB page\n", name,
                addr, vl);
        abort();
    }
    pim__debug_hook(name, addr, vl, kind);
}
#define PIM__DEBUG_SCALAR(name, addr, desc) pim__debug_scalar(name, addr, desc)
#define PIM__DEBUG_VECTOR(name, addr, vl, kind)                               \
    pim__debug_vector(name, addr, vl, kind)
#else
#define PIM__DEBUG_SCALAR(name, addr, desc) ((void)0)
#define PIM__DEBUG_VECTOR(name, addr, vl, kind) ((void)0)
#endif // PIM_DEBUG

#if defined(PIM_BACKEND_REF)

/* Each wrapper names its PimOp. */
#define PIM__SEL(select, op) op

/* The reference backend's model (util/pim/src/ref). */
void pim_ref_scalar(enum pim_op op, void *addr, uint64_t desc);
size_t pim_ref_vector(enum pim_op op, void *addr, size_t nbytes,
                      const void *src, void *dst);
void pim_ref_flush(const void *line);
void pim_ref_fence(void);
/* The vl a vsetvli at e8, m8 would give. */
size_t pim_ref_vlmax(size_t nbytes);

#define PIM__SCALAR(name, op)                                                 \
    static inline void name(void *addr, uint64_t desc)                        \
    {                                                                         \
        PIM__DEBUG_SCALAR(#name, addr, desc);                                 \
        pim_ref_scalar(op, addr, desc);                                       \
    }

#define PIM__VECTOR(name, op)                                                 \
    static inline size_t name(void *addr, size_t nbytes)                      \
    {                                                                         \
        PIM__DEBUG_VECTOR(#name, addr, pim_ref_vlmax(nbytes),                 \
                          PIM_DEBUG_KIND_BANK);                               \
        return pim_ref_vector(op, addr, nbytes, NULL, NULL);                  \
    }

#define PIM__STORE_MEM(name, op, kind)                                        \
    static inline size_t name(void *addr, const void *src, size_t nbytes)     \
    {                                                                         \
        PIM__DEBUG_VECTOR(#name, addr, pim_ref_vlmax(nbytes), kind);          \
        return pim_ref_vector(op, addr, nbytes, src, NULL);                   \
    }

#define PIM__LOAD_MEM(name, op, kind)                                         \
    static inline size_t name(const void *addr, void *dst, size_t nbytes)     \
    {                                                                         \
        PIM__DEBUG_VECTOR(#name, addr, pim_ref_vlmax(nbytes), kind);          \
        return pim_ref_vector(op, (void *)addr, nbytes, NULL, dst);           \
    }

static inline void
pim_fence(void)
{ pim_ref_fence(); }

static inline void
pim_cbo_flush(const void *line)
{ pim_ref_flush(line); }

#elif defined(__riscv)

/* .insn r wants x<N> where a vector form has v<N> (P-LIB section 12). */
__asm__(".macro pim_vinsn opcode, width, vreg, rs1, op\n"
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,"
        "22,23,24,25,26,27,28,29,30,31\n"
        ".ifc \\vreg, v\\n\n"
        ".insn r \\opcode, \\width, 1, x\\n, \\rs1, x\\op\n"
        ".endif\n"
        ".endr\n"
        ".endm\n");

/* Each wrapper names its select code (funct3, sumop or lumop). */
#define PIM__SEL(select, op) select
#define PIM__STR(x) #x
#define PIM__XSTR(x) PIM__STR(x)

#ifdef PIM_DEBUG
/* vl as the wrapper's vsetvli will set it (debug checks only). */
static inline size_t
pim__vl(size_t nbytes)
{
    size_t vl;
    __asm__ volatile("vsetvli %0, %1, e8, m8, ta, ma"
                     : "=r"(vl)
                     : "r"(nbytes)
                     : "vl", "vtype");
    return vl;
}
#define PIM__VL(nbytes) pim__vl(nbytes)
#endif

#define PIM__SCALAR(name, funct3)                                             \
    static inline void name(void *addr, uint64_t desc)                        \
    {                                                                         \
        PIM__DEBUG_SCALAR(#name, addr, desc);                                 \
        __asm__ volatile(".insn s 0x2b, " PIM__XSTR(funct3) ", %1, 0(%0)"     \
                         :                                                    \
                         : "r"(addr), "r"(desc)                               \
                         : "memory");                                         \
    }

#define PIM__VECTOR(name, sumop)                                              \
    static inline size_t name(void *addr, size_t nbytes)                      \
    {                                                                         \
        size_t vl;                                                            \
        PIM__DEBUG_VECTOR(#name, addr, PIM__VL(nbytes), PIM_DEBUG_KIND_BANK); \
        __asm__ volatile("vsetvli %0, %1, e8, m8, ta, ma\n\t"                 \
                         "pim_vinsn 0x5b, 0, v0, %2, " PIM__XSTR(sumop)       \
                         : "=&r"(vl)                                          \
                         : "r"(nbytes), "r"(addr)                             \
                         : "memory", "vl", "vtype");                          \
        return vl;                                                            \
    }

#define PIM__V8_CLOBBERS "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"

#define PIM__STORE_MEM(name, sumop, kind)                                     \
    static inline size_t name(void *addr, const void *src, size_t nbytes)     \
    {                                                                         \
        size_t vl;                                                            \
        PIM__DEBUG_VECTOR(#name, addr, PIM__VL(nbytes), kind);                \
        __asm__ volatile("vsetvli %0, %1, e8, m8, ta, ma\n\t"                 \
                         "vle8.v v8, (%2)\n\t"                                \
                         "pim_vinsn 0x5b, 0, v8, %3, " PIM__XSTR(sumop)       \
                         : "=&r"(vl)                                          \
                         : "r"(nbytes), "r"(src), "r"(addr)                   \
                         : "memory", "vl", "vtype", PIM__V8_CLOBBERS);        \
        return vl;                                                            \
    }

#define PIM__LOAD_MEM(name, lumop, kind)                                      \
    static inline size_t name(const void *addr, void *dst, size_t nbytes)     \
    {                                                                         \
        size_t vl;                                                            \
        PIM__DEBUG_VECTOR(#name, addr, PIM__VL(nbytes), kind);                \
        __asm__ volatile(                                                     \
            "vsetvli %0, %1, e8, m8, ta, ma\n\t"                              \
            "pim_vinsn 0x0b, 0, v8, %2, " PIM__XSTR(lumop) "\n\t"             \
                                                           "vse8.v v8, (%3)"  \
            : "=&r"(vl)                                                       \
            : "r"(nbytes), "r"(addr), "r"(dst)                                \
            : "memory", "vl", "vtype", PIM__V8_CLOBBERS);                     \
        return vl;                                                            \
    }

/* A full fence between PIM instructions and normal accesses (D4.1). */
static inline void
pim_fence(void)
{ __asm__ volatile("fence rw, rw" ::: "memory"); }

/* cbo.flush of the cache line holding p (Zicbom; .insn needs no -march). */
static inline void
pim_cbo_flush(const void *line)
{ __asm__ volatile(".insn i 0x0f, 2, x0, %0, 2" : : "r"(line) : "memory"); }

#if defined(__riscv_vector)
#include <riscv_vector.h>

/* pim.wr.gb / pim.wr.reg from a vector register group. */
static inline size_t
pim_wr_gb(void *addr, vuint8m8_t data, size_t nbytes)
{
    size_t vl;
    PIM__DEBUG_VECTOR("pim_wr_gb", addr, PIM__VL(nbytes), PIM_DEBUG_KIND_GB);
    __asm__ volatile("vsetvli %0, %2, e8, m8, ta, ma\n\t"
                     "pim_vinsn 0x5b, 0, %1, %3, 9"
                     : "=&r"(vl)
                     : "vr"(data), "r"(nbytes), "r"(addr)
                     : "memory", "vl", "vtype");
    return vl;
}

static inline size_t
pim_wr_reg(void *addr, vuint8m8_t data, size_t nbytes)
{
    size_t vl;
    PIM__DEBUG_VECTOR("pim_wr_reg", addr, PIM__VL(nbytes), PIM_DEBUG_KIND_REG);
    __asm__ volatile("vsetvli %0, %2, e8, m8, ta, ma\n\t"
                     "pim_vinsn 0x5b, 0, %1, %3, 10"
                     : "=&r"(vl)
                     : "vr"(data), "r"(nbytes), "r"(addr)
                     : "memory", "vl", "vtype");
    return vl;
}

/* pim.rd.gb / pim.rd.reg into a vector register group; *vl gets vl. */
static inline vuint8m8_t
pim_rd_gb(const void *addr, size_t nbytes, size_t *vl)
{
    vuint8m8_t v;
    size_t n;
    PIM__DEBUG_VECTOR("pim_rd_gb", addr, PIM__VL(nbytes), PIM_DEBUG_KIND_GB);
    __asm__ volatile("vsetvli %1, %2, e8, m8, ta, ma\n\t"
                     "pim_vinsn 0x0b, 0, %0, %3, 0"
                     : "=&vr"(v), "=&r"(n)
                     : "r"(nbytes), "r"(addr)
                     : "memory", "vl", "vtype");
    if (vl) {
        *vl = n;
    }
    return v;
}

static inline vuint8m8_t
pim_rd_reg(const void *addr, size_t nbytes, size_t *vl)
{
    vuint8m8_t v;
    size_t n;
    PIM__DEBUG_VECTOR("pim_rd_reg", addr, PIM__VL(nbytes), PIM_DEBUG_KIND_REG);
    __asm__ volatile("vsetvli %1, %2, e8, m8, ta, ma\n\t"
                     "pim_vinsn 0x0b, 0, %0, %3, 1"
                     : "=&vr"(v), "=&r"(n)
                     : "r"(nbytes), "r"(addr)
                     : "memory", "vl", "vtype");
    if (vl) {
        *vl = n;
    }
    return v;
}
#endif // __riscv_vector

#else
#error "pim_isa.h needs a RISC-V target or PIM_BACKEND_REF"
#endif

/* Scalar forms (custom-1; P2 section 1.2 rows 1-5). */
PIM__SCALAR(pim_sb_ewadd, PIM__SEL(0, PIM_OP_SB_EWADD))
PIM__SCALAR(pim_sb_ewmul, PIM__SEL(1, PIM_OP_SB_EWMUL))
PIM__SCALAR(pim_sb_ewadd_gbuf, PIM__SEL(2, PIM_OP_SB_EWADD_GBUF))
PIM__SCALAR(pim_sb_ewmul_gbuf, PIM__SEL(3, PIM_OP_SB_EWMUL_GBUF))
PIM__SCALAR(pim_sb_mac, PIM__SEL(4, PIM_OP_SB_MAC))

/* Vector forms without data (custom-2; rows 6-14). */
PIM__VECTOR(pim_sb_mov_bkgbuf, PIM__SEL(0, PIM_OP_SB_MOV_BKGBUF))
PIM__VECTOR(pim_sb_mov_gbufbk, PIM__SEL(1, PIM_OP_SB_MOV_GBUFBK))
PIM__VECTOR(pim_ab_ewadd, PIM__SEL(2, PIM_OP_AB_EWADD))
PIM__VECTOR(pim_ab_ewmul, PIM__SEL(3, PIM_OP_AB_EWMUL))
PIM__VECTOR(pim_ab_ewadd_gbuf, PIM__SEL(4, PIM_OP_AB_EWADD_GBUF))
PIM__VECTOR(pim_ab_ewmul_gbuf, PIM__SEL(5, PIM_OP_AB_EWMUL_GBUF))
PIM__VECTOR(pim_ab_mac, PIM__SEL(6, PIM_OP_AB_MAC))
PIM__VECTOR(pim_ab_mov_regbk, PIM__SEL(7, PIM_OP_AB_MOV_REGBK))
PIM__VECTOR(pim_ab_mov_bkreg, PIM__SEL(8, PIM_OP_AB_MOV_BKREG))

/* Non-bank forms from and to host memory (rows 15-18). */
PIM__STORE_MEM(pim_wr_gb_mem, PIM__SEL(9, PIM_OP_WR_GB), PIM_DEBUG_KIND_GB)
PIM__STORE_MEM(pim_wr_reg_mem, PIM__SEL(10, PIM_OP_WR_REG), PIM_DEBUG_KIND_REG)
PIM__LOAD_MEM(pim_rd_gb_mem, PIM__SEL(0, PIM_OP_RD_GB), PIM_DEBUG_KIND_GB)
PIM__LOAD_MEM(pim_rd_reg_mem, PIM__SEL(1, PIM_OP_RD_REG), PIM_DEBUG_KIND_REG)

/* Flush the cache lines of [p, p + n), without a fence. */
static inline void
pim_flush_range(const void *p, size_t n)
{
    const uintptr_t first = (uintptr_t)p & ~(uintptr_t)(PIM_CACHE_LINE - 1);
    const uintptr_t end = (uintptr_t)p + n;
    for (uintptr_t a = first; a < end; a += PIM_CACHE_LINE) {
        pim_cbo_flush((const void *)a);
    }
}

#ifdef __cplusplus
}
#endif

#endif // __ASSEMBLER__

#endif // __GEM5_PIM_PIM_ISA_H__
