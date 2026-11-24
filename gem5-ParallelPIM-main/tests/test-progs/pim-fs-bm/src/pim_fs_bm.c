/*
 * FS bare-metal translation test of the processing-in-memory (PIM)
 * instructions.
 *
 * Starts in M-mode at _start (start.S), sets PMP, builds Sv39 page tables
 * and runs the cases in S-mode. The window, virtual WIN_VA, is mapped to a
 * different physical base, WIN_PA, so a correct physical address proves
 * the translation:
 *   4 KiB pages of WIN_VA: 0 RW, 1 RW with D = 0, 2 R only, 3 X only,
 *   4 RW with U = 1, 5-15 RW; WIN_VA + 2 MiB: one 2 MiB RW page.
 * Every S-mode fault traps to M-mode (nothing is delegated), which records
 * mcause and mtval and skips the instruction (F12's access fault excepted:
 * it is re-executed). ecall from S-mode asks
 * M-mode for a PMP change or the end; ecall from U-mode (F13) returns
 * to S-mode. The run ends with
 * m5_fail(0, mask): bit i set if case Fi did not see what it expected.
 * tests/gem5/pim/check_fs_bm.py checks the trace and statistics.
 *
 * Built with PIM_FS_BM_NOPIM (pim-fs-bm-nopim), the same start-up runs the
 * cases' setup with normal accesses only and no PIM instruction: walks and TLB
 * hits on the 4 KiB and 2 MiB pages, the dirty-bit re-walk of page 1 by a
 * normal store (mask bit 2 if D stays 0), the read-only page, a PMP change,
 * page 7's accesses; and F11, a store on a mispredicted path to page 16
 * (D = 0), which must leave D = 0 (mask bit 11).
 *
 * Built with PIM_FS_BM_SV48 (pim-fs-bm-sv48; run with
 * RiscvISA max_paging_mode = "sv48"), the PIM program runs under Sv48: a
 * level-3 table l3 maps VA [0, 512 GiB) through root, so F1-F13 see the
 * same addresses, and adds
 *   F14 satp keeps mode Sv48 (else m5_fail(0, 1 << 14) before any case);
 *       a 512 GiB page (l3[1], VA 512 GiB) maps PA [0, 512 GiB): a load
 *       and a store through it reach sv48_word
 *   F15 a 512 GiB page whose PPN is not 512 GiB aligned (l3[2]): load
 *       page fault
 *   F16 the upper half (l3[256], VA 0xffff800000000000 + x maps x): a
 *       load and a store through it reach sv48_word
 *   F17 a non-canonical address (bit 47 set, bits 63:48 clear): load page
 *       fault with mtval = the address
 *   F18 a non-canonical alias (bit 48 set) of a window page in the TLB:
 *       load page fault, not a TLB hit
 */

#include <stdint.h>

#include "pim_insn.h"

#define WIN_VA 0x40000000ul
#define WIN_PA 0x10000000ul
#define PAGE 4096ul
#define MEGA2 0x200000ul

/* Sv39 PTE bits */
#define PTE_V (1ul << 0)
#define PTE_R (1ul << 1)
#define PTE_W (1ul << 2)
#define PTE_X (1ul << 3)
#define PTE_U (1ul << 4)
#define PTE_A (1ul << 6)
#define PTE_D (1ul << 7)
#define PTE(pa, f) ((((uint64_t)(pa)) >> 12) << 10 | (f))

/* Causes */
#define CAUSE_ILLEGAL_INST 2
#define CAUSE_LOAD_ACCESS 5
#define CAUSE_STORE_MISALIGNED 6
#define CAUSE_STORE_ACCESS 7
#define CAUSE_ECALL_U 8
#define CAUSE_ECALL_S 9
#define CAUSE_LOAD_PAGE 13
#define CAUSE_STORE_PAGE 15

void enter_s(void (*entry)(void));
#ifndef PIM_FS_BM_NOPIM
void f1_kernel(void *win4k, void *win2m, unsigned long desc);
/* F13: U-mode code (start.S), run at its address + USER_ALIAS. */
#define USER_ALIAS 0x80000000ul
extern char u_cbo_inval[], u_cbo_clean[], u_cbo_flush[], u_cbo_zero[];
void run_user(uint64_t entry, uint64_t arg);
#endif
void m5_fail(uint64_t ns_delay, uint64_t code);

static uint64_t root[512] __attribute__((aligned(4096)));
static uint64_t l1w[512] __attribute__((aligned(4096)));
static uint64_t l0a[512] __attribute__((aligned(4096)));
#ifdef PIM_FS_BM_SV48
static uint64_t l3[512] __attribute__((aligned(4096)));
#define GIGA512 (1ul << 39)
#define SV48_HI 0xffff800000000000ul
#define NONCANON 0x0000800000000000ul
#define NONCANON_ALIAS (1ul << 48)
#define SATP_MODE 9ul
#define SATP_ROOT l3
#else
#define SATP_MODE 8ul
#define SATP_ROOT root
#endif

/* Seen by the trap handler; one marker word per case for the trace. */
volatile uint64_t observed_cause, observed_tval;
#ifndef PIM_FS_BM_NOPIM
/* F12: an access fault re-executes the instruction once PMP entry 0 is
 * off; the handler records vstart. */
volatile uint64_t retry_access, observed_vstart;
#endif
#ifdef PIM_FS_BM_SV48
volatile uint64_t markers[32];
volatile uint64_t sv48_word;
#else
volatile uint64_t markers[16];
#endif
static volatile uint64_t failmask;

static inline void
csr_set_mstatus(uint64_t v)
{
    __asm__ volatile("csrs mstatus, %0" :: "r"(v));
}

void
m_main(void)
{
    for (int i = 0; i < 512; i++)
        root[i] = l1w[i] = l0a[i] = 0;
    /* VA [0, 1 GiB): identity gigapage for code, data and tables. */
    root[0] = PTE(0, PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D);
    /* VA [1 GiB, 2 GiB): the window. */
    root[1] = PTE((uint64_t)l1w, PTE_V);
    l1w[0] = PTE((uint64_t)l0a, PTE_V);
    l1w[1] = PTE(WIN_PA + MEGA2, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D);
    for (int i = 0; i < 16; i++) {
        uint64_t f = PTE_V | PTE_R | PTE_W | PTE_A | PTE_D;
        if (i == 1)
            f = PTE_V | PTE_R | PTE_W | PTE_A;          /* D = 0 */
        if (i == 2)
            f = PTE_V | PTE_R | PTE_A | PTE_D;          /* no W */
        if (i == 3)
            f = PTE_V | PTE_X | PTE_A;                  /* no R */
        if (i == 4)
            f |= PTE_U;                                 /* user page */
        l0a[i] = PTE(WIN_PA + i * PAGE, f);
    }
#ifndef PIM_FS_BM_NOPIM
    /* VA [2 GiB, 3 GiB): [0, 1 GiB) again with U = 1, for F13's U-mode
     * code and its cache block. */
    root[2] = PTE(0, PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D);
#endif
#ifdef PIM_FS_BM_NOPIM
    /* F11's page 16: RW with D = 0, touched by no committed store. */
    l0a[16] = PTE(WIN_PA + 16 * PAGE, PTE_V | PTE_R | PTE_W | PTE_A);
    /* F11's page 17, read once (a TLB miss). */
    l0a[17] = PTE(WIN_PA + 17 * PAGE, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D);
#endif
#ifdef PIM_FS_BM_SV48
    for (int i = 0; i < 512; i++)
        l3[i] = 0;
    /* VA [0, 512 GiB): root, as under Sv39. */
    l3[0] = PTE((uint64_t)root, PTE_V);
    /* F14: VA [512 GiB, 1 TiB), one 512 GiB page over PA [0, 512 GiB). */
    l3[1] = PTE(0, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D);
    /* F15: a 512 GiB page at PA 1 GiB, misaligned. */
    l3[2] = PTE(1ul << 30, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D);
    /* F16: the upper half, VA 0xffff800000000000 + x maps x. */
    l3[256] = PTE((uint64_t)root, PTE_V);
#endif
    uint64_t satp = (SATP_MODE << 60) | ((uint64_t)SATP_ROOT >> 12);
    __asm__ volatile("csrw satp, %0\n sfence.vma" :: "r"(satp) : "memory");
#ifdef PIM_FS_BM_SV48
    /* F14: satp keeps a refused mode's old value (Bare); stop before the
     * cases, which assume paging. */
    __asm__ volatile("csrr %0, satp" : "=r"(satp));
    if ((satp >> 60) != SATP_MODE)
        m5_fail(0, 1ul << 14);
#endif
    /* MPP = S, VS = Initial; SUM = 0, MXR = 0. */
    __asm__ volatile("csrc mstatus, %0" :: "r"((3ul << 11) | (1ul << 18) |
                                               (1ul << 19)));
    csr_set_mstatus((1ul << 11) | (1ul << 9));
    extern void s_main(void);
    enter_s(s_main);
}

static void
pmp_window(uint64_t perms)
{
    /* PMP entry 0: NAPOT over the window's first 4 KiB page, before the
     * catch-all entry 7. perms 0 switches it off. */
    uint64_t addr = (WIN_PA >> 2) | ((PAGE >> 3) - 1);
    __asm__ volatile("csrw pmpaddr0, %0" :: "r"(addr));
    uint64_t cfg;
    __asm__ volatile("csrr %0, pmpcfg0" : "=r"(cfg));
    cfg &= ~0xfful;
    if (perms != 0xff)
        cfg |= (3ul << 3) | perms; /* NAPOT */
    __asm__ volatile("csrw pmpcfg0, %0\n sfence.vma" :: "r"(cfg) : "memory");
}

#ifndef PIM_FS_BM_NOPIM
/* F12: PMP entry 0, NAPOT over [pa, pa + size), no access. */
static void
pmp_entry0_none(uint64_t pa, uint64_t size)
{
    uint64_t addr = (pa >> 2) | ((size >> 3) - 1);
    __asm__ volatile("csrw pmpaddr0, %0" :: "r"(addr));
    uint64_t cfg;
    __asm__ volatile("csrr %0, pmpcfg0" : "=r"(cfg));
    cfg = (cfg & ~0xfful) | (3ul << 3);
    __asm__ volatile("csrw pmpcfg0, %0\n sfence.vma" :: "r"(cfg) : "memory");
}
#endif

void
m_trap(uint64_t *regs)
{
    uint64_t cause, epc, tval;
    __asm__ volatile("csrr %0, mcause" : "=r"(cause));
    __asm__ volatile("csrr %0, mepc" : "=r"(epc));
    __asm__ volatile("csrr %0, mtval" : "=r"(tval));
#ifndef PIM_FS_BM_NOPIM
    if (cause == CAUSE_ECALL_U) {
        /* F13: back to S-mode, at run_user's return address. */
        __asm__ volatile("csrc mstatus, %0\n csrs mstatus, %1"
                         :: "r"(3ul << 11), "r"(1ul << 11));
        __asm__ volatile("csrw mepc, %0" :: "r"(regs[1]));
        return;
    }
#endif
    if (cause == CAUSE_ECALL_S) {
        uint64_t service = regs[10];
        if (service == 0) {
            m5_fail(0, failmask);
            for (;;)
                ;
        }
#ifndef PIM_FS_BM_NOPIM
        if (service == 2)
            pmp_entry0_none(regs[11], regs[12]);  /* F12 */
        else
#endif
        pmp_window(service == 0xff ? 0xff : regs[11]);
    } else {
        observed_cause = cause;
        observed_tval = tval;
#ifndef PIM_FS_BM_NOPIM
        if (retry_access && (cause == CAUSE_STORE_ACCESS ||
                             cause == CAUSE_LOAD_ACCESS)) {
            uint64_t vs;
            __asm__ volatile("csrr %0, vstart" : "=r"(vs));
            observed_vstart = vs;
            retry_access = 0;
            pmp_window(0xff);
            return;                               /* re-execute */
        }
#endif
    }
    __asm__ volatile("csrw mepc, %0" :: "r"(epc + 4));
}

static void
ecall(uint64_t service, uint64_t arg)
{
    register uint64_t a0 __asm__("a0") = service;
    register uint64_t a1 __asm__("a1") = arg;
    __asm__ volatile("ecall\n sfence.vma" : "+r"(a0) : "r"(a1) : "memory");
}

#ifndef PIM_FS_BM_NOPIM
static void
ecall2(uint64_t service, uint64_t arg, uint64_t arg2)
{
    register uint64_t a0 __asm__("a0") = service;
    register uint64_t a1 __asm__("a1") = arg;
    register uint64_t a2 __asm__("a2") = arg2;
    __asm__ volatile("ecall\n sfence.vma" : "+r"(a0) : "r"(a1), "r"(a2)
                     : "memory");
}
#endif

#ifndef PIM_FS_BM_NOPIM
static inline void
pim_sb_mac(uint64_t va)
{
    register uint64_t d __asm__("a2") = PIM_DESC(1);
    register uint64_t a __asm__("a1") = va;
    __asm__ volatile(".insn s 0x2b, 4, a2, 0(a1)" :: "r"(d), "r"(a)
                     : "memory");
}

static inline void
pim_ab_mac(uint64_t va)
{
    register uint64_t a __asm__("a1") = va;
    __asm__ volatile(".option push\n .option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     ".insn r 0x5b, 7, 1, x8, a1, x6\n .option pop"
                     :: "r"(a) : "t0", "memory");
}

/* pim.ab.mac over two VLENB-byte micro-ops (e64, m2); returns vstart
 * after it. */
static inline uint64_t
pim_ab_mac2(uint64_t va)
{
    register uint64_t a __asm__("a1") = va;
    uint64_t vlenb, vs;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    __asm__ volatile(".option push\n .option arch, +v\n"
                     "vsetvli t0, %1, e64, m2, ta, ma\n"
                     ".insn r 0x5b, 7, 1, x8, a1, x6\n"
                     "csrr %0, vstart\n .option pop"
                     : "=r"(vs) : "r"(vlenb / 4), "r"(a) : "t0", "memory");
    return vs;
}

static inline void
pim_rd_gb(uint64_t va)
{
    register uint64_t a __asm__("a1") = va;
    __asm__ volatile(".option push\n .option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     ".insn r 0x0b, 7, 1, x8, a1, x0\n .option pop"
                     :: "r"(a) : "t0", "memory");
}

/* F13's cache block. */
static uint64_t cmo_buf[8] __attribute__((aligned(64)));

static inline void
set_senvcfg(uint64_t v)
{
    __asm__ volatile("csrw 0x10a, %0" :: "r"(v) : "memory");
}

#endif /* !PIM_FS_BM_NOPIM */

static void
begin(int c)
{
    observed_cause = 0;
    observed_tval = 0;
    markers[c] = c;
    __asm__ volatile("" ::: "memory");
}

static void
expect(int c, uint64_t cause)
{
    if (observed_cause != cause)
        failmask |= 1ul << c;
}

#ifdef PIM_FS_BM_NOPIM

/*
 * F11: n iterations store to train while a flag is 1, so the branch on
 * the flag is predicted to fall through to the store. In the last
 * iteration the flag is 0 and the store's address is clean, so the store
 * runs only on the wrong path: on O3 while the flag's divide chain still
 * executes (O3 resolves the branch when it executes), on Minor while the
 * older load, a TLB miss to the untouched page slow in the last iteration,
 * still waits (Minor resolves the branch when it commits, in order).
 * Addresses and flag are computed without branches.
 */
static inline volatile uint64_t *
pick(volatile uint64_t *a, volatile uint64_t *b, uint64_t mask)
{
    uintptr_t r;
    /* a ^ ((a ^ b) & mask), in assembly so that it stays branch-free. */
    __asm__("xor %0, %1, %2\n\tand %0, %0, %3\n\txor %0, %0, %1"
            : "=&r"(r) : "r"(a), "r"(b), "r"(mask));
    return (volatile uint64_t *)r;
}

static void __attribute__((noinline))
f11_wrong_path(volatile uint64_t *train, volatile uint64_t *clean,
               volatile uint64_t *slow, int n)
{
    uint64_t k = 1000003;
    for (int i = 0; i < n; i++) {
        uint64_t last = -(uint64_t)(i == n - 1);
        volatile uint64_t *t = pick(train, clean, last);
        volatile uint64_t *l = pick(train, slow, last);
        uint64_t f = ~last & 1, v;
        __asm__ volatile("ld %0, 0(%1)" : "=r"(v) : "r"(l) : "memory");
        __asm__ volatile("mul %0, %0, %1\n\tdivu %0, %0, %1\n\t"
                         "mul %0, %0, %1\n\tdivu %0, %0, %1\n\t"
                         "mul %0, %0, %1\n\tdivu %0, %0, %1\n\t"
                         "mul %0, %0, %1\n\tdivu %0, %0, %1"
                         : "+r"(f) : "r"(k));
        if (f)
            *t = i;
    }
}

void
s_main(void)
{
    volatile uint64_t *w = (volatile uint64_t *)WIN_VA;
    volatile uint64_t *m = (volatile uint64_t *)(WIN_VA + MEGA2);
    uint64_t sum = 0;

    /* F1's pages: walks on the RW 4 KiB pages and the 2 MiB page, then
     * TLB hits on each. */
    begin(1);
    w[0] = 1;
    for (int i = 5; i < 16; i++)
        w[i * PAGE / 8] = i;
    for (int i = 0; i < 4; i++)
        m[i * PAGE / 8] = 0x200 + i;
    for (int i = 5; i < 16; i++)
        sum += w[i * PAGE / 8] + w[i * PAGE / 8 + 1];
    for (int i = 0; i < 4; i++)
        sum += m[i * PAGE / 8];
    expect(1, 0);

    /* F2's page (D = 0): a load maps it without write permission, so
     * the store after it re-walks (tlb.cc) and sets D. */
    begin(2);
    sum += w[PAGE / 8];
    w[PAGE / 8] = 2;
    expect(2, 0);
    /* The PTE update precedes the store, not a later load (RVWMO). */
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (!(l0a[1] & PTE_D))
        failmask |= 1ul << 2;

    /* F3's page (no W): a load only. */
    begin(3);
    sum += w[2 * PAGE / 8];
    expect(3, 0);

    /* F6's PMP: the window's first page RW through entry 0, then off. */
    begin(6);
    ecall(1, 0x3);
    w[1] = 6;
    sum += w[1];
    ecall(1, 0xff);
    expect(6, 0);

    /* F7's page. */
    begin(7);
    volatile uint64_t *p7 = (volatile uint64_t *)(WIN_VA + 7 * PAGE);
    p7[0] = 0x77;
    sum += p7[8];
    expect(7, 0);

    /* F11's page 16 (D = 0): only a wrong-path store reaches it. */
    begin(11);
    f11_wrong_path((volatile uint64_t *)(WIN_VA + 9 * PAGE),
                   (volatile uint64_t *)(WIN_VA + 16 * PAGE),
                   (volatile uint64_t *)(WIN_VA + 17 * PAGE), 32);
    expect(11, 0);
    /* The PTE update precedes the store, not a later load (RVWMO). */
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (l0a[16] & PTE_D)
        failmask |= 1ul << 11;

    markers[15] = sum;
    ecall(0, 0);
}

#else /* !PIM_FS_BM_NOPIM */

#ifdef PIM_FS_BM_SV48
/* Uncompressed, so that the trap handler's mepc + 4 skips them. */
static inline uint64_t
ld_va(uint64_t va)
{
    uint64_t v;
    __asm__ volatile(".option push\n .option norvc\n ld %0, 0(%1)\n"
                     ".option pop" : "=r"(v) : "r"(va) : "memory");
    return v;
}

static inline void
sd_va(uint64_t va, uint64_t v)
{
    __asm__ volatile(".option push\n .option norvc\n sd %1, 0(%0)\n"
                     ".option pop" :: "r"(va), "r"(v) : "memory");
}

static void
sv48_cases(void)
{
    uint64_t word = (uint64_t)&sv48_word;

    /* F14: the 512 GiB page reaches sv48_word (satp's mode: m_main).
     * The fences order the accesses through two VAs of one word: gem5's
     * O3 load-store queue matches stores to loads by VA
     * (src/cpu/o3/lsq_unit.cc), so without them a load through the other
     * VA misses an older store. */
    begin(14);
    sv48_word = 0x1448;
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (ld_va(GIGA512 + word) != 0x1448)
        failmask |= 1ul << 14;
    sd_va(GIGA512 + word, 0x4814);
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (sv48_word != 0x4814)
        failmask |= 1ul << 14;
    expect(14, 0);

    /* F15: the misaligned 512 GiB page. */
    begin(15);
    (void)ld_va(2 * GIGA512 + word);
    expect(15, CAUSE_LOAD_PAGE);

    /* F16: the upper half reaches sv48_word. */
    begin(16);
    sv48_word = 0x1648;
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (ld_va(SV48_HI + word) != 0x1648)
        failmask |= 1ul << 16;
    sd_va(SV48_HI + word, 0x4816);
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (sv48_word != 0x4816)
        failmask |= 1ul << 16;
    expect(16, 0);

    /* F17: bit 47 set, bits 63:48 clear. */
    begin(17);
    (void)ld_va(NONCANON);
    expect(17, CAUSE_LOAD_PAGE);
    if (observed_tval != NONCANON)
        failmask |= 1ul << 17;

    /* F18: page 5 is in the TLB (F1 and this load); its alias with bit
     * 48 set has the same Sv48 VPN. */
    begin(18);
    (void)ld_va(WIN_VA + 5 * PAGE);
    (void)ld_va(NONCANON_ALIAS + WIN_VA + 5 * PAGE);
    expect(18, CAUSE_LOAD_PAGE);
}
#endif /* PIM_FS_BM_SV48 */

void
s_main(void)
{
    /* F1, F10: walks on 4 KiB pages and the 2 MiB page, no fault. */
    begin(1);
    f1_kernel((void *)WIN_VA, (void *)(WIN_VA + MEGA2), PIM_DESC(1));
    expect(1, 0);

    /* F2: a store-like PIM instruction to a page with D = 0. */
    begin(2);
    pim_sb_mac(WIN_VA + 1 * PAGE);
    expect(2, 0);
    /* The PTE update precedes the store, not a later load (RVWMO). */
    __asm__ volatile("fence rw, rw" ::: "memory");
    if (!(l0a[1] & PTE_D))
        failmask |= 1ul << 2;

    /* F3: store permission. */
    begin(3);
    pim_ab_mac(WIN_VA + 2 * PAGE);
    expect(3, CAUSE_STORE_PAGE);

    /* F4: load permission. */
    begin(4);
    pim_rd_gb(WIN_VA + 3 * PAGE);
    expect(4, CAUSE_LOAD_PAGE);

    /* F5: an S-mode access to a user page with SUM = 0. */
    begin(5);
    pim_sb_mac(WIN_VA + 4 * PAGE);
    expect(5, CAUSE_STORE_PAGE);

    /* F6: PMP without W, then without R and W. */
    begin(6);
    ecall(1, 0x1); /* R only */
    pim_sb_mac(WIN_VA);
    expect(6, CAUSE_STORE_ACCESS);
    observed_cause = 0;
    ecall(1, 0x0); /* nothing */
    pim_rd_gb(WIN_VA);
    expect(6, CAUSE_LOAD_ACCESS);
    ecall(1, 0xff); /* entry off */

    /* F7: normal accesses and a PIM request to page 7 (the PMA run makes
     * the window uncacheable). */
    begin(7);
    volatile uint64_t *p7 = (volatile uint64_t *)(WIN_VA + 7 * PAGE);
    p7[0] = 0x77;
    uint64_t v = p7[8];
    __asm__ volatile("" :: "r"(v));
    pim_sb_mac(WIN_VA + 7 * PAGE + 64);
    expect(7, 0);

    /* F8: the ISA guards, before any TLB lookup. */
    begin(8);
    pim_sb_mac(WIN_VA + 4);
    expect(8, CAUSE_STORE_MISALIGNED);
    observed_cause = 0;
    pim_ab_mac(WIN_VA + PAGE - 16);
    expect(8, CAUSE_STORE_MISALIGNED);

    /* F12: PMP denies the second VLENB bytes of
     * page 6, so micro-op 1 of a two-micro-op pim.ab.mac faults after
     * micro-op 0 sent its request. The handler sees vstart = the
     * micro-op's first element (VLENB / 8 at e64), removes the entry and
     * re-executes the instruction, which sends only micro-op 1 again;
     * vstart is 0 after it. */
    begin(12);
    uint64_t vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    ecall2(2, WIN_PA + 6 * PAGE + vlenb, vlenb);
    retry_access = 1;
    uint64_t after = pim_ab_mac2(WIN_VA + 6 * PAGE);
    expect(12, CAUSE_STORE_ACCESS);
    if (observed_vstart != vlenb / 8 || after != 0 || retry_access)
        failmask |= 1ul << 12;

    /* F13: the senvcfg checks of the
     * cache-block operations. In U-mode, with senvcfg = 0 each of
     * cbo.inval, cbo.clean, cbo.flush and cbo.zero is an illegal
     * instruction; with CBIE = 01, CBCFE = 1 and CBZE = 1 none faults.
     * S-mode is not checked: its cbo.flush runs with senvcfg = 0. */
    begin(13);
    char *ops[4] = {u_cbo_inval, u_cbo_clean, u_cbo_flush, u_cbo_zero};
    uint64_t ubuf = (uint64_t)cmo_buf + USER_ALIAS;
    for (int on = 0; on < 2; on++) {
        set_senvcfg(on ? (1ul << 4) | (1ul << 6) | (1ul << 7) : 0);
        for (int i = 0; i < 4; i++) {
            observed_cause = 0;
            run_user((uint64_t)ops[i] + USER_ALIAS, ubuf);
            if (observed_cause != (on ? 0 : CAUSE_ILLEGAL_INST))
                failmask |= 1ul << 13;
        }
    }
    set_senvcfg(0);
    observed_cause = 0;
    __asm__ volatile(".insn i 0x0f, 2, x0, 2(%0)" :: "r"(cmo_buf)
                     : "memory");                      /* cbo.flush */
    expect(13, 0);

#ifdef PIM_FS_BM_SV48
    sv48_cases();
#endif
    ecall(0, 0);
}

#endif /* PIM_FS_BM_NOPIM */
