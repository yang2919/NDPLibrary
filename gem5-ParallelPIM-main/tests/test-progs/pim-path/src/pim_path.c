/*
 * The processing-in-memory (PIM) request-path test in SE mode.
 *
 *   pim-path <window base> <window size> <mode> [<register offset>
 *            [whole|regions]]
 *
 * The window (the PIM memory, or its stand-in) is mapped through /dev/mem
 * (PimMemDriver in SE): whole by default; with "regions" only as two mappings,
 * the bank buffer (the window's first BUF_MAP bytes, which every mode below
 * stays in) and the register rows (from <register offset> to the window's
 * end), because Linux fills and clears one page-table entry per mapped page.
 * The decode kernels' register forms (pim.wr.reg, pim.rd.reg) go to <register
 * offset> in the window (default 0), the others to its start: a PIM memory
 * serves register accesses only in its register rows and bank operations only
 * outside them. Every mapping prints a "PIM-PATH map" line. Modes:
 *   scalar|vstore|vload|all  the decode test's kernels on the window
 *   p1  cbo.clean, cbo.flush, cbo.inval on window lines; the dirty
 *       line survives cbo.inval, a flush with senvcfg.CBIE = 01,
 *       and cbo.zero zeroes a line
 *   p2  a normal store to a window line, then a PIM write to it
 *   p3  two threads: B stores a line, A flushes it, fences, PIM write
 *   p4  B stores a line, A PIM-writes it without a flush (fatal)
 *   p5  B stores a line, A PIM-reads it without a flush (fatal, or with
 *       pim_dirty_read_fatal off a warning)
 *   p6  B loads a line (clean copy), A PIM-writes it
 *   p7  a stride loop of normal loads over the window (prefetchers)
 * Speculation cases, on a PIM memory:
 *   s1  a store to a window word, then pim.rd.gb of the same address with
 *       no fence: the load gets the global buffer (zeros), not the store's
 *       data, and reaches the memory after the store's write-back
 *   s2  a PIM write on the wrong path of a mispredicted branch: the branch
 *       waits for a divide chain (O3 resolves it when it executes) and for
 *       an older load that misses the caches (Minor resolves it at
 *       commit); only training iterations may write, never the last one's
 *       line
 *   s3  sixteen pim.rd.gb, alternating between two lines in different
 *       channels, reach the memory in program order
 * A full-system checkpoint case, with the markers
 * of riscv-pim.py (gem5 hypercalls):
 *   ckpt  pim.wr.gb and pim.wr.reg of a pattern, read back with pim.rd.gb
 *         and pim.rd.reg and printed ("before"); hypercall 12 (riscv-pim.py
 *         --checkpoint-at marker writes a checkpoint and ends); hypercall
 *         10 (a restored run switches to --cpu); read back and printed again
 *         ("after")
 * With PIM_PATH_MARKERS set (full-system runs, riscv-pim.py --switch-at
 * marker), hypercall 10 follows the mapping and hypercall 11 the "done"
 * line, so that the CPU switch and the debug trace cover only the mode's
 * work.
 * Normal data accesses outside the window use ordinary memory. Output
 * lines start with "PIM-PATH".
 */

#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <termios.h>
#include <unistd.h>

#include "pim_insn.h"

void pim_scalar(void *buf, unsigned long desc);
void pim_vstore(void *buf, void *reg);
void pim_vmulti(void *buf);
void pim_vsplit(void *addr);
void pim_vload(void *buf, void *reg);

static volatile char *win;
static volatile int step;   /* thread hand-shake, in ordinary memory */
static int peer_mode;

static void
marker(const char *group, const char *what)
{
    printf("PIM-PATH %s %s\n", what, group);
    fflush(stdout);
}

#define GROUP(name, call)                                                     \
    do {                                                                      \
        marker(name, "begin");                                                \
        call;                                                                 \
        marker(name, "end");                                                  \
    } while (0)

/* Line 64 of the window's first page: the protocol cases' line. */
#define LINE_OFF 4096
/* The bank buffer of "regions": every mode stays in its first 16 KiB. */
#define BUF_MAP (64 * 1024)

static inline void
fence(void)
{
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void
pim_write(volatile void *p)
{
    /* pim.sb.mac with a one-column descriptor. */
    register unsigned long desc __asm__("a2") = PIM_DESC(1);
    register volatile void *addr __asm__("a1") = p;
    __asm__ volatile(".insn s 0x2b, 4, a2, 0(a1)"
                     :: "r"(desc), "r"(addr) : "memory");
}

static inline void
pim_read(volatile void *p)
{
    /* pim.rd.gb of 32 bytes (e64, vl 4) into v8. */
    register volatile void *addr __asm__("a1") = p;
    __asm__ volatile(".option push\n.option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     ".insn r 0x0b, 7, 1, x8, a1, x0\n"
                     ".option pop"
                     :: "r"(addr) : "t0", "memory");
}

/* pim.rd.gb of 32 bytes into v8 (e64, vl 4), then v8 to out. */
static inline void
pim_read_to(volatile void *p, uint64_t *out)
{
    register volatile void *addr __asm__("a1") = p;
    register uint64_t *dst __asm__("a2") = out;
    __asm__ volatile(".option push\n.option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     ".insn r 0x0b, 7, 1, x8, a1, x0\n"
                     "vse64.v v8, (a2)\n"
                     ".option pop"
                     :: "r"(addr), "r"(dst) : "t0", "memory");
}

/* pim.wr.gb (32 bytes, e64, vl 4) of src at p. */
static inline void
pim_write_gb(volatile void *p, const uint64_t *src)
{
    register volatile void *addr __asm__("a1") = p;
    register const uint64_t *s __asm__("a2") = src;
    __asm__ volatile(".option push\n.option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     "vle64.v v8, (a2)\n"
                     ".insn r 0x5b, 7, 1, x8, a1, x9\n"
                     ".option pop"
                     :: "r"(addr), "r"(s) : "t0", "memory");
}

/* pim.wr.reg (32 bytes, e64, vl 4) of src at p. */
static inline void
pim_write_reg(volatile void *p, const uint64_t *src)
{
    register volatile void *addr __asm__("a1") = p;
    register const uint64_t *s __asm__("a2") = src;
    __asm__ volatile(".option push\n.option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     "vle64.v v8, (a2)\n"
                     ".insn r 0x5b, 7, 1, x8, a1, x10\n"
                     ".option pop"
                     :: "r"(addr), "r"(s) : "t0", "memory");
}

/* pim.rd.reg (32 bytes, e64, vl 4) at p into out. */
static inline void
pim_read_reg_to(volatile void *p, uint64_t *out)
{
    register volatile void *addr __asm__("a1") = p;
    register uint64_t *dst __asm__("a2") = out;
    __asm__ volatile(".option push\n.option arch, +v\n"
                     "vsetivli t0, 4, e64, m1, ta, ma\n"
                     ".insn r 0x0b, 7, 1, x8, a1, x1\n"
                     "vse64.v v8, (a2)\n"
                     ".option pop"
                     :: "r"(addr), "r"(dst) : "t0", "memory");
}

/* A gem5 hypercall (util/m5's RISC-V m5 operation 0x7b | func << 25,
 * M5OP_HYPERCALL 0x71, the number in a0); output is flushed first. */
static void
hypercall(unsigned long n)
{
    /* The console must have sent the output first: hypercall 11 ends the
     * run. */
    fflush(stdout);
    tcdrain(STDOUT_FILENO);
    /* After the call, which may clobber a0. */
    register unsigned long a0 __asm__("a0") = n;
    __asm__ volatile(".long 0xe200007b" : "+r"(a0) : : "memory");
}

static void
print4(const char *what, const uint64_t *gb, const uint64_t *rg)
{
    printf("PIM-PATH ckpt %s gb %lx %lx %lx %lx reg %lx %lx %lx %lx\n",
           what, (unsigned long)gb[0], (unsigned long)gb[1],
           (unsigned long)gb[2], (unsigned long)gb[3], (unsigned long)rg[0],
           (unsigned long)rg[1], (unsigned long)rg[2], (unsigned long)rg[3]);
}

/* a ^ ((a ^ b) & mask), in assembly so that it stays branch-free. */
static inline uintptr_t
pick(uintptr_t a, uintptr_t b, uintptr_t mask)
{
    uintptr_t r;
    __asm__("xor %0, %1, %2\n\tand %0, %0, %3\n\txor %0, %0, %1"
            : "=&r"(r) : "r"(a), "r"(b), "r"(mask));
    return r;
}

/* s2: n iterations PIM-write the training line while a flag is 1, so the
 * branch on the flag is predicted to fall through to the write; in the
 * last iteration the flag is 0 and the write's address is the clean line,
 * so it runs only on the wrong path: on O3 while the flag's divide chain
 * executes (O3 resolves the branch when it executes); on Minor while the
 * older load, which misses the caches in the last iteration, is
 * outstanding (Minor resolves the branch when it commits). Addresses and
 * flag are branch-free. */
static void __attribute__((noinline))
wrong_path(uintptr_t train, uintptr_t clean, volatile uint64_t *warm,
           volatile uint64_t *cold, int n, volatile uint64_t *sink)
{
    uint64_t k = 1000003, g = 0;
    for (int i = 0; i < n; i++) {
        uintptr_t last = -(uintptr_t)(i == n - 1);
        uintptr_t t = pick(train, clean, last);
        volatile uint64_t *l = (volatile uint64_t *)pick(
            (uintptr_t)warm, (uintptr_t)cold, last);
        uint64_t f = ~last & 1, v;
        /* The PIM write's operands exist before the load: Minor sends a
         * memory instruction to its queue early only when their producers
         * have reached the head of execution. */
        register uint64_t desc __asm__("a2") = PIM_DESC(1);
        register uintptr_t addr __asm__("a1") = t;
        __asm__ volatile("" : "+r"(desc), "+r"(addr));
        __asm__ volatile("ld %0, 0(%1)" : "=r"(v) : "r"(l) : "memory");
        __asm__ volatile(".rept 2\n\tmul %0, %0, %1\n\t"
                         "divu %0, %0, %1\n\t.endr"
                         : "+r"(f) : "r"(k));
        if (f)
            __asm__ volatile(".insn s 0x2b, 4, a2, 0(a1)"
                             :: "r"(desc), "r"(addr) : "memory");
        g += v;
    }
    *sink = g;
}

static void *
peer(void *arg)
{
    (void)arg;
    while (step != 1)
        ;
    volatile uint64_t *line = (volatile uint64_t *)(win + LINE_OFF);
    if (peer_mode == 6) {
        uint64_t v = line[0];
        __asm__ volatile("" :: "r"(v));
    } else {
        line[0] = 0x1122334455667788ull;
    }
    fence();
    step = 2;
    while (step != 3)
        ;
    return NULL;
}

static void
two_threads(int mode)
{
    pthread_t t;
    peer_mode = mode;
    if (pthread_create(&t, NULL, peer, NULL)) {
        printf("PIM-PATH pthread_create failed\n");
        exit(1);
    }
    step = 1;
    while (step != 2)
        ;
    volatile char *line = win + LINE_OFF;
    marker("pim", "begin");
    if (mode == 3) {
        __asm__ volatile("cbo.flush (%0)" :: "r"(line) : "memory");
        fence();
        pim_write(line);
    } else if (mode == 5) {
        pim_read(line);
    } else {
        pim_write(line);
    }
    fence();
    marker("pim", "end");
    step = 3;
    pthread_join(t, NULL);
}

int
main(int argc, char **argv)
{
    if (argc < 4 || argc > 6 ||
        (argc == 6 && strcmp(argv[5], "whole") &&
         strcmp(argv[5], "regions"))) {
        fprintf(stderr, "usage: %s base size mode [reg_offset "
                "[whole|regions]]\n", argv[0]);
        return 2;
    }
    unsigned long base = strtoul(argv[1], 0, 0);
    unsigned long size = strtoul(argv[2], 0, 0);
    const char *mode = argv[3];
    unsigned long reg_off = argc >= 5 ? strtoul(argv[4], 0, 0) : 0;
    const int regions = argc == 6 && !strcmp(argv[5], "regions");
    if (regions && (reg_off < BUF_MAP || reg_off >= size ||
                    reg_off % 4096)) {
        fprintf(stderr, "PIM-PATH regions: the register offset must be a "
                "page multiple in [%#x, size)\n", BUF_MAP);
        return 2;
    }
    int fd = open("/dev/mem", O_RDWR);
    if (fd < 0) {
        perror("PIM-PATH open /dev/mem");
        return 1;
    }
    unsigned long buf_size = regions ? BUF_MAP : size;
    win = mmap(NULL, buf_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
    if (win == MAP_FAILED) {
        perror("PIM-PATH mmap");
        return 1;
    }
    printf("PIM-PATH map %p base %#lx size %#lx\n", (void *)win, base,
           buf_size);
    void *buf = (void *)win;
    void *reg = (char *)buf + reg_off;
    if (regions) {
        reg = mmap(NULL, size - reg_off, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, base + reg_off);
        if (reg == MAP_FAILED) {
            perror("PIM-PATH mmap");
            return 1;
        }
        printf("PIM-PATH map %p base %#lx size %#lx\n", reg, base + reg_off,
               size - reg_off);
    }
    printf("PIM-PATH reg %p offset %#lx\n", reg, reg_off);
    const int markers = getenv("PIM_PATH_MARKERS") != NULL;
    if (markers)
        hypercall(10);
    const int all = !strcmp(mode, "all");
    if (all || !strcmp(mode, "scalar"))
        GROUP("scalar", pim_scalar(buf, PIM_DESC(4)));
    if (all || !strcmp(mode, "vstore")) {
        GROUP("vstore", pim_vstore(buf, reg));
        GROUP("vmulti", pim_vmulti(buf));
        GROUP("vsplit", pim_vsplit((char *)buf + 48));
    }
    if (all || !strcmp(mode, "vload"))
        GROUP("vload", pim_vload(buf, reg));
    if (!strcmp(mode, "p1")) {
        volatile uint64_t *p = (volatile uint64_t *)(win + LINE_OFF);
        for (int i = 0; i < 32; i++)
            p[i] = i + 1;
        marker("cmo", "begin");
        __asm__ volatile("cbo.clean (%0)" :: "r"(p) : "memory");
        __asm__ volatile("cbo.flush (%0)" :: "r"(p + 8) : "memory");
        __asm__ volatile("cbo.inval (%0)" :: "r"(p + 16) : "memory");
        /* cbo.zero, in .insn form: -march has no Zicboz. */
        __asm__ volatile(".insn i 0x0f, 2, x0, 4(%0)" :: "r"(p + 24)
                         : "memory");
        fence();
        marker("cmo", "end");
        int kept = 0, zeroed = 0;
        for (int i = 0; i < 8; i++) {
            kept += p[16 + i] == (uint64_t)(16 + i + 1);
            zeroed += p[24 + i] == 0;
        }
        printf("PIM-PATH inval kept %d/8 zero %d/8\n", kept, zeroed);
    }
    if (!strcmp(mode, "p2")) {
        volatile uint64_t *p = (volatile uint64_t *)(win + LINE_OFF);
        p[0] = 0x0123456789abcdefull;
        fence();
        marker("pim", "begin");
        pim_write(p);
        fence();
        marker("pim", "end");
    }
    if (mode[0] == 'p' && mode[1] >= '3' && mode[1] <= '6' && !mode[2])
        two_threads(mode[1] - '0');
    if (!strcmp(mode, "s1")) {
        volatile uint64_t *p = (volatile uint64_t *)(win + LINE_OFF);
        uint64_t got[4] = {1, 1, 1, 1};
        marker("s1", "begin");
        p[0] = 0x5151515151515151ull;
        pim_read_to(p, got);
        fence();
        marker("s1", "end");
        printf("PIM-PATH s1 v8 %lx %lx %lx %lx\n", (unsigned long)got[0],
               (unsigned long)got[1], (unsigned long)got[2],
               (unsigned long)got[3]);
    }
    if (!strcmp(mode, "s2")) {
        /* One word per page touched first, so that the cold line (another
         * line of the last page) misses the caches without a page fault. */
        enum { PAGES = 64 };
        volatile uint64_t *buf = malloc(PAGES * 4096);
        for (int i = 0; i < PAGES; i++)
            buf[i * 512] = i;
        uintptr_t train = (uintptr_t)(win + 2 * 4096);
        uintptr_t clean = (uintptr_t)(win + 2 * 4096 + 0x800);
        marker("s2", "begin");
        static volatile uint64_t sink;
        wrong_path(train, clean, buf, buf + (PAGES - 1) * 512 + 64, 32,
                   &sink);
        fence();
        marker("s2", "end");
        printf("PIM-PATH s2 train %#lx clean %#lx\n",
               (unsigned long)(train - (uintptr_t)win),
               (unsigned long)(clean - (uintptr_t)win));
    }
    if (!strcmp(mode, "s3")) {
        /* Two lines 2 KiB apart: the LPDDR6 default's channel interleave
         * (one row), so they sit in different channels. */
        volatile char *a = win + 3 * 4096, *b = win + 3 * 4096 + 2048;
        uint64_t got[4];
        marker("s3", "begin");
        for (int i = 0; i < 8; i++) {
            pim_read_to(a + 64 * i, got);
            pim_read_to(b + 64 * i, got);
        }
        fence();
        marker("s3", "end");
        printf("PIM-PATH s3 a %#lx b %#lx\n",
               (unsigned long)(a - win), (unsigned long)(b - win));
    }
    if (!strcmp(mode, "ckpt")) {
        const uint64_t gpat[4] = {0x0123456789abcdefull, 0x1111222233334444ull,
                                  0x5555666677778888ull, 0x99990000aaaabbbbull};
        const uint64_t rpat[4] = {0x3f8000003f800000ull, 0x4000000040400000ull,
                                  0x4080000040a00000ull, 0x40c0000040e00000ull};
        uint64_t gb[4], rg[4];
        marker("ckpt", "begin");
        pim_write_gb(buf, gpat);
        pim_write_reg(reg, rpat);
        fence();
        pim_read_to(buf, gb);
        pim_read_reg_to(reg, rg);
        fence();
        print4("before", gb, rg);
        hypercall(12);
        hypercall(10);
        pim_read_to(buf, gb);
        pim_read_reg_to(reg, rg);
        fence();
        print4("after", gb, rg);
        marker("ckpt", "end");
    }
    if (!strcmp(mode, "p7")) {
        long s = 0;
        marker("stride", "begin");
        for (int i = 0; i < 64; i++)
            s += win[2 * 4096 + i * 64];
        marker("stride", "end");
        printf("PIM-PATH sum %ld\n", s);
    }
    printf("PIM-PATH done %s\n", mode);
    if (markers)
        hypercall(11);
    return 0;
}
