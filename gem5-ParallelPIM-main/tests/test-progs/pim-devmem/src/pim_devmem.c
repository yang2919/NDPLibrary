/*
 * The emulated /dev/mem of SE mode (PimMemDriver; check V12):
 *
 *   pim-devmem <window base> <window size> <alignment> <malloc MiB> [span]
 *
 * maps the window whole and checks the virtual base's alignment, a second
 * mapping of the same bytes, the refused out-of-window cases, MAP_FIXED,
 * ioctl and a large malloc. The pattern stays in the first <span> bytes
 * (default: the window); for a PIM memory, below its register rows, which
 * normal accesses may not touch. Prints one line per check and
 * "devmem: N failures" at the end.
 *
 *   pim-devmem unmap <window base> <window size> <malloc MiB> [span]
 *
 * checks that unmapping the window keeps its data: maps the window, writes
 * four words, unmaps it, maps it again, reads the words back and unmaps it a
 * second time; then mallocs and touches <malloc MiB>, which must run out of
 * memory when that is more than the OS memory, since no frame of the window
 * may enter the OS memory's pool.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int failures;

static void check(int ok, const char *what)
{
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        failures++;
}

/* Unmapping the window keeps its data and frees no frame of it. */
static int unmap_mode(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: pim-devmem unmap base size malloc_mib "
                        "[span]\n");
        return 2;
    }
    uint64_t base = strtoull(argv[2], 0, 0), size = strtoull(argv[3], 0, 0);
    size_t mib = strtoul(argv[4], 0, 0);
    uint64_t span = argc == 6 ? strtoull(argv[5], 0, 0) : size;
    const uint64_t offs[] = {0, 4096, span / 2, span - 8};

    int fd = open("/dev/mem", O_RDWR);
    check(fd >= 0, "open /dev/mem");
    if (fd < 0)
        return 1;
    volatile uint8_t *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, (off_t)base);
    check(p != MAP_FAILED, "mmap of the whole window");
    if (p == MAP_FAILED)
        return 1;
    for (int i = 0; i < 4; i++)
        *(volatile uint64_t *)(p + offs[i]) = 0x1234567800000000ull + i;
    check(munmap((void *)p, size) == 0, "munmap of the window");
    p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             (off_t)base);
    check(p != MAP_FAILED, "second mmap of the whole window");
    if (p == MAP_FAILED)
        return 1;
    int ok = 1;
    for (int i = 0; i < 4; i++)
        ok &= *(volatile uint64_t *)(p + offs[i]) ==
              0x1234567800000000ull + i;
    check(ok, "data kept across munmap and mmap");
    check(munmap((void *)p, size) == 0, "second munmap of the window");
    printf("devmem: %d failures\n", failures);
    fflush(stdout);
    /* More than the OS memory: runs out unless window frames entered the
     * pool. */
    if (mib) {
        volatile char *m = malloc(mib << 20);
        for (size_t i = 0; m && i < (mib << 20); i += 4096)
            m[i] = 1;
        printf("malloc of %zu MiB after the unmap: %s\n", mib,
               m ? "succeeded" : "NULL");
    }
    return failures != 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "unmap") == 0)
        return unmap_mode(argc, argv);
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: pim-devmem base size align malloc_mib [span]\n");
        return 2;
    }
    uint64_t base = strtoull(argv[1], 0, 0), size = strtoull(argv[2], 0, 0);
    uint64_t align = strtoull(argv[3], 0, 0);
    size_t mib = strtoul(argv[4], 0, 0);
    uint64_t span = argc == 6 ? strtoull(argv[5], 0, 0) : size;

    int fd = open("/dev/mem", O_RDWR);
    check(fd >= 0, "open /dev/mem");
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    volatile uint64_t *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, (off_t)base);
    check(p != MAP_FAILED, "mmap of the whole window");
    if (p == MAP_FAILED)
        return 1;
    printf("window mapped at %p\n", (void *)p);
    check(((uintptr_t)p & (align - 1)) == (base & (align - 1)),
          "virtual base congruent to the physical base");

    /* Write a pattern at spread offsets through the whole mapping. */
    uint64_t step = span / 64;
    for (uint64_t off = 0; off < span; off += step)
        p[off / 8] = off ^ 0x5a5a5a5a12345678ull;
    p[(span - 8) / 8] = 0xfeedfacecafebeefull;

    /* A second mapping of a sub-range: linear if it sees the same bytes. */
    uint64_t sub = 3 * step;
    volatile uint64_t *q = mmap(NULL, 2 * step, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, (off_t)(base + sub));
    check(q != MAP_FAILED, "mmap of a sub-range");
    int same = q != MAP_FAILED;
    for (uint64_t off = 0; same && off < 2 * step; off += step)
        same = q[off / 8] == ((sub + off) ^ 0x5a5a5a5a12345678ull);
    check(same, "the sub-range sees the bytes written through the window");
    if (q != MAP_FAILED) {
        q[8] = 0x1122334455667788ull;
        check(p[(sub + 64) / 8] == 0x1122334455667788ull,
              "a write through the sub-range is seen through the window");
    }
    int ok = 1;
    for (uint64_t off = 0; off < span; off += step)
        ok = ok && p[off / 8] == (off ^ 0x5a5a5a5a12345678ull);
    check(ok && p[(span - 8) / 8] == 0xfeedfacecafebeefull,
          "pattern read back through the window");

    /* Outside the window. */
    errno = 0;
    void *r = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, (off_t)(base - 4096));
    check(r == MAP_FAILED && errno == EINVAL, "mmap below the window: EINVAL");
    errno = 0;
    r = mmap(NULL, 8192, PROT_READ, MAP_SHARED, fd,
             (off_t)(base + size - 4096));
    check(r == MAP_FAILED && errno == EINVAL,
          "mmap across the window's end: EINVAL");
    errno = 0;
    r = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, (off_t)(base + size));
    check(r == MAP_FAILED && errno == EINVAL, "mmap above the window: EINVAL");

    /* MAP_FIXED at an address that is not congruent. */
    errno = 0;
    r = mmap((void *)((uintptr_t)p + size + align + 4096), 4096, PROT_READ,
             MAP_SHARED | MAP_FIXED, fd, (off_t)base);
    check(r == MAP_FAILED && errno == EINVAL,
          "MAP_FIXED, not congruent: EINVAL");

    errno = 0;
    check(ioctl(fd, 0x1234, 0) == -1 && errno == ENOTTY, "ioctl: ENOTTY");

    /* A large allocation, touched page by page (the Exec/MMU trace shows
     * where its frames are). */
    if (mib) {
        size_t n = mib << 20;
        volatile char *m = malloc(n);
        check(m != NULL, "large malloc");
        for (size_t i = 0; m && i < n; i += 4096)
            m[i] = (char)i;
        long sum = 0;
        for (size_t i = 0; m && i < n; i += 4096)
            sum += m[i] == (char)i;
        check(m && sum == (long)(n / 4096), "large malloc read back");
    }

    printf("devmem: %d failures\n", failures);
    return failures != 0;
}
