/*
 * An emulated /dev/mem for Processing-in-Memory (PIM) programs in
 * system-call emulation (SE) mode; see pim_mem_driver.hh.
 */

#include "sim/pim_mem_driver.hh"

#include <cerrno>

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "cpu/thread_context.hh"
#include "debug/PIM.hh"
#include "sim/fd_array.hh"
#include "sim/fd_entry.hh"
#include "sim/mem_state.hh"
#include "sim/process.hh"

namespace gem5
{

namespace
{

// MAP_FIXED of the generic Linux ABI, which RISC-V uses
// (src/arch/riscv/linux/linux.hh:212, :394).
constexpr int LinuxMapFixed = 0x10;

Addr
defaultAlignment(const PimMemDriverParams &p)
{
    if (p.alignment)
        return p.alignment;
    const Addr size = p.window.size();
    return isPowerOf2(size) ? size : Addr(1) << ceilLog2(size);
}

} // anonymous namespace

PimMemDriver::PimMemDriver(const PimMemDriverParams &p)
    : EmulatedDriver(p), window(p.window), alignment(defaultAlignment(p))
{
    fatal_if(!window.valid() || window.interleaved(),
             "%s: the window must be one non-interleaved range", name());
    fatal_if(!isPowerOf2(alignment),
             "%s: the alignment (%#x) must be a power of two", name(),
             alignment);
}

int
PimMemDriver::open(ThreadContext *tc, int mode, int flags)
{
    auto p = tc->getProcessPtr();
    auto entry = std::make_shared<DeviceFDEntry>(this, filename);
    const int fd = p->fds->allocFD(entry);
    DPRINTF(PIM, "PimMemDriver: open /dev/%s -> fd %d\n", filename, fd);
    return fd;
}

int
PimMemDriver::ioctl(ThreadContext *tc, unsigned req, Addr buf)
{
    return -ENOTTY;
}

Addr
PimMemDriver::mmap(ThreadContext *tc, Addr start, uint64_t length, int prot,
                   int tgt_flags, int tgt_fd, off_t offset)
{
    auto p = tc->getProcessPtr();
    const Addr paddr = offset;
    // mmapFunc has checked the page alignment of start and offset and
    // rounded length up to whole pages (src/sim/syscall_emul.hh).
    if (offset < 0 || paddr < window.start() || length == 0 ||
        length > window.end() - paddr) {
        DPRINTF(PIM, "PimMemDriver: mmap of %#x, %#x bytes outside %s\n",
                paddr, length, window.to_string());
        return -EINVAL;
    }

    const Addr mask = alignment - 1;
    const Addr phase = paddr & mask;
    Addr vaddr;
    if (tgt_flags & LinuxMapFixed) {
        if ((start & mask) != phase) {
            DPRINTF(PIM, "PimMemDriver: MAP_FIXED address %#x is not "
                    "congruent to %#x modulo %#x\n", start, paddr, alignment);
            return -EINVAL;
        }
        vaddr = start;
        p->memState->unmapRegion(vaddr, length);
    } else if (start && (start & mask) == phase &&
               p->memState->isUnmapped(start, length)) {
        vaddr = start;
    } else {
        // One alignment of slack, so that a congruent base fits.
        const Addr base = p->memState->extendMmap(length + alignment);
        vaddr = base + ((phase - base) & mask);
    }

    p->memState->mapRegion(vaddr, length, "pim-mem");
    p->map(vaddr, paddr, length, true);
    DPRINTF(PIM, "PimMemDriver: mmap %#x-%#x -> physical %#x-%#x\n", vaddr,
            vaddr + length - 1, paddr, paddr + length - 1);
    return vaddr;
}

} // namespace gem5
