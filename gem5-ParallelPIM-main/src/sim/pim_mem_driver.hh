/*
 * An emulated /dev/mem for Processing-in-Memory (PIM) programs in
 * system-call emulation (SE) mode.
 *
 * The driver maps the whole PIM memory, which is kept out of the SE frame
 * pool by its memory's conf_table_reported = False, so no reserved range
 * is needed.
 */

#ifndef __SIM_PIM_MEM_DRIVER_HH__
#define __SIM_PIM_MEM_DRIVER_HH__

#include "base/addr_range.hh"
#include "params/PimMemDriver.hh"
#include "sim/emul_driver.hh"

namespace gem5
{

class PimMemDriver : public EmulatedDriver
{
  public:
    PimMemDriver(const PimMemDriverParams &p);

    /** A device file descriptor for /dev/mem. */
    int open(ThreadContext *tc, int mode, int flags) override;

    /** No ioctl is defined: -ENOTTY. */
    int ioctl(ThreadContext *tc, unsigned req, Addr buf) override;

    /**
     * Map the physical range [offset, offset + length) linearly and
     * cacheable. -EINVAL unless it lies inside the window, or when a
     * MAP_FIXED address is not congruent to offset modulo the alignment.
     */
    Addr mmap(ThreadContext *tc, Addr start, uint64_t length, int prot,
              int tgt_flags, int tgt_fd, off_t offset) override;

  private:
    const AddrRange window;
    const Addr alignment;
};

} // namespace gem5

#endif // __SIM_PIM_MEM_DRIVER_HH__
