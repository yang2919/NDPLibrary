/*
 * Helpers of the processing-in-memory (PIM) instructions
 * (docs/pim/02-isa-plan.md section 4.5).
 */

#ifndef __ARCH_RISCV_INSTS_PIM_HH__
#define __ARCH_RISCV_INSTS_PIM_HH__

#include "arch/riscv/page_size.hh"
#include "base/types.hh"

namespace gem5
{

namespace RiscvISA
{

/**
 * Whether the bytes [base, base + bytes) of a vector PIM instruction
 * cross a page boundary. The first micro-op then faults before any
 * request is made, so the instruction translates all or nothing and a
 * PIM command is never issued twice (gem5 restarts a vector memory
 * instruction from its first micro-op after a fault).
 */
inline bool
pimCrossesPage(Addr base, Addr bytes)
{
    if (bytes == 0)
        return false;
    return (base & ~(PageBytes - 1)) !=
           ((base + bytes - 1) & ~(PageBytes - 1));
}

} // namespace RiscvISA

} // namespace gem5

#endif // __ARCH_RISCV_INSTS_PIM_HH__
