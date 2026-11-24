/*
 * Processing-in-Memory (PIM) operations carried by a memory Request.
 *
 * A PIM instruction issues uncacheable, strictly ordered requests that
 * carry Request::PIM and the operation in Request bits 54-58
 * (Request::pimOpFlags()). The values follow the specification's command
 * order (docs/pim/02-isa-plan.md section 1.2); 0 means "not PIM".
 */

#ifndef __MEM_PIM_OP_HH__
#define __MEM_PIM_OP_HH__

#include <cstdint>

namespace gem5
{

/** PIM operation carried in Request bits 54-58; 0 means "not PIM". */
enum class PimOp : uint8_t
{
    None = 0,
    SbEwadd,
    SbEwmul,
    SbEwaddGbuf,
    SbEwmulGbuf,
    SbMac,
    SbMovBkgbuf,
    SbMovGbufbk,
    AbEwadd,
    AbEwmul,
    AbEwaddGbuf,
    AbEwmulGbuf,
    AbMac,
    AbMovRegbk,
    AbMovBkreg,
    RdGb,
    WrGb,
    RdReg,
    WrReg,
};

constexpr unsigned NumPimOps = 18;
static_assert(static_cast<unsigned>(PimOp::WrReg) == NumPimOps,
              "PimOp values must be 1..NumPimOps");

/** Load-like operations (the custom-0 opcode). */
constexpr bool
pimOpIsLoad(PimOp op)
{
    return op == PimOp::RdGb || op == PimOp::RdReg;
}

/** Printable name of an operation, e.g. "AbMac", for traces. */
constexpr const char *
pimOpName(PimOp op)
{
    constexpr const char *names[NumPimOps + 1] = {
        "None",        "SbEwadd",     "SbEwmul",     "SbEwaddGbuf",
        "SbEwmulGbuf", "SbMac",       "SbMovBkgbuf", "SbMovGbufbk",
        "AbEwadd",     "AbEwmul",     "AbEwaddGbuf", "AbEwmulGbuf",
        "AbMac",       "AbMovRegbk",  "AbMovBkreg",  "RdGb",
        "WrGb",        "RdReg",       "WrReg",
    };
    unsigned i = static_cast<unsigned>(op);
    return i <= NumPimOps ? names[i] : "Invalid";
}

/** Layout of the scalar forms' rs2 descriptor. */
struct PimDesc
{
    /** Column count; 0 means "to the end of the row". */
    static constexpr unsigned NcolsShift = 0;
    static constexpr uint64_t NcolsMask = 0xffff;
};

} // namespace gem5

#endif // __MEM_PIM_OP_HH__
