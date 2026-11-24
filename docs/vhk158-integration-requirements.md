# VHK158 FPGA integration requirements

**Implementation status.** Library development and validation currently target gem5. This document records the FPGA team's November 2026 clarification, the evidence in the supplied NDP repository, and the contracts needed for a future L1 backend. The team described the execution flows below as examples. Descriptor encodings, addresses used by the drivers, and the integrated device ABI remain open.

## Platforms and execution ownership

| Platform | Host and control path reported by the FPGA team | Execution and reported coverage |
|---|---|---|
| NDP (PNM) prototype | Linux runs on the board's Cortex-A72 in the processing system (PS). Python uses the team's runtime and driver. The example uses submission and completion queues in shared DDR and interrupt notification. | A RISC-V core and vector unit in programmable logic (PL) execute compiled scalar/vector kernels. The team tested GEMM and softmax, among other kernels. |
| PIM emulator prototype | An external Linux host controls a separate VHK158 implementation over PCIe using the Xilinx QDMA Linux driver. The host places data by bank and writes the PIM command sequence to board command memory. | A doorbell starts the FPGA dispatcher, which reads the sequence and directs the bank engines. The team tested GEMV and element-wise multiplication. Emulator sources are still under development and have not been shared. |
| Planned integrated platform | Panesia's CXL IP is being ported to VHK158. The team expects external host control via PCIe after integration. Polling, interrupts and a hybrid policy are under consideration. | A common address space, a PNM-to-PIM command path and cross-engine completion semantics still require confirmation. |

The two prototypes use the same board model and have separate software environments. The team reports that VHK158 has been discontinued and all available boards are in use. Board access or remote testing must be arranged before hardware acceptance. The delivered [NDP release](../cxl-pim_ndp/README.md) contains RTL, binaries and benchmark sources. Its [XSDB helper](../cxl-pim_ndp/scripts/kernel_helpers.tcl) documents a JTAG bring-up path. The PS Linux queue flow above comes from the team's explanation.

## PNM example — GEMM from PS Linux

1. The Python application supplies A and B and requests an output Y. In the proposed L1 adapter, allocation handles identify driver-managed buffers that both the PS and PL can access. The adapter resolves the handles to the addresses expected by the driver.
2. The runtime selects a RISC-V GEMM kernel. The driver submits the operation and input/output buffer references through shared DDR. To implement this step, obtain the kernel selection or load API and the definition of the argument block, including matrix dimensions, orientation, strides, dtype and output extent.
3. The PL core runs the kernel. In the described prototype, an interrupt prompts the host to check the completion queue. L1 must match the completion to this submission and establish output visibility before exposing Y to the next operation or a host download.

The delivered FP16 GEMM benchmark initializes fixed-size arrays itself. Its JTAG helper loads a flat image at 0x40000000, and startup clears BSS. Use it as arithmetic and bring-up evidence. Request a PS-side example that supplies external buffers and launches twice with one resident weight. Whether kernels are preinstalled, loaded at startup or loaded per operation, and whether the PS path accepts ELF, are open questions. The library's [PNM launch ABI](pnm-launch-abi.md) is a proposal. [GEMM benchmark](../cxl-pim_ndp/sw/src/fmatmul_hp/main.c) · [Startup](../cxl-pim_ndp/sw/src/common/crt0.S)

## PIM example — GEMV from an external host

1. Prepare W once. The proposed L1 adapter obtains the board geometry and maps logical W rows and columns to bank tiles, accounting for padding, alignment and the last partial tile. It retains the placement with the prepared allocation so a second input can reuse W.
2. For each X, the **external host** builds the sequence for vector supply, multiply-accumulate and result collection. Transfer the operand data and write the sequence into the FPGA command memory using the board-defined QDMA path. Ensure these writes are visible to the dispatcher before ringing the doorbell.
3. The FPGA dispatcher issues the bank operations. After device compute completion, read the bank results and assemble Y in logical output order. For a tensor-parallel partial, confirm the result format supports FP32 before exposing that operation.

There are three distinct events to specify: command upload completed, bank execution completed, and result data available to the host. A QDMA transfer completion alone establishes only the transport event. The driver documentation describes queue management and host-to-card/card-to-host transfers. Obtain the board-specific command encoding, doorbell register and compute completion definition from the team. [Xilinx QDMA Linux driver](https://xilinx.github.io/dma_ip_drivers/master/QDMA/linux-kernel/html/index.html)

## Mapping to the current library

| Library operation | Proposed FPGA responsibility and integration condition |
|---|---|
| Session, allocate, release | Bind each prototype session to one endpoint. Keep a backend-owned mapping from allocation handles to buffer location, extent and lifetime. Preserve weight and KV allocations across successful launches. Treat transfer between prototypes as explicit until shared storage and coherency are defined. |
| Upload, download, region copy | Use the PS driver's shared DDR mapping for PNM and the agreed QDMA queues for PIM. Confirm allocation ownership, address translation, cache maintenance, transfer limits and supported copy directions for each path. |
| Layout query and transform | Keep tensor shape and dtype in the common runtime. L1 describes dense or banked storage and implements the placement transformation for its target. Reuse gem5/CENT packing only after comparing bank mapping, row size, tile order, padding and result order with a board example. |
| Operation query and execute | Advertise supported engine, shape, layout and numeric combinations for each endpoint. Check single-token Linear routing when selecting a PNM-only endpoint. The PIM prototype requires host-side sequence generation. Using a PNM core as PIM issuer in gem5 remains a distinct architecture to review at integration. |
| Completion and errors | L2 submits blocking backend work. The adapter may wait using interrupts, polling or both. Return only after writes are visible and the device has stopped accessing the buffers, including after a failure. Define drain/reset behavior before reusing timed-out buffers. |

The relevant interfaces are [IBackend](../include/ndp/l1/backend.hpp) and the [PyTorch adapter](../python/ndp/nn.py). The board can use hardware submission queues internally while keeping queue management inside its driver and L1 adapter.

## Constraints from the supplied NDP release

- The benchmark linker maps 1 GiB of DDR containing code, BSS, stack and fixed status records. Obtain the PS driver's actual tensor arena and reserved ranges before assigning addresses. [Linker map](../cxl-pim_ndp/sw/src/common/link.ld)
- The 512 KiB scratchpad at 0x30000000 is on the Ara vector load/store path. Keep scalar arguments and the C stack in DDR. Stage tiles with vector transfers and bound their size. [Board RTL](../cxl-pim_ndp/rtl/board/ara_soc_vhk158.sv) · [Release notes](../cxl-pim_ndp/README.md)
- VLEN is 2048 bits. An FP16 e16,m1 loop must handle tails beyond 128 elements. For softmax, verify the VHK158 workaround for max reduction and compare output with a numerical reference. [ISA limitations](../cxl-pim_ndp/docs/isa_support.md)

## Minimum handoff before hardware implementation

| Owner/path | Most useful artifact to request | First acceptance check when access is available |
|---|---|---|
| PNM driver/runtime | A minimal PS Linux GEMM caller, driver/UAPI version, buffer map, kernel build/load instructions, submission and completion record definitions, and error recovery procedure. | Run two GEMMs with one resident weight and different inputs. Compare returned matrices and confirm completion refers to the right job. Exercise an FP16 vector tail with a supported kernel. |
| PIM emulator | A worked GEMV for a matrix with a bank or tile tail: logical inputs, packed bytes and addresses, instruction bytes, QDMA setup, doorbell sequence, completion indication and expected bank outputs. A trace suffices for host-side preparation while source is unavailable. | Reproduce the packing and command sequence offline. On hardware, reuse W with two vectors, check each bank result and the reconstructed Y, then validate a supported element-wise multiply. |
| Integrated PCIe/CXL platform | Enumeration/control API, ownership of command generation, PNM/PIM address translation, inter-engine transfer/ordering, result format and final completion policy. | Once specified, chain a PIM output into a PNM operation and verify layout conversion, visibility and lifetime across the engine boundary. |
