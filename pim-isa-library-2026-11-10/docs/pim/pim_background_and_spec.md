# PIM Background and Specification
This document provide background of PIM-integrated systems, and describes what to implement of PIM.
Based on PIM background, our final goal is to integrate (a) custom PIM instructions on the RISC-V GV extensions (RVGV) into gem5 microarchitectural simulation and (b) PIM commands into gem5 DRAM models.

## PIM Background
### Generic PIM Architecture as an Accelerator
A processing-in-memory (PIM) processing unit (PU), wired to each DRAM bank's row buffer, can only access the data stored in the open local row buffer or other temporal storages.
PIM can incorporate a global SRAM buffer that all the banks share within the rank/pseudochannel/channel.
The global buffer size should be the same as the row size of the given DRAM architecture.
Each processing unit can also be attached to two types of temporary storages:
- An addressable SRAM data buffer serving as a register file either storing the intermediate results for the next computation or writing the final results to the local bank
- Simple flip-flops without any indices storing the final results the host CPU reads

PIM basically supports all-bank compute operations where a single command computes on the operands of the same row and column indices across the banks in parallel.
The important thing is that the number of banks that "all-bank" refers to differ from one DRAM architecture (i.e. type) to another, which determines the parallelism level.
For example, the all-bank operation computes across 16 banks
- Within the channel on GDDR6-based PIM: parallel at channel, no rank and pseudo channel, 1 PU per bank, 4 banks per bank group, and 4 bank groups per channel
- Within the pseudo channel on HBM-based PIM: parallel at pseudo channel, no rank and channel, 1 PU per bank, 4 banks per bank group, 4 bank groups per pseduo channel

Due to the bank locality, the PIM PUs cannot directly compute on the operand in a remote bank.
This is possible only if (a) the host CPU explicitly moves the data (operands) to the bank of the target PIM PU (addressed by the bank, row, and column indices), (b) via the global buffer (i.e., copying from one bank to the global buffer, and copying from the global buffer to another bank), (c) or the native PIM network between banks is implemented to reduce the host CPU overhead.
For now, ignore the PIM-native network.

### PIM Memory System
We should define custom PIM instructions to support overlap between PIM and read/write commands later by defining distinct timing parameters from the load/store instructions (i.e., read/write requests and commands).

#### Memories and Address Mapping Policies
A PIM-integrated system has two DRAM memories: the operating system (OS) memory, which holds the kernel and ordinary programs, and the PIM memory, whose banks have PUs. Both are simulated by the Ramulator 2.1 back end.
Both memories use the same physical-to-DRAM address mapping policy, chosen by one configuration setting. Six policies are supported: gem5's `RoRaBaChCo`, `RoRaBaCoCh` and `RoCoRaBaCh`, and Ramulator 2.1's `RoBaRaCoCh`, `ChRaBaRoCo` and `MOP4CLXOR`. Each name lists the DRAM levels from the highest address bits to the lowest.
The channel position follows the policy name. `RoRaBaChCo` interleaves the channels at one DRAM row (the rank-wide row buffer); `ChRaBaRoCo` does not interleave them (each channel is one contiguous range); for the other four policies the channel interleave is configurable, a power of two of at least one cache line, one cache line by default.
The policies trade host performance against PIM programming cost. A fine channel interleave spreads a sequential host stream over all channels; a row-sized interleave keeps each DRAM row of a bank in one address range, so PIM data is copied, and covered by vector PIM instructions, in fewer and larger pieces. The policy is therefore a study parameter, not a fixed design choice.

#### PIM Operand Addresses
Instructions see only virtual addresses. Every PIM instruction takes the virtual address of its operand in a register; the address goes through the full memory management unit (MMU) path, i.e., the translation lookaside buffer (TLB), the page-table walk, the physical memory protection (PMP) and physical memory attributes (PMA) checks, and faults, and the PIM request is never cached.
PIM operand addresses encode DRAM coordinates (channel, rank or pseudo channel, bank group, bank, row, column) under the configured policy. The user-level PIM library holds an encoder for each policy, the inverse of the memory controller's address mapping. A tiled operand must lie in DRAM rows in column order; the library places it by writing every column at its encoder address, so the DRAM content is the same under every policy, and only the number of copies and instructions differs.
The OS does not have any PIM-specific memory management system: PIM region definition and PIM operand allocation are part of the user-level library. The library uses the PIM memory in one of two ways.
- **Way 1: a linear PIM region (every policy; SE and FS modes).** The OS does not manage the PIM memory: it is outside the Syscall Emulation (SE) frame pool and absent from the Full System (FS) device-tree memory nodes. A program maps the whole PIM memory as one contiguous user virtual region through `/dev/mem` (FS) or the emulated PIM driver (SE). The mapping is linear: the virtual offset from the region base equals the physical offset in the PIM memory, so software forms every PIM address as `base + encode(channel, rank, bank group, bank, row, column)`. A user-level allocator hands out DRAM rows (row indices in chosen channels and banks), never byte ranges, so one allocator serves every policy.
- **Way 2: OS huge pages (row-on-top policies only; FS mode only).** The PIM memory is a separate non-uniform memory access (NUMA) node of the OS, and PIM operands are allocated as hugetlbfs huge pages bound to that node. The OS chooses the physical frame and thus the high bits of the row index; the page offset, which translation preserves, encodes every other level and the row within the page. The PIM virtual address space need not be linear or contiguous: a tensor is tiled row by row, and its rows may lie in different pages. This requires a policy with the row index on the highest bits (gem5's three policies, `RoBaRaCoCh` and `MOP4CLXOR`) and these conditions: every huge page holds whole row slices (the bytes of one row index in every bank of every channel; for `MOP4CLXOR`, the aligned group of row indices its XOR mixes); the pages come from hugetlbfs, are locked, and are never transparent huge pages; all operands of one all-bank command lie in one page. Plain `malloc()` or `new` is not used for PIM operands, because their memory is not aligned to row slices and is shared with other data that PIM writes would overwrite; the library's allocator takes whole huge pages and may be exposed as an `operator new`.
- **Registers.** The PU registers are exposed as the top rows of every bank (one row per register, plus one accumulator row). In both ways these rows are kept out of the OS and mapped linearly; in Way 2, through a separate `/dev/mem` mapping.

#### PIM Software Stack
- **PIM compiler**: tiles tensors at the DRAM-row granularity of the target architecture (8 KiB on DDR5, 2 KiB on LPDDR6 and GDDR7, 1 KiB on HBM4 in the default presets), never at the cache-line size. The granularity may become configurable later (for example, a matrix tile).
- **Layout header**: generated per architecture and configuration from the Ramulator 2.1 organization: row and column bytes, banks per parallelism unit, units per channel, channels, the policy, the channel interleave, the row-index bit position, the largest address-contiguous piece of a bank row, and the region bases. Every library function is architecture-aware through it.
- **PIM library**: PIM instruction wrappers (`asm volatile`, with the operand address in a register); the encoder; the allocator of Way 1 or Way 2; placement and copy routines that write tiles through the encoder in address-contiguous pieces; compute functions built from PIM instructions (for example, GEMV); and the coherence protocol for normal loads and stores to PIM data (`cbo.flush` and `fence` over every line of every row a PIM command touches).
- **Background**: an earlier PIM software stack runs on the TinyEMU emulator: PIM instructions through `asm volatile` (`include/`), compute functions such as `PIMgemv()` (`include/function.h`), and a PIM memory manager (`PIMregister()`, `PIMmalloc()`, `addr_gen()`) that maps a linear region at a fixed base through `/dev/mem`, allocates row indices, and encodes `ChRaBaRoCo` addresses, the policy Ramulator was tested with. Way 1 generalizes that design to every policy.

### Terms for PIM-Integrated Systems
For clear understanding of PIM-integrated systems, I define the terms as below:
- **PIM Instruction**: a custom instruction for PIM on RVGV extensions to distinguish from the existing load and store instructions
- **PIM Request**: a packet traversing the microarchitecture (i.e., from the load-store unit to the memory controller) after being invoked by PIM instructions in the core pipeline.
Its definition can be different for each microarchitectural layer (e.g., LSU, TLB (MMU), cache, memory controller, etc.) since the packet is an abstract concept. 
Counterpart examples of load/store instructions are read/write requests.
- **PIM Command**: a translated PIM-specific DRAM command that the memory controller handles after being dequeued.
PIM commands have various timing constraints themselves.
Some of them share the same timing constraints as the existing read and write commands and may require ACT and PRE conditions.
Other PIM commands can define and may require new ACT and PRE commands.

### PIM Programming Model in PIM-Integrated Systems
Only a host xPU (e.g., CPU, GPU, etc.) can program the PIM command sequences via the PIM instructions.

## PIM Command
Before we define custom PIM instructions upon RVGV extensions, we first describe how PIM commands work in DRAM.
We first classify the PIM commands into three types with respect to the computation granularity: single-bank, all-bank, and non-bank.
Each type shares distinct row-opening and -closing conditions, and timing parameters.
We further classify the instructions for single- and all-bank commands into two types: compute and data movement.
The computation granularity means that how many banks involve in the computation for the given PIM command, rather than the required DRAM address to be decoded (i.e., index of the level in the DRAM hierarchy).
We describe the detailed index to be decoded for each command type, and each index to be decoded will help you determine the timing constraints (conditions and parameters).
We also describe assumption, operands, and operaiton for every instruction, including the index to be decoded from the DRAM address.
Their respecitve meaning are:
- Assumption: assumption behind the PIM command.
- Operands: what the operands are and where they are stored.
- Index to be Decoded from DRAM Addr: not every index at every level of the physical address is necessary for PIM commands. Thus, we specify the necessary indices.
- Operation: overall explanation of how the PIM commands perform the high-level operation, including how the PIM command is translated from the given request, what a sequence of the PIM commands results in, etc.

### Single-bank PIM commands
These commands share the same ACT and PRE conditions and timing parameters with the existing RD and WR commands of the given DRAM architecture (or type).
Also, timing parameters of the single-bank PIM commands are the same as the existing RD and WR commands, except that PIM commands do not occupy the external data bus: the data path is bifurcated into the existing RD/WR branch and a PIM branch.
Please note that the conditions and parameters differ from one DRAM architecture to another.
#### Compute Command
- **PIM_SB_EWADD/EWMUL**
    - Assumption: only one of the two PUs of two banks performs element-wise operations, and a PU incorporates an element-wise vector adder/multiplier. Also, two neighboring banks are paired, and the PIM PU at the lower bank index computes.
    - Operands: two vectors whose sizes are the same by $n_{cols}\times BL$, stored in the DRAM rows in two different banks, respectively.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel), bank index, and row index. While the given address does not specify two different bank indices, we assuem that the designated row index implicitly chooses the neighboring bank.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the bank and row indices. Once the designated bank row is open, the memory controller translates the request into $n_{cols}$ PIM_SB_EWADD/EWMUL commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command adds/multiplies each burst of the two vector operands in element-wise. Thus, the $n_{cols}$ commands, each covering one burst of $BL$, result in the final output vector.
- **PIM_SB_EWADD/EWMUL_GBUF**
    - Assumption: a PU incorporates an element-wise vector adder/multiplier.
    - Operands: two vectors whose sizes are the same by $n_{cols}\times BL$, stored in the global buffer and the DRAM row, respectively.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel), bank index, and row index.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the bank and row indices. Once the designated bank row is open, the memory controller translates the request into $n_{cols}$ PIM_SB_EWADD/EWMUL_GBUF commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command adds/multiplies each burst of the two vector operands in element-wise. Thus, the $n_{cols}$ commands, each covering one burst of $BL$, result in the final output vector.
- **PIM_SB_MAC**
    - Assumption: a PU incorporates multiplier, adder tree, and accumulator. The global buffer size equals the DRAM row buffer size.
    - Operands: two vectors whose sizes are the same by $n_{cols}\times BL$, stored in the global buffer and the DRAM row, respectively.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel), bank index, and row index.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the bank and row indices. Once the designated bank row is open, the memory controller translates the request into $n_{cols}$ PIM_SB_MAC commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command multiplies each burst of the two vector operands, adds the multiplication results, and accumulates the addition results to the previous one. In other words, each PIM PU completes the computation and produces a scalar for the given command, which contributes to one of the element of the output vector, and the $n_{cols}$ commands accumulate the scalar results.
#### Data Movement Command
- **PIM_SB_MOV_BKGBUF**
    - Assumption: the global buffer size equals the DRAM row size.
    - Operands: data stored in a DRAM row
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel), bank index, and row index.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the bank and row indices. Once after the designated bank row is open, the memory controller translates the request into $n_{cols}$ PIM_SB_MOV_BKGBUF commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command copies a burst of the open row to the global buffer.
- **PIM_SB_MOV_GBUFBK**
    - Assumption: the global buffer size equals the DRAM row size.
    - Operands: data stored in the global buffer
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel), bank index, and row index.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the bank and row indices. Once after the designated bank row is open, the memory controller translates the request into $n_{cols}$ PIM_SB_MOV_GBUFBK commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command copies a burst of the global buffer to the open row.

### All-bank PIM commands
These commands have four available options for ACT and PRE command conditions and the timinig parameters.
Please note that the timing parameters of the all-bank PIM commands should be configurable.
1. **Ideal**: a single ACT/PRE command can open/close the rows (of the same index) for all the banks within the rank, pseudo channel, or channel (different by DRAM architecture) in parallel.
2. **AiM**: a single ACT command can open the rows for all the banks within the rank, pseudo channel, or channel in parallel, whereas multiple per-bank PRE commands close the opening rows sequentially, as a PRE command does for the RD and WR commands.
3. **Baseline**: multiple per-bank ACT/PRE commands for the existing RD and WR (i.e., single-bank command) opens/closes the rows sequentially .
4. **AESPA**: a special PIM architecture upon the baseline where neither supporting parallel ACT nor PRE. Since the baseline can begin computation with all-bank commands in parallel only after the multiple ACT commands open rows across the banks, the baseline should wait a time window by $n\times$ tFAW. AESPA issues single-bank compute commands concurrently for the open row, thereby completing computations across all banks via concurrent single bank commands. AESPA also concurrently closes the rows across the banks.
#### Compute Command
- **PIM_AB_EWADD/EWMUL**
    - Assumption: only one of the two PUs of two banks performs element-wise operations, and a PU incorporates an element-wise vector adder/multiplier. Also, two neighboring banks are paired, and the PIM PU at the lower bank index computes.
    - Operands: $2\times n_{banks}$ vectors whose sizes are the same by $n_{cols}\times BL$, stored in the DRAM rows in two different banks, respectively.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel) and row index. While the given address does not specify two different bank indices, we assuem that the designated row index implicitly chooses the neighboring bank.
    - Operation: performs the PIM_SB_EWADD/EWMUL in parallel across all the banks. For the given address (specified in the instruction and request), the memory controller identifies the row index. Once the designated rows are open, the memory controller translates the request into $n_{cols}$ PIM_SB_EWADD/EWMUL commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command adds/multiplies each burst of the two vector operands in element-wise. Thus, the $n_{cols}$ commands, each covering one burst of $BL$, result in the final output vector.
- **PIM_AB_EWADD/EWMUL_GBUF**
    - Assumption: a PU incorporates an element-wise vector adder/multiplier.
    - Operands: two vectors whose sizes are the same by $n_{cols}\times BL$, stored in the global buffer and the DRAM row, respectively.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel) and row index.
    - Operation: performs the PIM_SB_EWADD/EWMUL_GBUF in parallel across all the banks, where the operand vector stored in the global buffer are shared across the banks. For the given address (specified in the instruction and request), the memory controller identifies the row index. Once the designated rows are open, the memory controller translates the request into $n_{cols}$ PIM_SB_EWADD/EWMUL_GBUF commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command adds/multiplies each burst of the two vector operands in element-wise. Thus, the $n_{cols}$ commands, each covering one burst of $BL$, result in the final output vector.
- **PIM_AB_MAC**
    - Assumption: every PU incorporates multiplier, adder tree, and accumulator. The global buffer size equals the DRAM row buffer size.
    - Operands: a vector whose size is $n_{cols}\times BL$ stored in the global buffer and a matrix whose size is $n_{banks}\times (n_{cols}\times BL)$, where the $n_{banks}$ matrix rows are distributed across DRAM rows, and each matrix row whose size is $n_{cols}\times BL$ stored in the DRAM row.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel) and row index.
    - Operation: for the given address (specified in the instruction and request), the memory controller identifies the row index. Once after the designated rows of all banks are open, the memory controller translates the request into $n_{cols}$ PIM_AB_MAC commands, each of which increments the column index from zero to $n_{cols}$. During the configured timing parameter, each translated command multiplies each burst of the vector operand and each burst of the $n_{banks}$ rows of the matrix operand, adds the multiplication results, and accumulates the addition results to the previous one. In other words, each PIM PU completes the computation and produces a scalar for the given command, which contributes to one of the element of the output vector, and the $n_{cols}$ commands accumulate the scalar results. 
#### Data Movement Command
- **PIM_AB_MOV_REGBK**
    - Assumption: scratchpad (register file ) is also exposed to the PIM region (layout) so that programmers can address. While the number of registers is configurable, the default is 1.
    - Operands: all the registers of every PIM PU in the rank/pseudo channel/channel
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel) and row index.
    - Operation: copy all data stored in the temporary registers of each PIM PU of the bank to the DRAM row.
- **PIM_AB_MOV_BKREG**
    - Assumption: scratchpad (register file ) is also exposed to the PIM region (layout) so that programmers can address. While the number of registers is configurable, the default is 1.
    - Operands: the addressed DRAM row of every bank in the rank/pseudo channel/channel
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel) and row index.
    - Operation: copy all data stored in the DRAM row of each bank to the temporary registers of the PIM PU of that bank.

### Non-bank PIM commands
These commands are special load and store commands between the host CPU and the PIM-specific storages.
Thus, these commands do not require any row-opening/closing conditions and their timing parameters.
Detailed specification are:
- **RD_GB**
    - Assumption: the global buffer size equals the DRAM row size
    - Operands: data stored in the global buffer
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel)
    - Operation: load the data from the global buffer into the destination vector register; it is not allocated in any cache.
- **WR_GB**
    - Assumption: the global buffer size equals the DRAM row size
    - Operands: DRAM row-sized data stored in the host CPU registers
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel)
    - Operation: store the data from the register into the global buffer.
- **RD_REG**
    - Assumption: the registers are exposed to the PIM region (layout), as for PIM_AB_MOV_REGBK; a read has no side effects.
    - Operands: data stored in the all the temporal registers of the rank/pseudo channel/channel
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel)
    - Operation: load the data all the data stored in the register by gathering them from the designated rank/pseudo channel/channel.
- **WR_REG**
    - Assumption: the registers are exposed to the PIM region (layout); any value may be written, including zeros that reset the registers and accumulators.
    - Operands: data stored in the host CPU registers.
    - Index to be Decoded from DRAM Addr: indices from the channel to the parallelism level (i.e., one of rank, pseudo channel, or channel)
    - Operation: store the data all the data stored in the register by scattering them to the designated rank/pseudo channel/channel.

## PIM Instruction
We define custom PIM instructions upon RVGV extensions, following the PIM command classification.
Refer to the PIM command specifications, suggest how to define custom PIM instructions.
Note that you do not have to define custom instructions for new row-opening/closing commands.
Also, I do not provide detailed RV extension specification for custom instructions.
We delegate you defining the custom PIM instructions upon the PIM command descriptions.
### PIM Instruction for Single-Bank Command
#### Compute Command
- **PIM_SB_EWADD/EWMUL**
  - Implementation Suggestion: an RVG store-like instruction that takes operands from 
- **PIM_SB_MAC**
  - Implementation Suggestion: an RVG store-like instruction
#### Data Movement Command
- **PIM_SB_MOV_BKGBUF**
  - Implementation Suggestion: an RVV store-like instruction
- **PIM_SB_MOV_GBUFBK**
  - Implementation Suggestion: an RVV store-like instruction
### PIM Instruction for All-Bank Command
#### Compute Command
- **PIM_AB_EWADD/EWMUL**
  - Operation: 
  - Implementation Suggestion: an RVV store-like instruction
- **PIM_AB_MAC**
  - Operands: 
  - Operation: 
  - Implementation Suggestion: an RVV store-like instruction
#### Data Movement Command
- **PIM_AB_MOV_REGBK**
- **PIM_AB_MOV_BKREG**
### PIM Instruction for Non-Bank Command
- **RD_GB**
  - Implementation Suggestion: an RVV load-like instruction
- **WR_GB**
  - Implementation Suggestion: an RVV store-like instruction
- **RD_REG**
  - Implementation Suggestion: an RVV load-like instruction
- **WR_REG**
  - Implementation Suggestion: an RVV store-like instruction

## Errata (revised 2026-11-03)
The text above was revised in place on 2026-11-03 so that it states the specification as settled. Every revision of 2026-11-03 kept its line. The revision of 2026-11-06 (below) moved later lines; the line numbers in the table of 2026-11-03 and in "Settled elsewhere" use the numbering before 2026-11-06. Each entry names the decision in `docs/pim/decisions.md` that settled it; the text before the revision is quoted.

| Line | Before | Now | Settled by |
|---|---|---|---|
| 18 | "1 PU per two banks" | "1 PU per bank" | D8.5 (your answer 2026-10-20; P-DRAM Q1, 2026-11-03) |
| 72, 77, 107, 112 | "Thus, $BL\times n_{cols}$ commands results in the final output vector." | $n_{cols}$ commands, each covering one burst | D8.1 (the count is $n_{cols}$ column commands) |
| 82, 117 | "$BL\times n_{cols}$ commands accumulates the scalar results" | the $n_{cols}$ commands accumulate | D8.1 |
| 111 | "..., bank index, and row index." | the row index only | D8 (P4 block: AB requests ignore the bank field) |
| 112 | "identifies the bank and row indices. Once the designated bank rows are open" | the row index; the designated rows | D8 (P4 block) |
| 121 | "Operands: all the registers in the" (cut off) | the registers of every PU in the parallelism unit | D9.6 |
| 126 | "Operands: all the registers" (cut off) | the addressed row of every bank in the parallelism unit | D9.6 (P-DRAM Q2, 2026-11-03) |
| 128 | "copy all data stored in the temporary registers of each PIM PU of the bank to the DRAM row" (a copy of line 123) | row to registers | D9.6 (P-DRAM Q2, 2026-11-03); RD-class timing, D8.2 |
| 138 | "into the L2 cache" | the destination vector register, not allocated in any cache | D4.2 (a) |
| 145 | "Assumption:" (blank) | registers exposed to the PIM region; reads have no side effects | D9.4, D9.5 |
| 150 | "Assumption:" (blank) | registers exposed to the PIM region; any value, zeros reset | D9.4, D9.5 |
| 65 | "timing parameters of the single-bank PIM commands are the same as the existing RD and WR commands." | the same, except the external data bus: SB and AB commands use a separate PIM branch of the data path; this is the PIM/RD/WR overlap spec:25 anticipates | D8 (P-DRAM Q13 (b), 2026-11-03) |

Settled elsewhere, with the text above unchanged:
- spec:16-18 give 16-bank examples. The all-bank width is an architecture parameter (D8.5); the widths per standard are in `docs/pim/dram-architectures-plan.md` section 4.1.
- spec:157-192: the instruction forms are suggestions. The forms, names and encodings are decided in D1 and D2, and the blank or cut-off suggestions (for example spec:164) are covered there.
- spec:96-101: the ACT/PRE modes are recorded in D8.4.

### Revised 2026-11-06: the PIM Memory System section
The "PIM Memory System" section (lines 24-33 before this revision) was replaced by the address-mapping and PIM software stack specification settled in `docs/pim/address-mapping-sw-stack-update.md` (decisions AM1-AM8, 2026-11-06). Line 25 is unchanged. The section grew from 10 to 22 lines.

**Line map.** A citation `spec:L` written before 2026-11-06 with L >= 34 now means line L + 12 (for example, old spec:65 is now line 77). Citations of old lines 26-33 refer to the text quoted below. The recon document and the P2-P5a plans are not edited (D12.2), so they keep the old numbers.

| Old line | Before | Now | Settled by |
|---|---|---|---|
| 26 | "Because instructions can view only the virtual address space, specifying the physical addresses of PIM operands via the PIM instructions requires a PIM region, a dedicated virtual address space directly mapped to a physical address." | Section "PIM Operand Addresses": every PIM instruction takes a virtual address in a register and goes through the full MMU path | AM1 (e) |
| 27 | "- **Physically contiguous, linear mapping**. The mapping keeps the window linear: virtual offset from the mapping base equals PIM offset. The window base is aligned far beyond the page size, so every address bit below the window size survives translation unchanged." | Way 1: a linear region over the whole PIM memory, for every policy, in SE and FS; Way 2: OS huge pages, not linear, row-on-top policies, FS only | AM1 (e); NUMA kernel check (2026-11-06) |
| 28 | "- **The PIM virtual address space is one contiguous user virtual region**. (the mapping) whose offset encodes DRAM coordinates directly. Software forms every PIM address as `base + encode(channel, rank, bank group, bank, row, column)`, using the same bit layout as the physical window. Nothing in the virtual layout is PIM-specific apart from this convention." | Way 1 keeps `base + encode(...)`; Way 2 encodes the levels below the row in the page offset | AM1 (e) |
| 30 | "At this point, you should note two things." | Removed (the two notes are restated in the new subsections) | AM3 |
| 31 | "First, the OS does not have any PIM-specific memory management system. In other words, PIM region definition, PIM operand allocation, etc. are part of a user-level library." | Kept as the third paragraph of "PIM Operand Addresses" | AM1 (e) |
| 32 | "Second, PIM operand addresses are aware of physical address mapping onto indices of a DRAM hierarchy (e.g., channel, rank or pseudo channel, bank, row, column, etc.)." | Restated: operand addresses encode DRAM coordinates under the configured policy; the library holds one encoder per policy | AM5 (b), AM6 (a) |
| 33 | "In other words, both accesses via PIM instructions and load/store instructions, and the user-level PIM management library are aware of configurable physical address mapping policies." | "Memories and Address Mapping Policies": six policies, one setting for both memories, channel position from the policy name, configurable interleave | AM2 (d), AM5 (b), AM8 (b) |

New subsections with no old counterpart: "Memories and Address Mapping Policies" (the OS memory and the PIM memory, both on Ramulator 2.1, AM2 (d); the performance trade-off), "PIM Software Stack" (compiler tiling at the DRAM-row granularity, the generated layout header, the library, the earlier TinyEMU stack as background), and the register rows in both ways (AM4 (a), D9.4).
