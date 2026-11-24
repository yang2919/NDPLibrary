# NDP–gem5 packet specification v2

[한국어](gem5-extension-handoff.ko.md)

The NDP Library and its peer use the same packet structure for requests and replies. This specification defines the bytes exchanged and when an operation is complete. The team implementing the peer chooses how to receive and execute commands.

## 1. Packet structure

```text
|<------------ 32 bytes ------------>|<------- payload_bytes ------->|
| NIP1 header                        | request or response payload   |
```

Each table numbers bytes from zero within the structure it describes. For example, a field starting at byte 4 with size 2 occupies bytes 4 and 5. Integers are unsigned and stored least significant byte first (little-endian): 256 in two bytes is `00 01`. Fields are packed without gaps.

Every request and reply starts with the fixed four bytes `NIP1`. The peer checks these bytes before interpreting the remaining header fields using this format.

|Bytes|Bits|Field|Value or meaning|
|---|---|---|---|
|0–3|0–31|Packet format identifier|NIP1 = 4e 49 50 31|
|4–5|32–47|version|1|
|6–7|48–63|kind|4 request · 5 success · 6 error|
|8–15|64–127|epoch|Nonzero session identifier agreed at startup|
|16–23|128–191|sequence|Request number, starting at 1 and increasing by 1|
|24–27|192–223|payload_bytes|Bytes following the header, at most 8,388,608|
|28–31|224–255|payload_crc32|Checksum for detecting corruption of the payload|

The current transport uses a Unix stream socket. A read may return part of a packet. Collect 32 header bytes, then collect the indicated number of payload bytes. Verify the received payload as follows.

```python
import zlib

received_crc = int.from_bytes(header[28:32], "little")
assert zlib.crc32(payload) == received_crc
```

**Checksum example.** A successful Submit reply has a one-byte payload, `04`. The peer sending the reply calculates CRC32 over that byte. The result is `0xD56F2B94`, so header bytes 28–31 contain `94 2b 6f d5` in little-endian order. The peer receiving the reply calculates CRC32 over the payload it received and discards the reply if the two values differ.

**If a reply is lost.** The requester sends one request at a time and waits for its reply. Suppose the device finishes a MAC accumulation and the connection drops before the reply arrives. The accumulator may already contain the result. Sending the same request again could add the same products twice. If a header error, checksum mismatch, disconnect or timeout prevents a valid reply, treat execution as uncertain and do not automatically resend the request.

## 2. Requests and replies

The first payload byte selects the operation (op). The reply echoes the request epoch and sequence. The success column shows the reply payload alone, with byte values in hex.

|op|Request|Success payload|When to return success|
|---|---|---|---|
|1|Session configuration check|Exact 63-byte request payload|First request, after validating protocol and geometry|
|2|Upload|02|Written bytes are visible to subsequent commands|
|3|Download|03 + returned bytes|All requested bytes have been read|
|4|Submit|04|All commands and their writes have completed|
|5|Close|05|All device accesses for this session have stopped|

**Data movement**

|Operation|Source → destination|What travels in the packet|
|---|---|---|
|Upload request|Library → device memory|The request carries the destination address, length and **data bytes**.|
|Download request|Device memory → library|The request carries the source address and length. The **reply carries the data bytes**.|
|Transfer command inside Submit|Device memory → device memory|The command carries source address, destination address and length. The device copies the bytes locally. Submit returns success when the batch completes.|

For example, Upload writes an input vector into device memory. Transfer moves a resident vector tile from Shared to GPR for computation. Download reads the final result back into the library.

### Request 1 · Session configuration check

The existing code calls this operation `Hello`. Only its numeric value 1 is transmitted.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|1|op|1|
|1|2|codec_version|2|
|3|2|numeric_profile|1 = FP16 storage|
|5|2|mapping_id|1 = Shared address mapping below|
|7|8|channels|Channel count|
|15|8|banks|Banks per channel|
|23|8|rows|Rows per bank|
|31|8|row_bytes|Bytes per row|
|39|8|gb_bytes|Global Buffer bytes per channel|
|47|8|gpr_bytes|GPR capacity in bytes|
|55|8|scratchpad_bytes|Scratchpad capacity in bytes|

### Requests 2–5 · Data and command submission

|Request|Start byte|Size (bytes)|Field and value|
|---|---|---|---|
|Upload / Download|0|1|op = 2 / 3|
|Upload / Download|1|9|Destination / source (Address table below)|
|Upload / Download|10|8|N = number of bytes to write or read|
|Upload|18|N|Data to write. A Download request ends at byte 17|
|Submit|0|1|op = 4|
|Submit|1|Variable|NDP1 command batch (section 3)|
|Close|0|1|op = 5|

Maximum sizes are 8,388,590 bytes for Upload data and 8,388,607 bytes each for Download data and the NDP1 batch inside Submit.

### Error reply · kind = 6

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|1|op|Failed request op, or 0 if missing|
|1|2|error_id|1 InvalidArgument · 2 UnsupportedOperation · 3 OutOfMemory · 4 BackendFailure|
|3|0–256|message|Optional diagnostic; length is payload_bytes − 3, without terminator|

An error reply terminates the request. A runtime fault may leave partial writes. Device accesses must stop before buffers are reused.

## 3. Command batch structure

```text
Submit payload
[04] [NDP1 header: 16 bytes] [command header: 8 bytes] [command body] ...
```

**NDP1 header** · Byte 0 below is the letter N.

Inside Submit, the fixed four bytes `NDP1` follow the request number. They identify the following data as a command batch in the format below.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|4|Command batch format identifier|NDP1 = 4e 44 50 31|
|4|2|version|2|
|6|2|numeric_profile|1|
|8|4|total_bytes|Entire batch length including this 16-byte header|
|12|4|command_count|0–65,536|

**Command header** · The first starts at NDP1 byte 16.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|2|opcode|Command ID in section 4|
|2|2|reserved|0|
|4|4|body_bytes|Length of the command body immediately following this header|

Validate the entire batch, including lengths, counts, IDs, reserved fields and ranges, then execute in order. Commands are adjacent; reject trailing bytes. Device state and data persist between submissions.

**Shared field formats**

|Format|Start byte|Size (bytes)|Meaning|
|---|---|---|---|
|Address|0|1|1 Shared · 2 GPR · 3 Scratchpad · 4 Global Buffer (Transfer destination)|
|Address|1|8|Byte offset within selected space|
|Operand|0|9|Address|
|Operand|9|8|Number of FP16 elements|

```text
Shared byte offset = (((row * channels + channel) * banks + bank) * row_bytes) + byte_in_row
mask bit i = (mask >> i) & 1
```

The current Shared address mapping (`mapping_id = 1`) assumes a `[row][channel][bank][byte_in_row]` layout. Bytes within one bank row are contiguous. The bank index changes next, followed by channel and then row. The mapping can be changed to match the target gem5 memory controller.

A set mask bit selects that channel or bank. For example, channel mask `0x05` selects channels 0 and 2. Tensor data is little-endian FP16.

## 4. Commands and their bodies

Each table starts at byte zero of that command’s body.

### 1 · Transfer

Copy resident bytes within the device. Supported routes are Shared ↔ GPR, Shared ↔ Scratchpad and Shared/GPR → one channel’s Global Buffer.

Global Buffer destinations use space ID 4 and offset `channel × gb_bytes + byte_in_channel`. Both offsets and the positive byte count must be multiples of 32. Each transfer must fit in one channel. For example, with gb_bytes=64, channel 1 starts at {4, 64}. GEMV sends one staged GPR tile per channel. Command codec v2 leaves opcode 3 unused. The outer NIP1 header version remains 1.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|9|src|Source Address|
|9|9|dst|Destination Address|
|18|8|bytes|Number of bytes|

### 2 · InitializeMac

Seed selected accumulators from GPR

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|8|channel_mask|Channel selection mask|
|8|8|bank_mask|Bank selection mask|
|16|8|gpr_offset|Byte offset in GPR|

### 4 · MacAllBanks

Accumulate bank × Global Buffer products in selected channels

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|8|channel_mask|Channel selection mask|
|8|8|row|Bank row index|
|16|8|first_burst_column|Starting column in 32-byte bursts|
|24|8|burst_count|Number of 32-byte bursts|

### 5 · ReadMac

Write accumulated results to GPR in FP16

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|8|channel_mask|Channel selection mask|
|8|8|gpr_offset|Byte offset in GPR|

### 6 · LaunchPnmProgram

Launch a previously loaded RISC-V program by ID with an argument block and stack. The v2 socket protocol has no program-load request. The reference implementation rejects an unknown ID with InvalidArgument and returns UnsupportedOperation for a valid launch because no binary executor is connected.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|8|program_id|Program identifier|
|8|9|arguments|Argument block Address|
|17|8|argument_bytes|Argument block size|
|25|9|stack|Stack Address|
|34|8|stack_bytes|Stack size|
|42|8|instruction_limit|Instruction budget|

### 7 · PnmCompute

Select a predefined operation by kernel ID. For Matmul, the emulator uses the operand addresses and matrix dimensions to compute the result directly. The current decoder uses this path for PNM arithmetic without loading a program binary.

```text
I = input_count
J = index_count
B = 6 + 17 × I
body_bytes = 60 + 17 × I + 8 × J
```

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|2|kernel_id|1 Matmul · 2 Add · 3 Mul · 4 RMSNorm · 5 Softmax · 6 SiLU · 7 Copy · 8 Gather · 9 Scale · 10 Transpose|
|2|4|input_count|Number of input operands|
|6|17 × I|inputs|Ordered input operands|
|B|17|output|Output operand|
|B + 17|8|rows|Row count|
|B + 25|8|width|Output width|
|B + 33|8|inner|Matrix product inner dimension K|
|B + 41|1|transpose_b|0 = unchanged, 1 = transpose second matrix|
|B + 42|4|factor|Scale factor (IEEE 754 binary32)|
|B + 46|4|epsilon|Normalization constant (IEEE 754 binary32)|
|B + 50|4|index_count|Number of indices|
|B + 54|8 × J|indices|Index array, 8 bytes per entry|

### 8 · PnmRelayout

Convert matrix storage layout

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|9|src|Source Address|
|9|9|dst|Destination Address|
|18|8|rows|Row count|
|26|8|columns|Column count|
|34|8|banks|Total banks used by layout|
|42|8|tile_elements|FP16 elements per tile|
|50|8|row_bytes|Physical row size in bytes|
|58|1|pack|1 = dense → banked, 0 = banked → dense|

### 9 · CopyBytes

Copy between valid ranges within or across Shared, GPR and Scratchpad, with memmove semantics for overlapping ranges. L1 uses this command for Shared buffer region copies. Transfer (1) only permits Shared ↔ GPR, Shared ↔ Scratchpad and Shared/GPR → one channel’s Global Buffer and is used for GEMV staging. Both bodies encode the same source address, destination address and byte count.

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|9|src|Source Address|
|9|9|dst|Destination Address|
|18|8|bytes|Number of bytes|

### 10 · FillBytes

Fill a range with a repeated byte

|Start byte|Size (bytes)|Field|Meaning|
|---|---|---|---|
|0|9|dst|Destination Address|
|9|8|bytes|Number of bytes|
|17|1|value|Byte value to repeat|

## 5. Complete packet examples

After session configuration, fill Shared bytes 256 and 257 with `5a` and read them back. In the final reply payload `03 5a 5a`, the first byte identifies Download and the next two are the returned data.

① Submit request · one FillBytes command

```text
4e 49 50 31 01 00 04 00 01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 2b 00 00 00 88 25 8d 90
04 4e 44 50 31 02 00 01 00 2a 00 00 00 01 00 00 00 0a 00 00 00 12 00 00 00 01 00 01 00 00 00 00 00 00 02 00 00 00 00 00 00 00 5a
```

② Submit success reply

```text
4e 49 50 31 01 00 05 00 01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 01 00 00 00 94 2b 6f d5
04
```

③ Download request · two bytes

```text
4e 49 50 31 01 00 04 00 01 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 12 00 00 00 f9 05 e5 f1
03 01 00 01 00 00 00 00 00 00 02 00 00 00 00 00 00 00
```

④ Download success reply

```text
4e 49 50 31 01 00 05 00 01 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 03 00 00 00 7f 6a ed 36
03 5a 5a
```

The first line of each example is the header; the second is the payload.

[NIP1](../src/l1/gem5/metadata.cpp) · [Request / response](../src/l1/gem5/execution.cpp) · [NDP1](../src/l1/ndp/codec.cpp)
