/*
 * Copyright (c) 2002-2005 The Regents of The University of Michigan
 * Copyright (c) 2007 MIPS Technologies, Inc.
 * Copyright (c) 2020 Barkhausen Institut
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __ARCH_RISCV_PAGETABLE_H__
#define __ARCH_RISCV_PAGETABLE_H__

#include "base/bitunion.hh"
#include "base/logging.hh"
#include "base/trie.hh"
#include "base/types.hh"
#include "sim/serialize.hh"

namespace gem5
{

namespace RiscvISA {

BitUnion64(SATP)
    Bitfield<63, 60> mode;
    Bitfield<59, 44> asid;
    Bitfield<43, 0> ppn;
EndBitUnion(SATP)

enum AddrXlateMode
{
    BARE = 0,
    SV39 = 8,
    SV48 = 9,
    SV57 = 10,
};

// Sv39 paging
const Addr SV39_XLEN = 64;
const Addr SV39_VADDR_BITS  = 39;
const Addr SV39_LEVELS = 3;
const Addr SV39_LEVEL_BITS  = 9;
const Addr SV39X4_WIDENED_BITS = 2;

// Sv48 paging (satp only: vsatp and hgatp stay Sv39 and Sv39x4). Its PTEs
// have the Sv39 layout; PPN[2] is then bits 36:28 and PPN[3] bits 53:37.
const Addr SV48_VADDR_BITS = 48;
const Addr SV48_LEVELS = 4;

BitUnion64(PTESv39)
    Bitfield<63> n;
    Bitfield<62, 54> reserved;
    Bitfield<53, 10> ppn;
    Bitfield<53, 28> ppn2;
    Bitfield<27, 19> ppn1;
    Bitfield<18, 10> ppn0;
    Bitfield<7> d;
    Bitfield<6> a;
    Bitfield<5> g;
    Bitfield<4> u;
    Bitfield<3, 1> perm;
    Bitfield<3> x;
    Bitfield<2> w;
    Bitfield<1> r;
    Bitfield<0> v;
EndBitUnion(PTESv39)

/**
 * Remove the page offset and the upper bits that are
 * not part of the VPN from the address.
 * Note that this must assume the smallest page size
 */
Addr getVPNFromVAddr(Addr vaddr, Addr mode);

/** Virtual-address bits of a satp MODE: 48 for Sv48, else 39 (Sv39). */
unsigned vaddrBits(Addr mode);

/** Page-table levels of a satp MODE: 4 for Sv48, else 3 (Sv39). */
unsigned pagingLevels(Addr mode);

/**
 * Whether bits 63 down to the top virtual-address bit of mode are all
 * equal, as the privileged spec requires for Sv39 and Sv48 (else the
 * access page-faults).
 */
bool isCanonical(Addr vaddr, Addr mode);

struct TlbEntry;
typedef Trie<Addr, TlbEntry> TlbEntryTrie;

struct TlbEntry : public Serializable
{
    // The base of the physical page.
    Addr paddr;

    // The beginning of the virtual page this entry maps.
    Addr vaddr;
    // The size of the page this represents, in address bits.
    unsigned logBytes;

    uint16_t asid;

    // The satp MODE the entry was walked in; its TLB key depends on it.
    unsigned xlateMode;

    PTESv39 pte;

    PTESv39 gpte;

    TlbEntryTrie::Handle trieHandle;

    // A sequence number to keep track of LRU.
    uint64_t lruSeq;

    TlbEntry()
        : paddr(0), vaddr(0), logBytes(0), xlateMode(SV39), pte(), gpte(),
          lruSeq(0)
    {}

    // Return the page size in bytes
    Addr size() const
    {
        return (static_cast<Addr>(1) << logBytes);
    }

    void reset()
    {
        paddr = vaddr = logBytes = pte = gpte = lruSeq = 0;
        xlateMode = SV39;
    }

    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;
};

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_PAGETABLE_H__
