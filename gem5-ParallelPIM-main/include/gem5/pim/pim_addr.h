/*
 * Processing-in-Memory (PIM) address encoder and decoder.
 *
 * An offset into the PIM memory (the region base is added by the caller)
 * encodes dynamic random-access memory (DRAM) coordinates under one of
 * six address-mapping policies. pim_geom_encode() is the inverse of the
 * memory controller's decode:
 *
 * - gem5's RoRaBaChCo, RoRaBaCoCh and RoCoRaBaCh, as
 *   DRAMInterface::decodePacket (src/mem/dram_interface.cc:856-895) and
 *   its Ramulator 2.1 port; the flat bank holds the bank group lowest
 *   (:1140-1151), then the bank and, on HBM4, the stack ID (SID); the
 *   rank is the PIM unit (rank or pseudo channel);
 * - Ramulator 2.1's RoBaRaCoCh, ChRaBaRoCo and MOP4CLXOR
 *   (ext/ramulator2/src/ramulator/controller/addr_mapper/impl/);
 * - the channel bits sit at the channel interleave, as gem5's interleaved
 *   address ranges and Ramulator 2.1's Gem5PortInterleave place them.
 *
 * Arithmetic only: usable from C11 and C++17 (gem5 includes it), no
 * inline assembly, no system calls, nothing beyond <stdint.h>.
 *
 * The pim_geom_*() functions take the geometry as an argument. When a
 * layout header has been included first (<gem5/pim/pim_layout.h>), the
 * pim_encode() family uses that layout's constants, which the compiler
 * folds into shifts and masks.
 */

#ifndef __GEM5_PIM_PIM_ADDR_H__
#define __GEM5_PIM_PIM_ADDR_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Policy numbers, as PIM_LAYOUT_POLICY. */
#define PIM_POLICY_RoRaBaChCo 0
#define PIM_POLICY_RoRaBaCoCh 1
#define PIM_POLICY_RoCoRaBaCh 2
#define PIM_POLICY_RoBaRaCoCh 3
#define PIM_POLICY_ChRaBaRoCo 4
#define PIM_POLICY_MOP4CLXOR 5

/* The geometry of one PIM memory; every count a power of two. */
typedef struct pim_geom
{
    uint32_t policy;          /* PIM_POLICY_* */
    uint32_t channels;        /* channels */
    uint64_t intlv_bytes;     /* channel interleave */
    uint32_t units;           /* ranks or pseudo channels per channel */
    uint32_t sids;            /* stack IDs per unit (1 if none) */
    uint32_t bank_groups;     /* bank groups per SID (1 if none) */
    uint32_t banks_per_group; /* banks per bank group */
    uint32_t rows;            /* rows per bank */
    uint32_t cols;            /* columns (transactions) per row */
    uint32_t col_bytes;       /* bytes per column */
} pim_geom_t;

/* DRAM coordinates; unit is the rank or pseudo channel. */
typedef struct pim_coord
{
    uint32_t ch;
    uint32_t unit;
    uint32_t sid;
    uint32_t bg;
    uint32_t bank; /* bank within its bank group */
    uint32_t row;
    uint32_t col;
} pim_coord_t;

/* Inlined into every caller, so that a layout's constants fold away. */
#if defined(__GNUC__) || defined(__clang__)
#define PIM__INLINE static inline __attribute__((always_inline))
#else
#define PIM__INLINE static inline
#endif

/* Internal: bit widths of a geometry. */
typedef struct pim__bits
{
    unsigned tx, col, ch, s, unit, sid, bg, bank, row, stripe;
} pim__bits_t;

PIM__INLINE unsigned
pim__log2(uint64_t x)
{ return x ? (unsigned)__builtin_ctzll(x) : 0; }

PIM__INLINE uint64_t
pim__mask(unsigned w)
{ return w >= 64 ? ~0ULL : (1ULL << w) - 1; }

PIM__INLINE pim__bits_t
pim__geom_bits(const pim_geom_t *g)
{
    pim__bits_t b;
    b.tx = pim__log2(g->col_bytes);
    b.col = pim__log2(g->cols);
    b.ch = pim__log2(g->channels);
    b.s = pim__log2(g->intlv_bytes);
    b.unit = pim__log2(g->units);
    b.sid = pim__log2(g->sids);
    b.bg = pim__log2(g->bank_groups);
    b.bank = pim__log2(g->banks_per_group);
    b.row = pim__log2(g->rows);
    /* RoCoRaBaCh: the column bits of one channel stripe; gem5 uses a
     * stripe of one burst when there is one channel
     * (src/mem/mem_interface.cc:64-65). */
    b.stripe = 0;
    if (g->channels > 1 && g->intlv_bytes > g->col_bytes) {
        b.stripe = pim__log2(g->intlv_bytes / g->col_bytes);
        if (b.stripe > b.col) {
            b.stripe = b.col;
        }
    }
    return b;
}

/*
 * MOP4CLXOR XORs every level with bits (unit, SID, bank group, bank,
 * row, in that order) with the next column bits (mop4clxor.cpp:36-43).
 */
PIM__INLINE void
pim__mop_masks(const pim__bits_t *b, uint32_t col, uint32_t m[5])
{
    const unsigned w[5] = {b->unit, b->sid, b->bg, b->bank, b->row};
    unsigned x = 0;
    for (int i = 0; i < 5; i++) {
        m[i] = 0;
        if (w[i] > 0) {
            m[i] = x < 32 ? (uint32_t)((col >> x) & pim__mask(w[i])) : 0;
            x += w[i];
        }
    }
}

PIM__INLINE void
pim__put(uint64_t *a, unsigned *pos, uint64_t v, unsigned w)
{
    *a |= (v & pim__mask(w)) << *pos;
    *pos += w;
}

PIM__INLINE uint32_t
pim__get(uint64_t a, unsigned *pos, unsigned w)
{
    uint32_t v = (uint32_t)((a >> *pos) & pim__mask(w));
    *pos += w;
    return v;
}

/* Offset in the PIM memory of byte 0 of the coordinates' column. */
PIM__INLINE uint64_t
pim_geom_encode(const pim_geom_t *g, const pim_coord_t *c)
{
    const pim__bits_t b = pim__geom_bits(g);
    uint64_t a = 0;
    unsigned pos = 0;
    uint32_t m[5];

    switch (g->policy) {
        case PIM_POLICY_RoRaBaChCo:
        case PIM_POLICY_RoRaBaCoCh:
            pim__put(&a, &pos, c->col, b.col);
            pim__put(&a, &pos, c->bg, b.bg);
            pim__put(&a, &pos, c->bank, b.bank);
            pim__put(&a, &pos, c->sid, b.sid);
            pim__put(&a, &pos, c->unit, b.unit);
            pim__put(&a, &pos, c->row, b.row);
            break;
        case PIM_POLICY_RoCoRaBaCh:
            pim__put(&a, &pos, c->col, b.stripe);
            pim__put(&a, &pos, c->bg, b.bg);
            pim__put(&a, &pos, c->bank, b.bank);
            pim__put(&a, &pos, c->sid, b.sid);
            pim__put(&a, &pos, c->unit, b.unit);
            pim__put(&a, &pos, c->col >> b.stripe, b.col - b.stripe);
            pim__put(&a, &pos, c->row, b.row);
            break;
        case PIM_POLICY_RoBaRaCoCh:
            pim__put(&a, &pos, c->col, b.col);
            pim__put(&a, &pos, c->unit, b.unit);
            pim__put(&a, &pos, c->sid, b.sid);
            pim__put(&a, &pos, c->bg, b.bg);
            pim__put(&a, &pos, c->bank, b.bank);
            pim__put(&a, &pos, c->row, b.row);
            break;
        case PIM_POLICY_ChRaBaRoCo:
            pim__put(&a, &pos, c->col, b.col);
            pim__put(&a, &pos, c->row, b.row);
            pim__put(&a, &pos, c->bank, b.bank);
            pim__put(&a, &pos, c->bg, b.bg);
            pim__put(&a, &pos, c->sid, b.sid);
            pim__put(&a, &pos, c->unit, b.unit);
            break;
        case PIM_POLICY_MOP4CLXOR:
            pim__mop_masks(&b, c->col, m);
            pim__put(&a, &pos, c->col, 2);
            pim__put(&a, &pos, c->unit ^ m[0], b.unit);
            pim__put(&a, &pos, c->sid ^ m[1], b.sid);
            pim__put(&a, &pos, c->bg ^ m[2], b.bg);
            pim__put(&a, &pos, c->bank ^ m[3], b.bank);
            pim__put(&a, &pos, c->col >> 2, b.col - 2);
            pim__put(&a, &pos, c->row ^ m[4], b.row);
            break;
        default:
            return ~0ULL;
    }
    /* Insert the channel at the interleave. */
    const uint64_t bytes = a << b.tx;
    const uint64_t low = bytes & pim__mask(b.s);
    return ((bytes >> b.s) << (b.s + b.ch)) |
           (((uint64_t)c->ch & pim__mask(b.ch)) << b.s) | low;
}

/*
 * Coordinates of an offset in the PIM memory; returns the byte within the
 * column. Offset bits above the PIM memory are ignored.
 */
PIM__INLINE uint32_t
pim_geom_decode(const pim_geom_t *g, uint64_t off, pim_coord_t *c)
{
    const pim__bits_t b = pim__geom_bits(g);
    const uint64_t low = off & pim__mask(b.s);
    const uint64_t bytes = ((off >> (b.s + b.ch)) << b.s) | low;
    const uint64_t a = bytes >> b.tx;
    unsigned pos = 0;
    uint32_t m[5];
    uint32_t hi;

    c->ch = (uint32_t)((off >> b.s) & pim__mask(b.ch));
    switch (g->policy) {
        case PIM_POLICY_RoRaBaChCo:
        case PIM_POLICY_RoRaBaCoCh:
            c->col = pim__get(a, &pos, b.col);
            c->bg = pim__get(a, &pos, b.bg);
            c->bank = pim__get(a, &pos, b.bank);
            c->sid = pim__get(a, &pos, b.sid);
            c->unit = pim__get(a, &pos, b.unit);
            c->row = pim__get(a, &pos, b.row);
            break;
        case PIM_POLICY_RoCoRaBaCh:
            c->col = pim__get(a, &pos, b.stripe);
            c->bg = pim__get(a, &pos, b.bg);
            c->bank = pim__get(a, &pos, b.bank);
            c->sid = pim__get(a, &pos, b.sid);
            c->unit = pim__get(a, &pos, b.unit);
            c->col |= pim__get(a, &pos, b.col - b.stripe) << b.stripe;
            c->row = pim__get(a, &pos, b.row);
            break;
        case PIM_POLICY_RoBaRaCoCh:
            c->col = pim__get(a, &pos, b.col);
            c->unit = pim__get(a, &pos, b.unit);
            c->sid = pim__get(a, &pos, b.sid);
            c->bg = pim__get(a, &pos, b.bg);
            c->bank = pim__get(a, &pos, b.bank);
            c->row = pim__get(a, &pos, b.row);
            break;
        case PIM_POLICY_ChRaBaRoCo:
            c->col = pim__get(a, &pos, b.col);
            c->row = pim__get(a, &pos, b.row);
            c->bank = pim__get(a, &pos, b.bank);
            c->bg = pim__get(a, &pos, b.bg);
            c->sid = pim__get(a, &pos, b.sid);
            c->unit = pim__get(a, &pos, b.unit);
            break;
        case PIM_POLICY_MOP4CLXOR:
            c->col = pim__get(a, &pos, 2);
            c->unit = pim__get(a, &pos, b.unit);
            c->sid = pim__get(a, &pos, b.sid);
            c->bg = pim__get(a, &pos, b.bg);
            c->bank = pim__get(a, &pos, b.bank);
            hi = pim__get(a, &pos, b.col - 2);
            c->col |= hi << 2;
            c->row = pim__get(a, &pos, b.row);
            pim__mop_masks(&b, c->col, m);
            c->unit ^= m[0];
            c->sid ^= m[1];
            c->bg ^= m[2];
            c->bank ^= m[3];
            c->row ^= m[4];
            break;
        default:
            c->col = c->unit = c->sid = c->bg = c->bank = c->row = 0;
            break;
    }
    return (uint32_t)(bytes & pim__mask(b.tx));
}

/* Sizes. */
static inline uint64_t
pim_geom_row_bytes(const pim_geom_t *g)
{ return (uint64_t)g->cols * g->col_bytes; }

static inline uint32_t
pim_geom_banks_per_unit(const pim_geom_t *g)
{ return g->sids * g->bank_groups * g->banks_per_group; }

/* Bank rows of one row slice (one row index in every bank). */
static inline uint64_t
pim_geom_bank_rows(const pim_geom_t *g)
{ return (uint64_t)g->channels * g->units * pim_geom_banks_per_unit(g); }

/* Bytes of one row slice. */
static inline uint64_t
pim_geom_slice_bytes(const pim_geom_t *g)
{ return pim_geom_bank_rows(g) * pim_geom_row_bytes(g); }

/* Size of the PIM memory. */
static inline uint64_t
pim_geom_size(const pim_geom_t *g)
{ return pim_geom_slice_bytes(g) * g->rows; }

/*
 * The largest aligned range that stays inside one bank row with
 * consecutive columns (PIM_LAYOUT_CONTIG_BYTES): the run of column bits
 * above the byte offset, ended by a channel bit, another level, or a
 * column bit that MOP4CLXOR XORs into a level.
 */
static inline uint64_t
pim_geom_contig_bytes(const pim_geom_t *g)
{
    const uint64_t r = pim_geom_row_bytes(g);
    const uint64_t i = g->intlv_bytes < r ? g->intlv_bytes : r;
    switch (g->policy) {
        case PIM_POLICY_RoRaBaChCo:
        case PIM_POLICY_ChRaBaRoCo:
            return r;
        case PIM_POLICY_RoRaBaCoCh:
        case PIM_POLICY_RoBaRaCoCh:
            return g->channels == 1 ? r : i;
        case PIM_POLICY_RoCoRaBaCh:
            return g->channels == 1 ? g->col_bytes : i;
        default:
            return g->col_bytes;
    }
}

/* The physical bit position of row bit 0 (PIM_LAYOUT_ROW_SHIFT). */
static inline unsigned
pim_geom_row_shift(const pim_geom_t *g)
{
    pim_coord_t c = {0, 0, 0, 0, 0, 1, 0};
    return pim__log2(pim_geom_encode(g, &c));
}

/*
 * Whether row slice r is the one range [r << row shift, (r + 1) << row
 * shift): the row is the top field and MOP4CLXOR XORs none of its bits.
 */
static inline int
pim_geom_slice_contiguous(const pim_geom_t *g)
{
    const pim__bits_t b = pim__geom_bits(g);
    uint32_t m[5];
    if (g->policy == PIM_POLICY_MOP4CLXOR) {
        pim__mop_masks(&b, g->cols - 1, m);
        if (m[4] != 0) {
            return 0;
        }
    }
    return pim_geom_row_shift(g) + b.row == pim__log2(pim_geom_size(g));
}

/* Offset of row slice r; valid when pim_geom_slice_contiguous(). */
static inline uint64_t
pim_geom_slice_offset(const pim_geom_t *g, uint32_t r)
{ return (uint64_t)r << pim_geom_row_shift(g); }

/* Address-contiguous pieces of one bank row. */
static inline uint32_t
pim_geom_pieces_per_row(const pim_geom_t *g)
{ return (uint32_t)(pim_geom_row_bytes(g) / pim_geom_contig_bytes(g)); }

/*
 * Offset of piece i of the bank row of c (c->col is ignored); the piece
 * holds pim_geom_contig_bytes() bytes, columns i * n to i * n + n - 1
 * with n = contig bytes / column bytes.
 */
static inline uint64_t
pim_geom_piece_offset(const pim_geom_t *g, const pim_coord_t *c, uint32_t i)
{
    pim_coord_t p = *c;
    p.col = i * (uint32_t)(pim_geom_contig_bytes(g) / g->col_bytes);
    return pim_geom_encode(g, &p);
}

#ifdef __cplusplus
}
#endif

#endif // __GEM5_PIM_PIM_ADDR_H__

/*
 * The functions of the included layout header (pim_layout.h), with its
 * constants folded in. Outside the include guard, so that it is added
 * whenever this header is included after a layout header;
 * <gem5/pim/pim_layout.h> includes this header at its end.
 */
#if defined(PIM_LAYOUT_POLICY) && !defined(__GEM5_PIM_PIM_ADDR_LAYOUT__)
#define __GEM5_PIM_PIM_ADDR_LAYOUT__

#ifdef __cplusplus
extern "C" {
#endif

#define PIM_GEOM_INIT                                                         \
    {PIM_LAYOUT_POLICY,                                                       \
     PIM_LAYOUT_CHANNELS,                                                     \
     PIM_LAYOUT_INTLV_BYTES,                                                  \
     PIM_LAYOUT_UNITS,                                                        \
     (PIM_LAYOUT_SIDS > 0 ? PIM_LAYOUT_SIDS : 1),                             \
     (PIM_LAYOUT_BANK_GROUPS > 0 ? PIM_LAYOUT_BANK_GROUPS : 1),               \
     PIM_LAYOUT_BANKS_PER_GROUP,                                              \
     PIM_LAYOUT_ROWS,                                                         \
     PIM_LAYOUT_COLS,                                                         \
     PIM_LAYOUT_COL_BYTES}

/* The geometry of the included layout. */
PIM__INLINE const pim_geom_t *
pim_geom(void)
{
    static const pim_geom_t g = PIM_GEOM_INIT;
    return &g;
}

PIM__INLINE uint64_t
pim_encode(const pim_coord_t *c)
{ return pim_geom_encode(pim_geom(), c); }

PIM__INLINE uint32_t
pim_decode(uint64_t off, pim_coord_t *c)
{ return pim_geom_decode(pim_geom(), off, c); }

static inline uint64_t
pim_piece_offset(const pim_coord_t *c, uint32_t i)
{ return pim_geom_piece_offset(pim_geom(), c, i); }

#ifdef __cplusplus
}
#endif

#endif // PIM_LAYOUT_POLICY
