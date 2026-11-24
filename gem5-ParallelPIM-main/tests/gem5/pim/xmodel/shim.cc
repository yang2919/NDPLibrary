/*
 * The reference model of the 18 PIM commands, behind pim_isa.h's
 * wrappers when built with PIM_BACKEND_REF.
 *
 * - Bank data is the region; Row(k)[i] is byte i of the addressed row of
 *   bank k, at its encoder address.
 * - Per unit (channel x rank or pseudo channel): the GB, one row. Per PU
 *   (one per bank): num_regs registers of one row and an accumulator of
 *   acc_fmt. EW results go to register 0.
 * - Element-wise results are computed in fp32 and rounded to num_fmt
 *   (RNE); integers wrap. MAC: per column command, the elements in range
 *   are multiplied and summed in acc_fmt in ascending element order, and
 *   added to the accumulator once.
 * - Vector forms cover vl = min(nbytes, VLMAX) bytes; their bytes must be
 *   consecutive bytes of one bank row (bank operations) or of one unit's
 *   row-sized GB or register window (non-bank operations).
 * - A software error aborts with a message, or, with
 *   errors_fatal = 0, is counted and the request does nothing (loads
 *   return zeros).
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ref/pim_ref.h"

#include <vector>

#include "mem/ramulator2/pim/pim_func.hh"
#include "mem/ramulator2/pim/pim_numfmt.hh"

/* Cross-check: libpim/src/ref/model.c with the command semantics
 * and the number formats replaced by gem5's PimFunctionalModel and
 * pim_numfmt. Address checks, counters and the flush log are model.c's. */
using gem5::PimOp;
namespace gp = gem5::pim;

static struct
{
    int attached;
    pim_ref_config_t cfg;
    uint8_t *base;
    uint64_t size;
    uint64_t row_bytes;
    uint32_t banks_per_unit;
    uint32_t units_total;
    size_t elem;
    size_t acc_bytes;
    gp::PimFunctionalModel *model;
    uint64_t errors;
    uint64_t count[PIM_OP_WR_REG + 1];
    uint64_t fences;
    uintptr_t *flush_log;
    size_t flush_n, flush_cap;
} m;

class RegionStore : public gp::PimBankStore
{
  public:
    uint8_t *
    column(unsigned unit, unsigned bank, uint32_t row, unsigned col) override
    {
        const pim_geom_t *g = &m.cfg.geom;
        pim_coord_t c = {};
        c.ch = unit / g->units;
        c.unit = unit % g->units;
        c.sid = bank / (g->bank_groups * g->banks_per_group);
        c.bg = (bank / g->banks_per_group) % g->bank_groups;
        c.bank = bank % g->banks_per_group;
        c.row = row;
        c.col = col;
        return m.base + pim_geom_encode(g, &c);
    }
};
static RegionStore store;

static const char *const op_names[] = {
    "none",
    "pim.sb.ewadd",
    "pim.sb.ewmul",
    "pim.sb.ewadd.gbuf",
    "pim.sb.ewmul.gbuf",
    "pim.sb.mac",
    "pim.sb.mov.bkgbuf",
    "pim.sb.mov.gbufbk",
    "pim.ab.ewadd",
    "pim.ab.ewmul",
    "pim.ab.ewadd.gbuf",
    "pim.ab.ewmul.gbuf",
    "pim.ab.mac",
    "pim.ab.mov.regbk",
    "pim.ab.mov.bkreg",
    "pim.rd.gb",
    "pim.wr.gb",
    "pim.rd.reg",
    "pim.wr.reg",
};

/* Report a software error; returns 1 (the request is then a no-op). */
static int
sw_error(enum pim_op op, uint64_t off, const char *fmt, ...)
{
    m.errors++;
    if (!m.cfg.errors_fatal) {
        return 1;
    }
    va_list ap;
    fprintf(stderr, "pim_ref: %s at offset %#llx: ", op_names[op],
            (unsigned long long)off);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}

int
pim_ref_attach(void *base, const pim_ref_config_t *cfg)
{
    pim_ref_detach();
    const pim_geom_t *g = &cfg->geom;
    m.cfg = *cfg;
    m.base = (uint8_t *)base;
    m.size = pim_geom_size(g);
    m.row_bytes = pim_geom_row_bytes(g);
    m.banks_per_unit = pim_geom_banks_per_unit(g);
    m.units_total = g->channels * g->units;
    m.elem = pim_ref_fmt_bytes(cfg->num_fmt);
    m.acc_bytes = pim_ref_fmt_bytes(cfg->acc_fmt);
    if (cfg->num_regs < 1 || cfg->vlenb == 0 || g->col_bytes % m.elem ||
        m.acc_bytes < m.elem ||
        pim_ref_fmt_is_float(cfg->acc_fmt) !=
            pim_ref_fmt_is_float(cfg->num_fmt)) {
        return -1;
    }
    gp::FuncConfig fc;
    fc.units = m.units_total;
    fc.banksPerUnit = m.banks_per_unit;
    fc.rowsPerBank = g->rows;
    fc.colsPerRow = g->cols;
    fc.colBytes = g->col_bytes;
    fc.numRegs = cfg->num_regs;
    fc.numFmt = gp::NumFmt(cfg->num_fmt);
    fc.accFmt = gp::NumFmt(cfg->acc_fmt);
    if (gp::PimFunctionalModel::checkConfig(fc)) {
        return -1;
    }
    m.model = new gp::PimFunctionalModel(fc, &store);
    m.attached = 1;
    return 0;
}

void
pim_ref_detach(void)
{
    delete m.model;
    free(m.flush_log);
    memset(&m, 0, sizeof(m));
}

const pim_ref_config_t *
pim_ref_config(void)
{ return m.attached ? &m.cfg : NULL; }

void
pim_ref_set_errors_fatal(int fatal)
{ m.cfg.errors_fatal = fatal; }

uint64_t
pim_ref_errors(void)
{ return m.errors; }

uint64_t
pim_ref_count(enum pim_op op)
{ return m.count[op]; }

uint64_t
pim_ref_fences(void)
{ return m.fences; }

const uintptr_t *
pim_ref_flush_log(size_t *n)
{
    *n = m.flush_n;
    return m.flush_log;
}

void
pim_ref_flush_log_clear(void)
{ m.flush_n = 0; }

void
pim_ref_flush(const void *line)
{
    if (m.flush_n == m.flush_cap) {
        m.flush_cap = m.flush_cap ? 2 * m.flush_cap : 1024;
        m.flush_log =
            (uintptr_t *)realloc(m.flush_log, m.flush_cap * sizeof(uintptr_t));
        if (!m.flush_log) {
            abort();
        }
    }
    m.flush_log[m.flush_n++] =
        (uintptr_t)line & ~(uintptr_t)(PIM_CACHE_LINE - 1);
}

void
pim_ref_fence(void)
{ m.fences++; }

size_t
pim_ref_vlmax(size_t nbytes)
{
    const size_t vlmax = 8 * (m.attached ? m.cfg.vlenb : 1);
    return nbytes < vlmax ? nbytes : vlmax;
}

/* State ---------------------------------------------------------------- */

static uint32_t
unit_index(const pim_coord_t *c)
{ return c->ch * m.cfg.geom.units + c->unit; }

static uint32_t
pu_index(const pim_coord_t *c)
{
    const pim_geom_t *g = &m.cfg.geom;
    return unit_index(c) * m.banks_per_unit +
           (c->sid * g->bank_groups + c->bg) * g->banks_per_group + c->bank;
}

uint8_t *
pim_ref_gb(uint32_t ch, uint32_t unit)
{ return m.model->gb(ch * m.cfg.geom.units + unit); }

uint8_t *
pim_ref_reg(const pim_coord_t *bank, uint32_t j)
{ return m.model->reg(unit_index(bank), pu_index(bank) - unit_index(bank) * m.banks_per_unit, j); }

uint8_t *
pim_ref_acc(const pim_coord_t *bank)
{ return m.model->acc(unit_index(bank), pu_index(bank) - unit_index(bank) * m.banks_per_unit); }

/* Byte i of the row of bank c in the region. */
static uint8_t *
row_byte(const pim_coord_t *c, uint64_t i)
{
    pim_coord_t p = *c;
    p.col = (uint32_t)(i / m.cfg.geom.col_bytes);
    return m.base + pim_geom_encode(&m.cfg.geom, &p) +
           i % m.cfg.geom.col_bytes;
}

static uint32_t
reg_row0(void)
{ return m.cfg.geom.rows - m.cfg.num_regs - 1; }

/* Arithmetic ----------------------------------------------------------- */

/* dst = a (+ or *) b, element-wise over row bytes [lo, hi). */
static void
ew(int mul, uint8_t *dst, const uint8_t *(*a)(uint64_t, const void *),
   const void *actx, const uint8_t *(*b)(uint64_t, const void *),
   const void *bctx, uint64_t lo, uint64_t hi)
{
    const uint32_t fmt = m.cfg.num_fmt;
    for (uint64_t i = lo; i < hi; i += m.elem) {
        const uint8_t *pa = a(i, actx);
        const uint8_t *pb = b(i, bctx);
        if (pim_ref_fmt_is_float(fmt)) {
            const float x = pim_ref_load_f(fmt, pa);
            const float y = pim_ref_load_f(fmt, pb);
            const float r = mul ? x * y : x + y;
            pim_ref_store_f(fmt, dst + i, r);
        } else {
            const int64_t x = pim_ref_load_i(fmt, pa);
            const int64_t y = pim_ref_load_i(fmt, pb);
            pim_ref_store_i(fmt, dst + i,
                            mul ? (int64_t)((uint64_t)x * (uint64_t)y)
                                : x + y);
        }
    }
}

/* acc += sum of gb[i] * row[i], one update per column of [lo, hi). */
static void
mac(uint8_t *acc, const uint8_t *gb, const pim_coord_t *bank, uint64_t lo,
    uint64_t hi)
{
    const uint32_t nf = m.cfg.num_fmt, af = m.cfg.acc_fmt;
    const uint64_t cb = m.cfg.geom.col_bytes;
    uint64_t i = lo;
    while (i < hi) {
        const uint64_t end = (i / cb + 1) * cb < hi ? (i / cb + 1) * cb : hi;
        if (pim_ref_fmt_is_float(nf)) {
            float partial = 0.0f;
            int first = 1;
            for (; i < end; i += m.elem) {
                const float x = pim_ref_to_float(
                    af, pim_ref_from_float(af, pim_ref_load_f(nf, gb + i)));
                const float y = pim_ref_to_float(
                    af, pim_ref_from_float(
                            af, pim_ref_load_f(nf, row_byte(bank, i))));
                const float p =
                    pim_ref_to_float(af, pim_ref_from_float(af, x * y));
                partial = first ? p
                                : pim_ref_to_float(
                                      af, pim_ref_from_float(af, partial + p));
                first = 0;
            }
            const float a = pim_ref_load_f(af, acc);
            pim_ref_store_f(af, acc, a + partial);
        } else {
            uint64_t partial = 0;
            for (; i < end; i += m.elem) {
                const int64_t x = pim_ref_load_i(nf, gb + i);
                const int64_t y = pim_ref_load_i(nf, row_byte(bank, i));
                partial += (uint64_t)x * (uint64_t)y;
            }
            pim_ref_store_i(af, acc,
                            pim_ref_load_i(af, acc) + (int64_t)partial);
        }
    }
}

/* Operand accessors for ew(). */
static const uint8_t *
acc_row(uint64_t i, const void *ctx)
{ return row_byte((const pim_coord_t *)ctx, i); }

static const uint8_t *
acc_buf(uint64_t i, const void *ctx)
{ return (const uint8_t *)ctx + i; }

/* Commands ------------------------------------------------------------- */

static int
is_ew(enum pim_op op)
{
    return op == PIM_OP_SB_EWADD || op == PIM_OP_SB_EWMUL ||
           op == PIM_OP_AB_EWADD || op == PIM_OP_AB_EWMUL;
}

static int
writes_reg(enum pim_op op)
{
    return (op >= PIM_OP_SB_EWADD && op <= PIM_OP_SB_EWMUL_GBUF) ||
           (op >= PIM_OP_AB_EWADD && op <= PIM_OP_AB_EWMUL_GBUF) ||
           op == PIM_OP_AB_MOV_REGBK || op == PIM_OP_AB_MOV_BKREG;
}

/* One bank command on row bytes [lo, hi) of bank b (or every bank). */
static void
bank_command(enum pim_op op, const pim_coord_t *b, uint64_t lo, uint64_t hi,
             uint64_t off)
{
    const unsigned unit = unit_index(b);
    const unsigned bank = pu_index(b) - unit * m.banks_per_unit;
    if (const char *why = m.model->checkBank(PimOp(op), bank, b->row, lo, hi)) {
        sw_error(op, off, "%s", why);
        return;
    }
    m.model->bankOp(PimOp(op), unit, bank, b->row, lo, hi);
}

void
pim_ref_scalar(enum pim_op op, void *addr, uint64_t desc)
{
    const uint64_t off = (uint64_t)((uint8_t *)addr - m.base);
    if (!m.attached || (uint8_t *)addr < m.base || off >= m.size) {
        sw_error(op, off, "address outside the PIM memory");
        return;
    }
    m.count[op]++;
    if (off & 7) {
        sw_error(op, off, "address not 8-byte aligned");
        return;
    }
    if (desc >> 16) {
        sw_error(op, off, "reserved descriptor bits in %#llx",
                 (unsigned long long)desc);
        return;
    }
    pim_coord_t c;
    pim_geom_decode(&m.cfg.geom, off, &c);
    const uint32_t cols = m.cfg.geom.cols;
    const uint32_t n =
        (uint32_t)(desc & 0xffff) ? (uint32_t)(desc & 0xffff) : cols - c.col;
    if (c.col + n > cols) {
        sw_error(op, off, "columns %u + %u beyond the row", c.col, n);
        return;
    }
    const uint64_t cb = m.cfg.geom.col_bytes;
    bank_command(op, &c, c.col * cb, (uint64_t)(c.col + n) * cb, off);
}

/*
 * The row bytes [lo, hi) a vector request covers. Bank operations need
 * consecutive bytes of one bank row (a range that crosses a row, bank or
 * unit is an error); the GB operations take the bytes from the first one
 * on. Returns non-zero on error.
 */
static int
vector_range(enum pim_op op, uint64_t off, size_t vl, pim_coord_t *c,
             uint64_t *lo, uint64_t *hi)
{
    const uint64_t cb = m.cfg.geom.col_bytes;
    const uint32_t byte = pim_geom_decode(&m.cfg.geom, off, c);
    *lo = (uint64_t)c->col * cb + byte;
    *hi = *lo + vl;
    if (*hi > m.row_bytes) {
        return sw_error(op, off, "%zu bytes cross the end of the row", vl);
    }
    if (op > PIM_OP_AB_MOV_BKREG) {
        return 0;
    }
    for (size_t i = cb - byte; i < vl; i += cb) {
        pim_coord_t d;
        pim_geom_decode(&m.cfg.geom, off + i, &d);
        if (d.ch != c->ch || d.unit != c->unit || d.sid != c->sid ||
            d.bg != c->bg || d.bank != c->bank || d.row != c->row ||
            (uint64_t)d.col * cb != *lo + i) {
            return sw_error(op, off,
                            "byte %zu is not the next byte of the bank row "
                            "(the range crosses a row, bank or unit)",
                            i);
        }
    }
    return 0;
}

/* The window coordinates of offset off; 0 outside the window. */
static int
window_at(uint64_t off, unsigned *unit, unsigned *bank, uint32_t *row,
          uint64_t *i)
{
    pim_coord_t d;
    const uint32_t byte = pim_geom_decode(&m.cfg.geom, off, &d);
    *i = (uint64_t)d.col * m.cfg.geom.col_bytes + byte;
    *unit = unit_index(&d);
    *bank = pu_index(&d) - *unit * m.banks_per_unit;
    *row = d.row;
    return m.model->isWindowRow(d.row);
}

size_t
pim_ref_vector(enum pim_op op, void *addr, size_t nbytes, const void *src,
               void *dst)
{
    const size_t vl = pim_ref_vlmax(nbytes);
    const uint64_t off = (uint64_t)((uint8_t *)addr - m.base);
    if (vl == 0) {
        return 0; /* no request */
    }
    if (dst) {
        memset(dst, 0, vl); /* what a failed load returns */
    }
    if (!m.attached || (uint8_t *)addr < m.base || off + vl > m.size) {
        sw_error(op, off, "address outside the PIM memory");
        return vl;
    }
    m.count[op]++;
    if (op == PIM_OP_RD_REG || op == PIM_OP_WR_REG) {
        /* Burst by burst through the window. */
        unsigned u, k;
        uint32_t row;
        uint64_t at;
        for (size_t i = 0; i < vl; i++) {
            if (!window_at(off + i, &u, &k, &row, &at)) {
                sw_error(op, off + i, "outside the register window");
                return vl;
            }
        }
        for (size_t i = 0; i < vl; i++) {
            window_at(off + i, &u, &k, &row, &at);
            if (op == PIM_OP_RD_REG) {
                m.model->readWindow(u, k, row, at, (uint8_t *)dst + i, 1);
            } else {
                m.model->writeWindow(u, k, row, at, (const uint8_t *)src + i, 1);
            }
        }
        return vl;
    }
    pim_coord_t c;
    uint64_t lo, hi;
    if (vector_range(op, off, vl, &c, &lo, &hi)) {
        return vl;
    }
    if (op <= PIM_OP_AB_MOV_BKREG) {
        bank_command(op, &c, lo, hi, off);
    } else if (op == PIM_OP_RD_GB) {
        m.model->readGb(unit_index(&c), lo, (uint8_t *)dst, vl);
    } else {
        m.model->writeGb(unit_index(&c), lo, (const uint8_t *)src, vl);
    }
    return vl;
}

/* Number formats through gem5's pim_numfmt. */
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
size_t pim_ref_fmt_bytes(uint32_t fmt) { return gp::fmtBytes(gp::NumFmt(fmt)); }
int pim_ref_fmt_is_float(uint32_t fmt) { return gp::fmtIsFloat(gp::NumFmt(fmt)); }
float pim_ref_to_float(uint32_t fmt, uint32_t bits) { return u2f(gp::toF32(gp::NumFmt(fmt), bits)); }
uint32_t pim_ref_from_float(uint32_t fmt, float v) { return gp::fromF32(gp::NumFmt(fmt), f2u(v)); }
float pim_ref_load_f(uint32_t fmt, const uint8_t *p) { return u2f(gp::toF32(gp::NumFmt(fmt), gp::loadBits(gp::NumFmt(fmt), p))); }
int64_t pim_ref_load_i(uint32_t fmt, const uint8_t *p) { return gp::loadInt(gp::NumFmt(fmt), p); }
void pim_ref_store_f(uint32_t fmt, uint8_t *p, float v) { gp::storeBits(gp::NumFmt(fmt), p, gp::fromF32(gp::NumFmt(fmt), f2u(v))); }
void pim_ref_store_i(uint32_t fmt, uint8_t *p, int64_t v) { gp::storeInt(gp::NumFmt(fmt), p, v); }
