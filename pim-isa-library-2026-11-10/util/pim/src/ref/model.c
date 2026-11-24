/*
 * The reference model of the 18 PIM commands (04-memctrl-plan.md
 * section 9.4), behind pim_isa.h's wrappers when built with
 * PIM_BACKEND_REF.
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
 * - A software error (section 8.5) aborts with a message, or, with
 *   errors_fatal = 0, is counted and the request does nothing (loads
 *   return zeros).
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ref/pim_ref.h"

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
    uint8_t *gb;  /* units_total x row_bytes */
    uint8_t *reg; /* PUs x num_regs x row_bytes */
    uint8_t *acc; /* PUs x 8 */
    uint64_t errors;
    uint64_t count[PIM_OP_WR_REG + 1];
    uint64_t fences;
    uintptr_t *flush_log;
    size_t flush_n, flush_cap;
    /* gem5's counts for the same program (P-LIB step L7, LQ14 (a)). */
    uint64_t g5_reqs[PIM_OP_WR_REG + 1];
    uint64_t *g5_cols; /* channels x PIM_REF_CLASSES */
} m;

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
    const size_t pus = (size_t)m.units_total * m.banks_per_unit;
    m.gb = (uint8_t *)calloc(m.units_total, m.row_bytes);
    m.reg = (uint8_t *)calloc(pus * cfg->num_regs, m.row_bytes);
    m.acc = (uint8_t *)calloc(pus, 8);
    m.g5_cols = (uint64_t *)calloc((size_t)g->channels * 11, sizeof(uint64_t));
    if (!m.gb || !m.reg || !m.acc || !m.g5_cols) {
        pim_ref_detach();
        return -1;
    }
    m.attached = 1;
    return 0;
}

static void g5_write(void);

void
pim_ref_detach(void)
{
    g5_write();
    free(m.g5_cols);
    free(m.gb);
    free(m.reg);
    free(m.acc);
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
{ return m.gb + (size_t)(ch * m.cfg.geom.units + unit) * m.row_bytes; }

uint8_t *
pim_ref_reg(const pim_coord_t *bank, uint32_t j)
{ return m.reg + ((size_t)pu_index(bank) * m.cfg.num_regs + j) * m.row_bytes; }

uint8_t *
pim_ref_acc(const pim_coord_t *bank)
{ return m.acc + (size_t)pu_index(bank) * 8; }

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
    const pim_geom_t *g = &m.cfg.geom;
    const int ab = op >= PIM_OP_AB_EWADD && op <= PIM_OP_AB_MOV_BKREG;
    if (b->row >= reg_row0()) {
        sw_error(op, off, "bank operation on window row %u", b->row);
        return;
    }
    if (is_ew(op) && (b->bank & 1)) {
        sw_error(op, off, "odd bank %u of an element-wise pair", b->bank);
        return;
    }
    if (writes_reg(op) && m.cfg.num_regs > 1) {
        sw_error(op, off, "needs one register per PU, not %u", m.cfg.num_regs);
        return;
    }
    const int compute = (op >= PIM_OP_SB_EWADD && op <= PIM_OP_SB_MAC) ||
                        (op >= PIM_OP_AB_EWADD && op <= PIM_OP_AB_MAC);
    if (compute && (lo % m.elem || hi % m.elem)) {
        sw_error(op, off, "range [%llu, %llu) is not element-aligned",
                 (unsigned long long)lo, (unsigned long long)hi);
        return;
    }
    uint8_t *gb = pim_ref_gb(b->ch, b->unit);
    pim_coord_t k = *b;
    for (k.sid = ab ? 0 : b->sid; k.sid < (ab ? g->sids : b->sid + 1);
         k.sid++) {
        for (k.bg = ab ? 0 : b->bg; k.bg < (ab ? g->bank_groups : b->bg + 1);
             k.bg++) {
            for (k.bank = ab ? 0 : b->bank;
                 k.bank < (ab ? g->banks_per_group : b->bank + 1); k.bank++) {
                uint8_t *reg0 = pim_ref_reg(&k, 0);
                pim_coord_t pair = k;
                pair.bank = k.bank + 1;
                switch (op) {
                    case PIM_OP_SB_EWADD:
                    case PIM_OP_SB_EWMUL:
                    case PIM_OP_AB_EWADD:
                    case PIM_OP_AB_EWMUL:
                        if (k.bank & 1) {
                            break; /* the even bank of each pair computes */
                        }
                        ew(op == PIM_OP_SB_EWMUL || op == PIM_OP_AB_EWMUL,
                           reg0, acc_row, &k, acc_row, &pair, lo, hi);
                        break;
                    case PIM_OP_SB_EWADD_GBUF:
                    case PIM_OP_SB_EWMUL_GBUF:
                    case PIM_OP_AB_EWADD_GBUF:
                    case PIM_OP_AB_EWMUL_GBUF:
                        ew(op == PIM_OP_SB_EWMUL_GBUF ||
                               op == PIM_OP_AB_EWMUL_GBUF,
                           reg0, acc_buf, gb, acc_row, &k, lo, hi);
                        break;
                    case PIM_OP_SB_MAC:
                    case PIM_OP_AB_MAC:
                        mac(pim_ref_acc(&k), gb, &k, lo, hi);
                        break;
                    case PIM_OP_SB_MOV_BKGBUF:
                        for (uint64_t i = lo; i < hi; i++) {
                            gb[i] = *row_byte(&k, i);
                        }
                        break;
                    case PIM_OP_SB_MOV_GBUFBK:
                        for (uint64_t i = lo; i < hi; i++) {
                            *row_byte(&k, i) = gb[i];
                        }
                        break;
                    case PIM_OP_AB_MOV_REGBK:
                        for (uint64_t i = lo; i < hi; i++) {
                            *row_byte(&k, i) = reg0[i];
                        }
                        break;
                    case PIM_OP_AB_MOV_BKREG:
                        for (uint64_t i = lo; i < hi; i++) {
                            reg0[i] = *row_byte(&k, i);
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
}

/*
 * gem5's request and column-command counts (P-LIB step L7, LQ14 (a)).
 *
 * A vector PIM instruction becomes one request per vector register
 * (VLENB bytes) and the CPU splits each again at 64-byte cache lines
 * (02-isa-plan.md section 4.3); a scalar one is one request. Each request
 * is one job of the PIM memory, which issues one column command per column
 * it covers (src/mem/ramulator2/pim/pim_layout.cc's decode; all-bank modes
 * other than AESPA), counted per channel and timing class
 * (pim_col_cmds_<class>, ext/ramulator2/src/ramulator/pim/pim_sequencer.cpp).
 * The class of each operation is PimRamulator2::classOf's.
 */
static const char *const g5_op_names[] = {
    "None",        "SbEwadd",     "SbEwmul",     "SbEwaddGbuf", "SbEwmulGbuf",
    "SbMac",       "SbMovBkgbuf", "SbMovGbufbk", "AbEwadd",     "AbEwmul",
    "AbEwaddGbuf", "AbEwmulGbuf", "AbMac",       "AbMovRegbk",  "AbMovBkreg",
    "RdGb",        "WrGb",        "RdReg",       "WrReg",
};

#define PIM_REF_CLASSES 11
static const char *const g5_class_names[PIM_REF_CLASSES] = {
    "SbRd",    "SbRd2", "SbWr", "AbEw",  "AbMac", "AbBkreg",
    "AbRegbk", "GbRd",  "GbWr", "RegRd", "RegWr"};

static int
g5_class(enum pim_op op)
{
    switch (op) {
        case PIM_OP_SB_EWADD:
        case PIM_OP_SB_EWMUL:
            return 1; /* SbRd2 */
        case PIM_OP_SB_EWADD_GBUF:
        case PIM_OP_SB_EWMUL_GBUF:
        case PIM_OP_SB_MAC:
        case PIM_OP_SB_MOV_BKGBUF:
            return 0; /* SbRd */
        case PIM_OP_SB_MOV_GBUFBK:
            return 2; /* SbWr */
        case PIM_OP_AB_EWADD:
        case PIM_OP_AB_EWMUL:
        case PIM_OP_AB_EWADD_GBUF:
        case PIM_OP_AB_EWMUL_GBUF:
            return 3; /* AbEw */
        case PIM_OP_AB_MAC:
            return 4;
        case PIM_OP_AB_MOV_BKREG:
            return 5;
        case PIM_OP_AB_MOV_REGBK:
            return 6;
        case PIM_OP_RD_GB:
            return 7;
        case PIM_OP_WR_GB:
            return 8;
        case PIM_OP_RD_REG:
            return 9;
        default:
            return 10; /* WrReg */
    }
}

static void
g5_count(enum pim_op op, uint32_t ch, uint64_t ncols)
{
    m.g5_reqs[op]++;
    if (m.g5_cols) {
        m.g5_cols[(size_t)ch * PIM_REF_CLASSES + g5_class(op)] += ncols;
    }
}

/* The requests of a vector instruction of vl bytes at offset off. */
static void
g5_vector(enum pim_op op, uint64_t off, size_t vl)
{
    const uint64_t cb = m.cfg.geom.col_bytes;
    for (size_t k = 0; k < vl; k += m.cfg.vlenb) {
        const uint64_t uend =
            off + (k + m.cfg.vlenb < vl ? k + m.cfg.vlenb : vl);
        for (uint64_t a = off + k; a < uend;) {
            const uint64_t line_end = (a / 64 + 1) * 64;
            const uint64_t end = line_end < uend ? line_end : uend;
            pim_coord_t c;
            uint32_t byte = pim_geom_decode(&m.cfg.geom, a, &c);
            uint64_t ncols;
            if (op == PIM_OP_RD_REG || op == PIM_OP_WR_REG) {
                /* One column at a time through the window. */
                ncols = 0;
                for (uint64_t i = a; i < end;) {
                    pim_coord_t d;
                    const uint32_t b = pim_geom_decode(&m.cfg.geom, i, &d);
                    i += (end - i) < cb - b ? (end - i) : cb - b;
                    ncols++;
                }
            } else {
                const uint64_t lo = (uint64_t)c.col * cb + byte;
                ncols = (lo + (end - a) + cb - 1) / cb - lo / cb;
            }
            g5_count(op, c.ch, ncols);
            a = end;
        }
    }
}

/* Write the counts to the file named by PIM_REF_COUNTS, if any. */
static void
g5_write(void)
{
    const char *path = getenv("PIM_REF_COUNTS");
    if (!path || !*path || !m.attached) {
        return;
    }
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        return;
    }
    fprintf(f, "vlenb %zu\n", m.cfg.vlenb);
    for (int op = PIM_OP_SB_EWADD; op <= PIM_OP_WR_REG; op++) {
        fprintf(f, "reqs %s %llu\n", g5_op_names[op],
                (unsigned long long)m.g5_reqs[op]);
    }
    for (uint32_t ch = 0; ch < m.cfg.geom.channels; ch++) {
        for (int k = 0; k < PIM_REF_CLASSES; k++) {
            fprintf(f, "cols %u %s %llu\n", ch, g5_class_names[k],
                    (unsigned long long)
                        m.g5_cols[(size_t)ch * PIM_REF_CLASSES + k]);
        }
    }
    fclose(f);
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
    g5_count(op, c.ch, n);
    bank_command(op, &c, c.col * cb, (uint64_t)(c.col + n) * cb, off);
}

/*
 * The row bytes [lo, hi) a vector request covers. Bank operations need
 * consecutive bytes of one bank row (section 8.5: a range that crosses a
 * row, bank or unit is an error); the GB operations take the bytes from
 * the first one on (section 9.4 rows 15-16). Returns non-zero on error.
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

/* The register-window byte at offset off, or NULL outside the window;
 * *zero is set for accumulator-row bytes past the accumulator. */
static uint8_t *
window_byte(uint64_t off, int *zero)
{
    pim_coord_t d;
    const uint32_t byte = pim_geom_decode(&m.cfg.geom, off, &d);
    const uint64_t i = (uint64_t)d.col * m.cfg.geom.col_bytes + byte;
    const uint32_t r0 = reg_row0();
    *zero = 0;
    if (d.row < r0) {
        return NULL;
    }
    if (d.row < r0 + m.cfg.num_regs) {
        return pim_ref_reg(&d, d.row - r0) + i;
    }
    if (i < m.acc_bytes) {
        return pim_ref_acc(&d) + i;
    }
    *zero = 1;
    return NULL;
}

size_t
pim_ref_vector(enum pim_op op, void *addr, size_t nbytes, const void *src,
               void *dst)
{
    const size_t vl = pim_ref_vlmax(nbytes);
    const uint64_t off = (uint64_t)((uint8_t *)addr - m.base);
    if (vl == 0) {
        return 0; /* no request (P2 section 4.1) */
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
        /* Burst by burst through the window (section 8.3). */
        int zero;
        for (size_t i = 0; i < vl; i++) {
            if (!window_byte(off + i, &zero) && !zero) {
                sw_error(op, off + i, "outside the register window");
                return vl;
            }
        }
        g5_vector(op, off, vl);
        for (size_t i = 0; i < vl; i++) {
            uint8_t *p = window_byte(off + i, &zero);
            if (op == PIM_OP_RD_REG) {
                ((uint8_t *)dst)[i] = p ? *p : 0;
            } else if (p) {
                *p = ((const uint8_t *)src)[i];
            }
        }
        return vl;
    }
    pim_coord_t c;
    uint64_t lo, hi;
    if (vector_range(op, off, vl, &c, &lo, &hi)) {
        return vl;
    }
    g5_vector(op, off, vl);
    if (op <= PIM_OP_AB_MOV_BKREG) {
        bank_command(op, &c, lo, hi, off);
    } else if (op == PIM_OP_RD_GB) {
        memcpy(dst, pim_ref_gb(c.ch, c.unit) + lo, vl);
    } else {
        memcpy(pim_ref_gb(c.ch, c.unit) + lo, src, vl);
    }
    return vl;
}
