/*
 * L3: the reference model (src/ref) against hand-computed results.
 *
 *   test_model [--dump] vectors.txt ...
 *
 * Runs the test vectors of tests/vectors/ (format: tests/vectors/
 * README.md) and checks every "expect" line; with --dump it prints the
 * actual values of every "expect" line instead, the output the
 * cross-check with tests/gem5/pim/pim_ref.py compares (P-LIB section 8).
 * Without arguments it runs the built-in checks of the number formats,
 * vl, flushes and fences.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "pim_test.h"
#include "ref/pim_ref.h"

static const char *const fmt_names[] = {"bf16", "fp16",  "fp32",
                                        "int8", "int16", "int32"};

static int
fmt_of(const char *s)
{
    for (int i = 0; i < 6; i++) {
        if (strcmp(s, fmt_names[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static const struct
{
    const char *name;
    enum pim_op op;
} ops[] = {
    {"pim.sb.ewadd", PIM_OP_SB_EWADD},
    {"pim.sb.ewmul", PIM_OP_SB_EWMUL},
    {"pim.sb.ewadd.gbuf", PIM_OP_SB_EWADD_GBUF},
    {"pim.sb.ewmul.gbuf", PIM_OP_SB_EWMUL_GBUF},
    {"pim.sb.mac", PIM_OP_SB_MAC},
    {"pim.sb.mov.bkgbuf", PIM_OP_SB_MOV_BKGBUF},
    {"pim.sb.mov.gbufbk", PIM_OP_SB_MOV_GBUFBK},
    {"pim.ab.ewadd", PIM_OP_AB_EWADD},
    {"pim.ab.ewmul", PIM_OP_AB_EWMUL},
    {"pim.ab.ewadd.gbuf", PIM_OP_AB_EWADD_GBUF},
    {"pim.ab.ewmul.gbuf", PIM_OP_AB_EWMUL_GBUF},
    {"pim.ab.mac", PIM_OP_AB_MAC},
    {"pim.ab.mov.regbk", PIM_OP_AB_MOV_REGBK},
    {"pim.ab.mov.bkreg", PIM_OP_AB_MOV_BKREG},
    {"pim.rd.gb", PIM_OP_RD_GB},
    {"pim.wr.gb", PIM_OP_WR_GB},
    {"pim.rd.reg", PIM_OP_RD_REG},
    {"pim.wr.reg", PIM_OP_WR_REG},
};

static int
op_of(const char *s)
{
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
        if (strcmp(s, ops[i].name) == 0) {
            return (int)ops[i].op;
        }
    }
    return -1;
}

/* The runner's state. */
static struct
{
    pim_ref_config_t cfg;
    uint8_t *region;
    uint64_t size;
    uint8_t last_read[4096];
    size_t last_read_n;
    int dump;
    const char *file;
    int line;
} r;

#define MAX_TOK 600

static void
die(const char *msg, const char *tok)
{
    fprintf(stderr, "%s:%d: %s%s%s\n", r.file, r.line, msg, tok ? ": " : "",
            tok ? tok : "");
    exit(2);
}

static void
attach(void)
{
    if (r.region) {
        munmap(r.region, r.size);
    }
    r.size = pim_geom_size(&r.cfg.geom);
    r.region =
        (uint8_t *)mmap(NULL, r.size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r.region == MAP_FAILED || pim_ref_attach(r.region, &r.cfg)) {
        die("cannot start the model for this geometry", NULL);
    }
}

/* key=value; returns 1 if tok has the key. */
static int
kv(const char *tok, const char *key, const char **val)
{
    const size_t n = strlen(key);
    if (strncmp(tok, key, n) == 0 && tok[n] == '=') {
        *val = tok + n + 1;
        return 1;
    }
    return 0;
}

static uint32_t
reg_row0(void)
{ return r.cfg.geom.rows - r.cfg.num_regs - 1; }

/* Coordinates from key=value tokens; row=reg<j> / row=acc for the window.
 * Returns the index of the first token that is not a coordinate. */
static int
parse_coord(char **tok, int n, int first, pim_coord_t *c, uint32_t *byte,
            uint32_t *j, uint64_t *count, int *count_is_cols)
{
    memset(c, 0, sizeof(*c));
    *byte = 0;
    if (j) {
        *j = 0;
    }
    if (count) {
        *count = 0;
        *count_is_cols = 0;
    }
    int i;
    for (i = first; i < n; i++) {
        const char *v;
        if (kv(tok[i], "ch", &v)) {
            c->ch = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "unit", &v)) {
            c->unit = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "sid", &v)) {
            c->sid = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "bg", &v)) {
            c->bg = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "bank", &v)) {
            c->bank = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "row", &v)) {
            if (strncmp(v, "reg", 3) == 0) {
                c->row = reg_row0() + (uint32_t)strtoul(v + 3, NULL, 0);
            } else if (strcmp(v, "acc") == 0) {
                c->row = reg_row0() + r.cfg.num_regs;
            } else {
                c->row = (uint32_t)strtoul(v, NULL, 0);
            }
        } else if (kv(tok[i], "col", &v)) {
            c->col = (uint32_t)strtoul(v, NULL, 0);
        } else if (kv(tok[i], "byte", &v)) {
            *byte = (uint32_t)strtoul(v, NULL, 0);
        } else if (j && kv(tok[i], "j", &v)) {
            *j = (uint32_t)strtoul(v, NULL, 0);
        } else if (count && kv(tok[i], "bytes", &v)) {
            *count = strtoull(v, NULL, 0);
        } else if (count && kv(tok[i], "ncols", &v)) {
            *count = strtoull(v, NULL, 0);
            *count_is_cols = 1;
        } else {
            break;
        }
    }
    return i;
}

/* "<fmt> v0 v1 ..." into bytes; returns the byte count. */
static size_t
parse_values(char **tok, int n, int first, uint8_t *out, size_t cap)
{
    if (first >= n) {
        return 0;
    }
    const int fmt = fmt_of(tok[first]);
    if (fmt < 0) {
        die("unknown format", tok[first]);
    }
    const size_t e = pim_ref_fmt_bytes((uint32_t)fmt);
    size_t len = 0;
    for (int i = first + 1; i < n; i++) {
        if (len + e > cap) {
            die("too many values", NULL);
        }
        if (strncmp(tok[i], "0x", 2) == 0) {
            const uint64_t bits = strtoull(tok[i], NULL, 16);
            for (size_t k = 0; k < e; k++) {
                out[len + k] = (uint8_t)(bits >> (8 * k));
            }
        } else if (pim_ref_fmt_is_float((uint32_t)fmt)) {
            pim_ref_store_f((uint32_t)fmt, out + len,
                            (float)strtod(tok[i], NULL));
        } else {
            pim_ref_store_i((uint32_t)fmt, out + len,
                            strtoll(tok[i], NULL, 0));
        }
        len += e;
    }
    return len;
}

static void
print_bytes(FILE *out, int fmt, const uint8_t *p, size_t n)
{
    const size_t e = pim_ref_fmt_bytes((uint32_t)fmt);
    fprintf(out, "%s", fmt_names[fmt]);
    for (size_t i = 0; i + e <= n; i += e) {
        uint64_t bits = 0;
        for (size_t k = 0; k < e; k++) {
            bits |= (uint64_t)p[i + k] << (8 * k);
        }
        fprintf(out, " 0x%0*llx", (int)(2 * e), (unsigned long long)bits);
    }
}

/* Byte i of the row of bank c, in the region. */
static uint8_t *
row_ptr(const pim_coord_t *c, uint64_t i)
{
    pim_coord_t p = *c;
    p.col = (uint32_t)(i / r.cfg.geom.col_bytes);
    return r.region + pim_geom_encode(&r.cfg.geom, &p) +
           i % r.cfg.geom.col_bytes;
}

/* Where a state statement (row, gb, reg, acc) points: fills ptrs. */
static int
state_ptrs(const char *kind, char **tok, int n, uint8_t **ptrs, size_t cap,
           int *next)
{
    pim_coord_t c;
    uint32_t byte, j;
    *next = parse_coord(tok, n, 1, &c, &byte, &j, NULL, NULL);
    const uint64_t cb = r.cfg.geom.col_bytes;
    const uint64_t start = c.col * cb + byte;
    if (strcmp(kind, "row") == 0) {
        for (size_t i = 0; i < cap; i++) {
            ptrs[i] = row_ptr(&c, start + i);
        }
    } else if (strcmp(kind, "gb") == 0) {
        for (size_t i = 0; i < cap; i++) {
            ptrs[i] = pim_ref_gb(c.ch, c.unit) + start + i;
        }
    } else if (strcmp(kind, "reg") == 0) {
        for (size_t i = 0; i < cap; i++) {
            ptrs[i] = pim_ref_reg(&c, j) + start + i;
        }
    } else if (strcmp(kind, "acc") == 0) {
        for (size_t i = 0; i < cap && i < 8; i++) {
            ptrs[i] = pim_ref_acc(&c) + i;
        }
    } else {
        return -1;
    }
    return 0;
}

static void
statement(char **tok, int n, char *text)
{
    const char *v;
    uint8_t buf[4096];
    uint8_t *ptrs[4096];
    if (strcmp(tok[0], "geom") == 0) {
        pim_ref_config_t *cfg = &r.cfg;
        memset(cfg, 0, sizeof(*cfg));
        cfg->num_regs = 1;
        cfg->num_fmt = PIM_REF_BF16;
        cfg->acc_fmt = PIM_REF_FP32;
        cfg->errors_fatal = 1;
        cfg->geom.units = cfg->geom.sids = cfg->geom.bank_groups = 1;
        for (int i = 1; i < n; i++) {
            pim_geom_t *g = &cfg->geom;
            if (kv(tok[i], "policy", &v)) {
                g->policy = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "channels", &v)) {
                g->channels = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "intlv", &v)) {
                g->intlv_bytes = strtoull(v, NULL, 0);
            } else if (kv(tok[i], "units", &v)) {
                g->units = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "sids", &v)) {
                g->sids = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "bgs", &v)) {
                g->bank_groups = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "banks", &v)) {
                g->banks_per_group = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "rows", &v)) {
                g->rows = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "cols", &v)) {
                g->cols = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "col_bytes", &v)) {
                g->col_bytes = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "num_regs", &v)) {
                cfg->num_regs = (uint32_t)strtoul(v, NULL, 0);
            } else if (kv(tok[i], "num_fmt", &v)) {
                cfg->num_fmt = (uint32_t)fmt_of(v);
            } else if (kv(tok[i], "acc_fmt", &v)) {
                cfg->acc_fmt = (uint32_t)fmt_of(v);
            } else if (kv(tok[i], "vlenb", &v)) {
                cfg->vlenb = strtoull(v, NULL, 0);
            } else {
                die("unknown geom key", tok[i]);
            }
        }
        if (!cfg->vlenb) {
            cfg->vlenb = cfg->geom.col_bytes;
        }
        attach();
        return;
    }
    if (!r.region) {
        die("no geom line before", tok[0]);
    }
    if (strcmp(tok[0], "errors") == 0 && n == 2) {
        r.cfg.errors_fatal = strcmp(tok[1], "fatal") == 0;
        pim_ref_set_errors_fatal(r.cfg.errors_fatal);
        return;
    }
    if (strcmp(tok[0], "row") == 0 || strcmp(tok[0], "gb") == 0 ||
        strcmp(tok[0], "reg") == 0 || strcmp(tok[0], "acc") == 0) {
        int next;
        pim_coord_t c;
        uint32_t byte, j;
        next = parse_coord(tok, n, 1, &c, &byte, &j, NULL, NULL);
        const size_t len = parse_values(tok, n, next, buf, sizeof(buf));
        state_ptrs(tok[0], tok, n, ptrs, len, &next);
        for (size_t i = 0; i < len; i++) {
            *ptrs[i] = buf[i];
        }
        return;
    }
    if (strcmp(tok[0], "cmd") == 0 && n >= 2) {
        const int op = op_of(tok[1]);
        if (op < 0) {
            die("unknown command", tok[1]);
        }
        pim_coord_t c;
        uint32_t byte;
        uint64_t count;
        int is_cols;
        const int next =
            parse_coord(tok, n, 2, &c, &byte, NULL, &count, &is_cols);
        const size_t len = parse_values(tok, n, next, buf, sizeof(buf));
        uint8_t *addr = r.region + pim_geom_encode(&r.cfg.geom, &c) + byte;
        if (op <= PIM_OP_SB_MAC) {
            pim_ref_scalar((enum pim_op)op, addr, count);
        } else if (op == PIM_OP_RD_GB || op == PIM_OP_RD_REG) {
            r.last_read_n = pim_ref_vector((enum pim_op)op, addr, count, NULL,
                                           r.last_read);
        } else {
            pim_ref_vector((enum pim_op)op, addr, len ? len : count,
                           len ? buf : NULL, NULL);
        }
        return;
    }
    if (strcmp(tok[0], "expect") == 0 && n >= 2) {
        if (strcmp(tok[1], "errors") == 0 && n == 3) {
            const uint64_t want = strtoull(tok[2], NULL, 0);
            if (r.dump) {
                printf("%s => %llu\n", text,
                       (unsigned long long)pim_ref_errors());
                return;
            }
            CHECK(pim_ref_errors() == want, "%s:%d: errors %llu, want %llu",
                  r.file, r.line, (unsigned long long)pim_ref_errors(),
                  (unsigned long long)want);
            return;
        }
        if (strcmp(tok[1], "count") == 0 && n == 4) {
            const int op = op_of(tok[2]);
            if (op < 0) {
                die("unknown command", tok[2]);
            }
            const uint64_t got = pim_ref_count((enum pim_op)op);
            if (r.dump) {
                printf("%s => %llu\n", text, (unsigned long long)got);
                return;
            }
            CHECK(got == strtoull(tok[3], NULL, 0), "%s:%d: count %llu",
                  r.file, r.line, (unsigned long long)got);
            return;
        }
        int next;
        const uint8_t *actual[4096];
        size_t len;
        if (strcmp(tok[1], "read") == 0) {
            next = 2;
            len = parse_values(tok, n, next, buf, sizeof(buf));
            for (size_t i = 0; i < len; i++) {
                actual[i] = i < r.last_read_n ? &r.last_read[i] : NULL;
            }
        } else {
            pim_coord_t c;
            uint32_t byte, j;
            next = parse_coord(tok, n, 2, &c, &byte, &j, NULL, NULL);
            len = parse_values(tok, n, next, buf, sizeof(buf));
            int unused;
            if (state_ptrs(tok[1], tok + 1, n - 1, ptrs, len, &unused)) {
                die("unknown expect kind", tok[1]);
            }
            for (size_t i = 0; i < len; i++) {
                actual[i] = ptrs[i];
            }
        }
        uint8_t got[4096];
        for (size_t i = 0; i < len; i++) {
            got[i] = actual[i] ? *actual[i] : 0;
        }
        const int fmt = next < n ? fmt_of(tok[next]) : -1;
        if (fmt < 0) {
            die("expect needs values", NULL);
        }
        if (r.dump) {
            printf("%s => ", text);
            print_bytes(stdout, fmt, got, len);
            printf("\n");
            return;
        }
        if (memcmp(got, buf, len) != 0) {
            CHECK(0, "%s:%d: %s", r.file, r.line, text);
            fprintf(stderr, "    got: ");
            print_bytes(stderr, fmt, got, len);
            fprintf(stderr, "\n");
        } else {
            CHECK(1, "ok");
        }
        return;
    }
    die("unknown statement", tok[0]);
}

static void
run_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(2);
    }
    char line[8192];
    r.file = path;
    r.line = 0;
    while (fgets(line, sizeof(line), f)) {
        r.line++;
        char *hash = strchr(line, '#');
        if (hash) {
            *hash = 0;
        }
        line[strcspn(line, "\r\n")] = 0;
        char text[8192];
        strcpy(text, line);
        char *tok[MAX_TOK];
        int n = 0;
        for (char *t = strtok(line, " \t"); t && n < MAX_TOK;
             t = strtok(NULL, " \t")) {
            tok[n++] = t;
        }
        if (n) {
            /* Trim the copy for printing. */
            char *s = text;
            while (*s == ' ' || *s == '\t') {
                s++;
            }
            size_t len = strlen(s);
            while (len && (s[len - 1] == ' ' || s[len - 1] == '\t')) {
                s[--len] = 0;
            }
            statement(tok, n, s);
        }
    }
    fclose(f);
}

/* Built-in checks ------------------------------------------------------ */

static uint32_t
f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static void
check_formats(void)
{
    /* bf16: round to nearest even, NaN stays NaN. */
    CHECK(pim_ref_from_float(PIM_REF_BF16, 1.0f) == 0x3f80, "bf16 1");
    CHECK(pim_ref_from_float(PIM_REF_BF16, 1.00390625f) == 0x3f80,
          "bf16 tie to even (down)");
    CHECK(pim_ref_from_float(PIM_REF_BF16, 1.01171875f) == 0x3f82,
          "bf16 tie to even (up)");
    CHECK(pim_ref_from_float(PIM_REF_BF16, 0x1.0101p0f) == 0x3f81,
          "bf16 above the tie");
    CHECK(pim_ref_from_float(PIM_REF_BF16, NAN) == 0x7fc0, "bf16 NaN");
    CHECK(pim_ref_from_float(PIM_REF_BF16, 3.4e38f) == 0x7f80,
          "bf16 overflow to infinity");
    /* fp16: normal, subnormal, overflow, ties. */
    CHECK(pim_ref_from_float(PIM_REF_FP16, 1.0f) == 0x3c00, "fp16 1");
    CHECK(pim_ref_from_float(PIM_REF_FP16, 65504.0f) == 0x7bff, "fp16 max");
    CHECK(pim_ref_from_float(PIM_REF_FP16, 65520.0f) == 0x7c00,
          "fp16 overflow tie");
    CHECK(pim_ref_from_float(PIM_REF_FP16, 5.9604645e-8f) == 0x0001,
          "fp16 smallest subnormal");
    CHECK(pim_ref_from_float(PIM_REF_FP16, 2.9802322e-8f) == 0x0000,
          "fp16 half the smallest subnormal rounds to even");
    CHECK(pim_ref_from_float(PIM_REF_FP16, 1.00048828125f) == 0x3c00,
          "fp16 tie to even");
    long bad16 = 0, badbf = 0;
    for (uint32_t h = 0; h < 0x10000; h++) {
        const int nan16 = (h & 0x7c00) == 0x7c00 && (h & 0x3ff);
        const int nanbf = (h & 0x7f80) == 0x7f80 && (h & 0x7f);
        const float f16 = pim_ref_to_float(PIM_REF_FP16, h);
        const float fbf = pim_ref_to_float(PIM_REF_BF16, h);
        if (nan16 ? !isnan(f16) : pim_ref_from_float(PIM_REF_FP16, f16) != h) {
            bad16++;
        }
        if (nanbf ? !isnan(fbf) : pim_ref_from_float(PIM_REF_BF16, fbf) != h) {
            badbf++;
        }
    }
    CHECK(bad16 == 0, "%ld fp16 values do not round-trip", bad16);
    CHECK(badbf == 0, "%ld bf16 values do not round-trip", badbf);
    CHECK(f2u(pim_ref_to_float(PIM_REF_FP16, 0x0001)) == 0x33800000,
          "fp16 subnormal to fp32");
    /* Integers: sign extension and wrap. */
    uint8_t b[4];
    pim_ref_store_i(PIM_REF_INT16, b, 32768);
    CHECK(pim_ref_load_i(PIM_REF_INT16, b) == -32768, "int16 wrap");
    pim_ref_store_i(PIM_REF_INT8, b, -1);
    CHECK(pim_ref_load_i(PIM_REF_INT8, b) == -1 && b[0] == 0xff, "int8");
    pim_ref_store_i(PIM_REF_INT32, b, 0x80000000LL);
    CHECK(pim_ref_load_i(PIM_REF_INT32, b) == INT32_MIN, "int32 wrap");
}

static void
check_vl_flush_fence(void)
{
    pim_ref_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    pim_geom_t g = {PIM_POLICY_RoRaBaChCo, 2, 128, 1, 1, 2, 2, 16, 4, 32};
    cfg.geom = g;
    cfg.num_regs = 1;
    cfg.num_fmt = PIM_REF_BF16;
    cfg.acc_fmt = PIM_REF_FP32;
    cfg.vlenb = 4;
    cfg.errors_fatal = 1;
    const uint64_t size = pim_geom_size(&g);
    uint8_t *mem = (uint8_t *)aligned_alloc(4096, size);
    memset(mem, 0, size);
    CHECK(pim_ref_attach(mem, &cfg) == 0, "attach");
    /* vl = min(nbytes, 8 x VLENB). */
    CHECK(pim_wr_gb_mem(mem, mem + 1024, 100) == 32, "vl capped at VLMAX");
    CHECK(pim_ref_count(PIM_OP_WR_GB) == 1, "one request");
    CHECK(pim_ab_mac(mem, 0) == 0, "vl 0");
    CHECK(pim_ref_count(PIM_OP_AB_MAC) == 0, "vl 0 issues nothing");
    pim_sb_mac(mem, PIM_DESC(1));
    CHECK(pim_ref_count(PIM_OP_SB_MAC) == 1, "scalar request");
    /* Flushes are logged by line; fences counted. */
    pim_ref_flush_log_clear();
    pim_flush_range(mem + 10, 118); /* [10, 128): 2 lines */
    size_t n;
    const uintptr_t *log = pim_ref_flush_log(&n);
    CHECK(n == 2 && log[0] == (uintptr_t)mem && log[1] == (uintptr_t)mem + 64,
          "flush log of 2 lines, got %zu", n);
    pim_fence();
    pim_fence();
    CHECK(pim_ref_fences() == 2, "fences");
    /* Errors are counted, not fatal, when asked. */
    cfg.errors_fatal = 0;
    CHECK(pim_ref_attach(mem, &cfg) == 0, "re-attach");
    pim_sb_mac(mem + 4, PIM_DESC(1));
    pim_sb_mac(mem, 1ULL << 20);
    CHECK(pim_ref_errors() == 2, "two errors, got %llu",
          (unsigned long long)pim_ref_errors());
    pim_ref_detach();
    free(mem);
}

int
main(int argc, char **argv)
{
    int i = 1;
    if (i < argc && strcmp(argv[i], "--dump") == 0) {
        r.dump = 1;
        i++;
    }
    if (i == argc) {
        check_formats();
        check_vl_flush_fence();
    }
    for (; i < argc; i++) {
        run_file(argv[i]);
    }
    pim_ref_detach();
    if (r.dump) {
        return 0;
    }
    return pim_test_summary("test_model");
}
