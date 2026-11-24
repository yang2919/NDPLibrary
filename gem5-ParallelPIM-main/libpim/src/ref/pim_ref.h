/*
 * The reference backend's model of the PIM commands on the host: the
 * state, number formats, command semantics and software errors of gem5's
 * functional model (src/mem/ramulator2/pim/pim_func.cc), written
 * independently, so that the two can be compared.
 *
 * The PIM memory is a host byte array (the region); the model decodes
 * every PIM address with pim_addr.h's decoder and keeps the global
 * buffers (GB), registers and accumulators itself. The wrappers of
 * pim_isa.h call pim_ref_scalar() and pim_ref_vector() when built with
 * PIM_BACKEND_REF.
 *
 * Private to libpim (the runtime and the tests).
 */

#ifndef __PIM_REF_PIM_REF_H__
#define __PIM_REF_PIM_REF_H__

#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim_addr.h>
#include <gem5/pim/pim_isa.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number formats, as PIM_LAYOUT_NUM_FMT. */
enum pim_ref_fmt
{
    PIM_REF_BF16 = 0,
    PIM_REF_FP16 = 1,
    PIM_REF_FP32 = 2,
    PIM_REF_INT8 = 3,
    PIM_REF_INT16 = 4,
    PIM_REF_INT32 = 5,
};

typedef struct pim_ref_config
{
    pim_geom_t geom;
    uint32_t num_regs; /* registers per PU; rows REG_ROW0 .. */
    uint32_t num_fmt;  /* operands and element-wise results */
    uint32_t acc_fmt;  /* the MAC accumulator */
    size_t vlenb;      /* vector register bytes; VLMAX at e8, m8 = 8x */
    int errors_fatal;  /* 1: abort on a software error, as gem5's default */
} pim_ref_config_t;

/* Start the model on a region of pim_geom_size() bytes; 0 on success. */
int pim_ref_attach(void *base, const pim_ref_config_t *cfg);
void pim_ref_detach(void);
const pim_ref_config_t *pim_ref_config(void);
void pim_ref_set_errors_fatal(int fatal);

/* Software errors seen (with errors_fatal = 0 each is a no-op). */
uint64_t pim_ref_errors(void);
/* Requests issued per operation (index PIM_OP_*), and fences. */
uint64_t pim_ref_count(enum pim_op op);
uint64_t pim_ref_fences(void);

/* The lines flushed since the last pim_ref_flush_log_clear(). */
const uintptr_t *pim_ref_flush_log(size_t *n);
void pim_ref_flush_log_clear(void);

/* State, for tests: the GB of a unit, register j and the accumulator of
 * the PU of a bank (coordinates ch, unit, sid, bg, bank). */
uint8_t *pim_ref_gb(uint32_t ch, uint32_t unit);
uint8_t *pim_ref_reg(const pim_coord_t *bank, uint32_t j);
uint8_t *pim_ref_acc(const pim_coord_t *bank);

/* Element formats. */
size_t pim_ref_fmt_bytes(uint32_t fmt);
int pim_ref_fmt_is_float(uint32_t fmt);
/* float <-> a floating format, round to nearest even. */
float pim_ref_to_float(uint32_t fmt, uint32_t bits);
uint32_t pim_ref_from_float(uint32_t fmt, float value);
/* Load and store one element; integers sign-extended or wrapped. */
float pim_ref_load_f(uint32_t fmt, const uint8_t *p);
int64_t pim_ref_load_i(uint32_t fmt, const uint8_t *p);
void pim_ref_store_f(uint32_t fmt, uint8_t *p, float v);
void pim_ref_store_i(uint32_t fmt, uint8_t *p, int64_t v);

#ifdef __cplusplus
}
#endif

#endif // __PIM_REF_PIM_REF_H__
