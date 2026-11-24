/*
 * L2: one function per wrapper, for test_encoding.sh. The first argument
 * is unused, so the address is in a1 and the descriptor or byte count in
 * a2, the registers of P2 section 1.2's words.
 */

#include <gem5/pim/pim_isa.h>

#define SCALAR(name)                                                          \
    void t_##name(long unused, void *addr, uint64_t desc)                     \
    {                                                                         \
        (void)unused;                                                         \
        name(addr, desc);                                                     \
    }
#define VECTOR(name)                                                          \
    size_t t_##name(long unused, void *addr, size_t n)                        \
    {                                                                         \
        (void)unused;                                                         \
        return name(addr, n);                                                 \
    }
#define STORE_MEM(name)                                                       \
    size_t t_##name(long unused, void *addr, size_t n, const void *src)       \
    {                                                                         \
        (void)unused;                                                         \
        return name(addr, src, n);                                            \
    }
#define LOAD_MEM(name)                                                        \
    size_t t_##name(long unused, void *addr, size_t n, void *dst)             \
    {                                                                         \
        (void)unused;                                                         \
        return name(addr, dst, n);                                            \
    }

SCALAR(pim_sb_ewadd)
SCALAR(pim_sb_ewmul)
SCALAR(pim_sb_ewadd_gbuf)
SCALAR(pim_sb_ewmul_gbuf)
SCALAR(pim_sb_mac)
VECTOR(pim_sb_mov_bkgbuf)
VECTOR(pim_sb_mov_gbufbk)
VECTOR(pim_ab_ewadd)
VECTOR(pim_ab_ewmul)
VECTOR(pim_ab_ewadd_gbuf)
VECTOR(pim_ab_ewmul_gbuf)
VECTOR(pim_ab_mac)
VECTOR(pim_ab_mov_regbk)
VECTOR(pim_ab_mov_bkreg)
STORE_MEM(pim_wr_gb_mem)
STORE_MEM(pim_wr_reg_mem)
LOAD_MEM(pim_rd_gb_mem)
LOAD_MEM(pim_rd_reg_mem)

size_t
t_pim_wr_gb(long unused, void *addr, size_t n, const uint8_t *src)
{
    (void)unused;
    return pim_wr_gb(addr, __riscv_vle8_v_u8m8(src, n), n);
}

size_t
t_pim_wr_reg(long unused, void *addr, size_t n, const uint8_t *src)
{
    (void)unused;
    return pim_wr_reg(addr, __riscv_vle8_v_u8m8(src, n), n);
}

void
t_pim_rd_gb(long unused, void *addr, size_t n, uint8_t *dst)
{
    size_t vl;
    vuint8m8_t v = pim_rd_gb(addr, n, &vl);
    (void)unused;
    __riscv_vse8_v_u8m8(dst, v, vl);
}

void
t_pim_rd_reg(long unused, void *addr, size_t n, uint8_t *dst)
{
    size_t vl;
    vuint8m8_t v = pim_rd_reg(addr, n, &vl);
    (void)unused;
    __riscv_vse8_v_u8m8(dst, v, vl);
}

void
t_pim_cbo_flush(void *line)
{ pim_cbo_flush(line); }

void
t_pim_fence(void)
{ pim_fence(); }
