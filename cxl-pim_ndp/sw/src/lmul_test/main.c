// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#ifdef HP
#define SEW  "e16"
#define LVLE "vle16.v"
#define LVSE "vse16.v"
#else
#define SEW  "e32"
#define LVLE "vle32.v"
#define LVSE "vse32.v"
#endif

#define NEL 16

real_t vbuf[2048] __attribute__((aligned(64)));
real_t obuf[2048] __attribute__((aligned(64)));

#define VS(LM)  "vsetvli t0, %[N], " SEW ", " LM ", ta, ma\n\t"
#define LD8     LVLE " v8, (%[s])\n\t"
#define LD16    LVLE " v16, (%[s])\n\t"
#define ST8     LVSE " v8, (%[d])\n\t"
#define ST16    LVSE " v16, (%[d])\n\t"
#define CLB "t0","v8","v9","v10","v11","v12","v13","v14","v15", \
            "v16","v17","v18","v19","v20","v21","v22","v23","memory"

#define ILOP(LM, OP) asm volatile(VS(LM) LD8 OP " v8, v8, v8\n\t" ST8 \
  :: [N]"r"((size_t)NEL), [s]"r"(vbuf), [d]"r"(obuf) : CLB)

#define MACC(LM) asm volatile(VS(LM) LD8 LD16 "vfmacc.vv v16, v8, v8\n\t" ST16 \
  :: [N]"r"((size_t)NEL), [s]"r"(vbuf), [d]"r"(obuf) : CLB)

#define REDUSUM(LM) ({ real_t _r; asm volatile(VS(LM) LD8 \
  "vmv.s.x v16, zero\n\t" "vfredusum.vs v16, v8, v16\n\t" "vfmv.f.s %[o], v16\n\t" \
  : [o]"=f"(_r) : [N]"r"((size_t)NEL), [s]"r"(vbuf) : CLB); (double)_r; })
#define REDMIN(LM) ({ real_t _r; real_t _big = (real_t)1e4; asm volatile(VS(LM) LD8 \
  "vfmv.s.f v16, %[b]\n\t" "vfredmin.vs v16, v8, v16\n\t" "vfmv.f.s %[o], v16\n\t" \
  : [o]"=f"(_r) : [N]"r"((size_t)NEL), [s]"r"(vbuf), [b]"f"(_big) : CLB); (double)_r; })

#define SLIDEDN(LM) asm volatile(VS(LM) LD8 "vslidedown.vx v16, v8, %[o1]\n\t" ST16 \
  :: [N]"r"((size_t)NEL), [s]"r"(vbuf), [d]"r"(obuf), [o1]"r"((size_t)1) : CLB)
#define SLIDEUP(LM) asm volatile(VS(LM) LD8 LD16 "vslideup.vx v16, v8, %[o1]\n\t" ST16 \
  :: [N]"r"((size_t)NEL), [s]"r"(vbuf), [d]"r"(obuf), [o1]"r"((size_t)1) : CLB)

static int npass, nfail; static uint32_t failmap;
static void rep(const char *nm, int ok) {
  if (npass + nfail < 32 && !ok) failmap |= (1u << (npass + nfail));
  if (ok) npass++; else nfail++;
  uart_puts("  "); uart_puts(nm); uart_puts(ok ? " PASS\n" : " FAIL\n");
}

static int ck_2x(void)   { for (int i=0;i<NEL;i++) if ((double)obuf[i]!=2.0*(double)vbuf[i]) return 0; return 1; }
static int ck_0(void)    { for (int i=0;i<NEL;i++) if ((double)obuf[i]!=0.0) return 0; return 1; }
static int ck_sq(void)   { for (int i=0;i<NEL;i++){double v=(double)vbuf[i]; if ((double)obuf[i]!=v*v) return 0;} return 1; }
static int ck_macc(void) { for (int i=0;i<NEL;i++){double v=(double)vbuf[i]; if ((double)obuf[i]!=v+v*v) return 0;} return 1; }
static int ck_sldn(void) { for (int i=0;i<NEL-1;i++) if ((double)obuf[i]!=(double)vbuf[i+1]) return 0; return 1; }
static int ck_slup(void) { for (int i=0;i<NEL;i++){double e=(i>=1)?(double)vbuf[i-1]:(double)vbuf[0]; if ((double)obuf[i]!=e) return 0;} return 1; }

#define BATTERY(LM) do { \
  uart_puts(" " LM ":\n"); \
  ILOP(LM, "vfadd.vv"); rep(LM " vfadd",  ck_2x()); \
  ILOP(LM, "vfsub.vv"); rep(LM " vfsub",  ck_0());  \
  ILOP(LM, "vfmul.vv"); rep(LM " vfmul",  ck_sq()); \
  MACC(LM);             rep(LM " vfmacc", ck_macc()); \
  { double rs = 0; for (int i=0;i<NEL;i++) rs += (double)vbuf[i]; \
    rep(LM " vfredusum", REDUSUM(LM) == rs); } \
  rep(LM " vfredmin", REDMIN(LM) == 1.0); \
  SLIDEDN(LM); rep(LM " vslidedown", ck_sldn()); \
  SLIDEUP(LM); rep(LM " vslideup",   ck_slup()); \
} while (0)

int main(void) {
  HB[0] = 0xCAFE0001; uart_init(UART_DIVISOR); HB[0] = 0xCAFE0002;
  uart_puts("\n[lmul_test " DTYPE_NAME "] m2/m4/m8 op battery (in-lane + reduction + slide)\n");
  npass = nfail = 0; failmap = 0;
  for (int i = 0; i < 2048; i++) vbuf[i] = (real_t)((i & 15) + 1);

  BATTERY("m2"); HB[0] = 0xCAFE0003;
  BATTERY("m4"); HB[0] = 0xCAFE0004;
  BATTERY("m8"); HB[0] = 0xCAFE0005;

  int total = npass + nfail, pass = (nfail == 0);
  VERIFY[0] = 0xA1A1B1B1; VERIFY[1] = pass ? 1 : 0;
  VERIFY[2] = (uint32_t)total; VERIFY[3] = (uint32_t)npass;
  VERIFY[4] = (uint32_t)nfail; VERIFY[5] = failmap;
  VERIFY[6] = 0; VERIFY[7] = 0;
  uart_puts("==== "); print_u64(npass); uart_puts(" / "); print_u64(total);
  uart_puts(pass ? " PASS ====\n" : " (some FAIL) ====\n");
  HB[0] = 0xCAFE000F;
  while (1) {}
}
