// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_PLATFORM_H
#define ARA_VHK158_PLATFORM_H

#ifndef ARA_NR_LANES
#define ARA_NR_LANES          2
#endif
#ifndef ARA_VLEN
#define ARA_VLEN              2048
#endif
#ifndef ARA_ELEN
#define ARA_ELEN              64
#endif
#ifndef ARA_AXI_DATA_WIDTH
#define ARA_AXI_DATA_WIDTH    64

#endif
#ifndef ARA_SOC_CLK_HZ
#define ARA_SOC_CLK_HZ        120000000u

#endif

#define ARA_ENTRY_WIDTH_BITS  (64 * ARA_NR_LANES)
#define ARA_BYTES_PER_CYCLE   (((ARA_AXI_DATA_WIDTH) < (ARA_ENTRY_WIDTH_BITS) \
                                ? (ARA_AXI_DATA_WIDTH) : (ARA_ENTRY_WIDTH_BITS)) / 8)

#define ARA_LANE_ELEM(elem_bytes)  ((unsigned)ARA_NR_LANES * (ARA_ELEN / (8u * (unsigned)(elem_bytes))))
#define ARA_FMA_PEAK(elem_bytes)   (2u * ARA_LANE_ELEM(elem_bytes))

#define ARA_UART_DIVISOR      ((ARA_SOC_CLK_HZ + 8u * 115200u) / (16u * 115200u))

#ifndef ARA_ERRATUM_FP_MINMAX_REDUCTION
#define ARA_ERRATUM_FP_MINMAX_REDUCTION   1
#endif

#ifndef ARA_ERRATUM_VRGATHER_UNVERIFIED
#define ARA_ERRATUM_VRGATHER_UNVERIFIED   1
#endif

#ifndef ARA_ERRATUM_STORE_SEGMENT_UNVERIFIED
#define ARA_ERRATUM_STORE_SEGMENT_UNVERIFIED  1
#endif

#ifndef ARA_TOOLCHAIN_NO_AUTOVEC_REQUIRED
#define ARA_TOOLCHAIN_NO_AUTOVEC_REQUIRED     1
#endif

#ifndef ARA_HAS_VTRANS_FU
#ifdef USE_VTRANS
#define ARA_HAS_VTRANS_FU     1
#else
#define ARA_HAS_VTRANS_FU     0
#endif
#endif

#endif
