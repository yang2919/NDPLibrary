// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef RVV_COMPAT_H
#define RVV_COMPAT_H

#define CLEAR(reg)       VCLEAR(reg)
#ifndef MEMBARRIER
#define MEMBARRIER       MEMORY_BARRIER
#endif

#define VEC_CMP_8(...)   VCMP_I8(__VA_ARGS__)
#define VEC_CMP_16(...)  VCMP_I16(__VA_ARGS__)
#define VEC_CMP_32(...)  VCMP_I32(__VA_ARGS__)
#define VEC_CMP_64(...)  VCMP_I64(__VA_ARGS__)
#define VEC_CMP_U8(...)  VCMP_U8(__VA_ARGS__)
#define VEC_CMP_U16(...) VCMP_U16(__VA_ARGS__)
#define VEC_CMP_U32(...) VCMP_U32(__VA_ARGS__)
#define VEC_CMP_U64(...) VCMP_U64(__VA_ARGS__)

#define VLOAD_U8(...)    VLOAD_8(__VA_ARGS__)
#define VLOAD_U16(...)   VLOAD_16(__VA_ARGS__)
#define VLOAD_U32(...)   VLOAD_32(__VA_ARGS__)
#define VLOAD_U64(...)   VLOAD_64(__VA_ARGS__)

#endif
