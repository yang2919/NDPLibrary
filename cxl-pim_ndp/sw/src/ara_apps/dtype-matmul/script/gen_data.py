#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Matteo Perotti
#
# Local extension, not in upstream Ara:
#   optional 5th argv = mode  (random | corner)
#   random : original [0,1) uniform matrices (default, backwards-compat)
#   corner : fault-trigger pattern matrix — alternating powers-of-2 +
#            zero + extremes, dtype-aware. Stresses vmfmacc / vmacc
#            reduction chains with mantissa cancellation / sign mix.

# C = AB with A=[MxN], B=[NxP], C=[MxP]
# arg1: dtype, arg2/3/4: M, N, P, arg5(optional): mode

import random as rand
import numpy as np
import sys

def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)

# ─── Corner matrix generator (dtype-aware) ───────────────────────────────
def corner_matrix(rows, cols, dtype):
    np_dt = np.dtype(dtype)
    if np_dt.kind == 'f':                 # float64/32/16
        # mix of ±powers of 2 (exact), ±0, ±1.5 (mantissa odd), small,
        # large for cancellation
        seq = [
             0.0,    1.0,   -1.0,    0.5,   -0.5,
             2.0,   -2.0,    1.5,   -1.5,    0.125,
            -0.125,  4.0,   -4.0,    1.0e-7, 1.0e+6,
            -1.0e+6, 0.0,    1.0,   -1.0,    3.0,
        ]
    else:                                  # int64/32/16/8
        info = np.iinfo(np_dt)
        m = info.max
        seq = [
            0, 1, -1, 2, -2,
            m // 4, -(m // 4), 4, -4, 0,
            1, -1, m // 8, -(m // 8), 0,
            3, -3, m // 16, -(m // 16), 0,
        ]
    Nt = rows * cols
    flat = np.array([seq[i % len(seq)] for i in range(Nt)], dtype=dtype)
    return flat.reshape(rows, cols)

############
## SCRIPT ##
############

if len(sys.argv) == 5:
    dtype = str(sys.argv[1])
    M = int(sys.argv[2]); N = int(sys.argv[3]); P = int(sys.argv[4])
    mode = "random"
elif len(sys.argv) == 6:
    dtype = str(sys.argv[1])
    M = int(sys.argv[2]); N = int(sys.argv[3]); P = int(sys.argv[4])
    mode = str(sys.argv[5])
else:
    print("Error. usage: gen_data.py <dtype> <M> <N> <P> [mode]")
    print("  dtype in [float64, float32, float16, int64, int32, int16, int8]")
    print("  mode  in [random, corner]    (default: random)")
    print("  C = AB with A=[MxN], B=[NxP], C=[MxP]")
    sys.exit()

# Matrices and results
if mode == "random":
    A = np.random.rand(M, N).astype(dtype)
    B = np.random.rand(N, P).astype(dtype)
elif mode == "corner":
    A = corner_matrix(M, N, dtype)
    B = corner_matrix(N, P, dtype)
else:
    print("Error. unknown mode: %s" % mode); sys.exit()

C = np.zeros([M, P], dtype=dtype)
# Golden result matrix
G = np.matmul(A, B).astype(dtype)

# Create the file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("P", np.array(P, dtype=np.uint64))
emit("a", A, 'NR_LANES*4')
emit("b", B, 'NR_LANES*4')
emit("c", C, 'NR_LANES*4')
emit("g", G, 'NR_LANES*4')
