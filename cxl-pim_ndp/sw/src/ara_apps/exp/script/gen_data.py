#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and University of Bologna.
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
# Local extension, not in upstream Ara:
#   argv 2 modes — random / linear / corner.
#   random : original [0,1) uniform (default, backwards-compat)
#   linear : -4 .. +4 sweep
#   corner : 16 fault-trigger candidates repeated

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

def rand_matrix(N, dtype):
  return np.random.rand(N).astype(dtype)

# Corner case set, CORNER_LEN = 16
CORNER_F = [
   0.0,
   0.69314718055994531,    -0.69314718055994531,    # ±ln(2)
   0.34657359027997265,    -0.34657359027997265,    # ±ln(2)/2
   1.03972077083991795,    -1.03972077083991795,    # ±1.5·ln(2)
   1.38629436111989061,    -1.38629436111989061,    # ±2·ln(2)
   1.0,                    -1.0,
   3.0,                    -3.0,
  87.3,                   -87.3,                    # near ±88.376 clamp
   1.0e-15,
]

def corner_seq(N):
    return np.array([CORNER_F[i % len(CORNER_F)] for i in range(N)])

def linear_seq(N):
    step = 8.0 / N
    return np.array([-4.0 + step * i for i in range(N)])

# Bit-exact emulation of kernel/exp.h Cephes 6-coefficient exp.
# Replaces libm np.exp on the gold path so golden matches HW output.
def cephes_exp(x):
  x = np.asarray(x, dtype=np.float64)
  EXP_HI = np.float64(88.3762626647949)
  EXP_LO = np.float64(-88.3762626647949)
  LOG2EF = np.float64(1.44269504088896341)
  C1 = np.float64(0.693359375)
  C2 = np.float64(-2.12194440e-4)
  p0 = np.float64(1.9875691500E-4)
  p1 = np.float64(1.3981999507E-3)
  p2 = np.float64(8.3334519073E-3)
  p3 = np.float64(4.1665795894E-2)
  p4 = np.float64(1.6666665459E-1)
  p5 = np.float64(5.0000001201E-1)
  x = np.minimum(x, EXP_HI)
  x = np.maximum(x, EXP_LO)
  fx = np.float64(0.5) + x * LOG2EF
  tmp = np.trunc(fx)
  fx = np.where(fx < tmp, tmp - np.float64(1.0), tmp)
  x = x - fx * C1
  x = x - fx * C2
  z = x * x
  y = p0
  y = y * x + p1
  y = y * x + p2
  y = y * x + p3
  y = y * x + p4
  y = y * x + p5
  y = y * z + x
  y = y + np.float64(1.0)
  return np.ldexp(y, fx.astype(np.int64))

############
## SCRIPT ##
############

if len(sys.argv) == 2:
  N_f64 = int(sys.argv[1])
  mode = "random"
elif len(sys.argv) == 3:
  N_f64 = int(sys.argv[1])
  mode = sys.argv[2]
else:
  print("Error. usage: gen_data.py <N> [mode]    mode=random|linear|corner")
  sys.exit()

N_f32 = 2 * N_f64

if mode == "random":
  exponents_f64 = rand_matrix(N_f64, np.float64).astype(np.float64)
  exponents_f32 = rand_matrix(N_f32, np.float32).astype(np.float32)
elif mode == "linear":
  exponents_f64 = linear_seq(N_f64).astype(np.float64)
  exponents_f32 = linear_seq(N_f32).astype(np.float32)
elif mode == "corner":
  exponents_f64 = corner_seq(N_f64).astype(np.float64)
  exponents_f32 = corner_seq(N_f32).astype(np.float32)
else:
  print("Error. unknown mode: %s" % mode)
  sys.exit()

# Results buffer
results_f64 = np.zeros(N_f64, dtype=np.float64)
results_f32 = np.zeros(N_f32, dtype=np.float32)

# Gold results — bit-exact Cephes emulation of kernel/exp.h
gold_results_f64 = cephes_exp(exponents_f64).astype(np.float64)
gold_results_f32 = cephes_exp(exponents_f32.astype(np.float64)).astype(np.float32)

# Create the file
print(".section .data,\"aw\",@progbits")
emit("N_f64", np.array(N_f64, dtype=np.uint64))
emit("exponents_f64", exponents_f64, 'NR_LANES*4')
emit("results_f64", results_f64, 'NR_LANES*4')
emit("gold_results_f64", gold_results_f64, 'NR_LANES*4')
emit("N_f32", np.array(N_f32, dtype=np.uint32))
emit("exponents_f32", exponents_f32, 'NR_LANES*4')
emit("results_f32", results_f32, 'NR_LANES*4')
emit("gold_results_f32", gold_results_f32, 'NR_LANES*4')
