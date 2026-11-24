# Tensor-parallel full-model validation

The latest RMSNorm correction results are recorded at the end of this document;
the initial baseline is retained for comparison. Full numerical acceptance
remains open.

## Scope and reproducibility

Validated on 2026-11-16 using the native FP16 gem5_pim numerical worker on
macOS, with two independent worker connections. No extended gem5 FS execution
or timing/performance claim is included.

- Checkpoint: HuggingFaceTB/SmolLM2-135M, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, loaded as CPU FP16.
- Reference: Transformers 4.57.6, PyTorch 2.9.1, eager Llama attention.
- Architecture: 30 layers, hidden width 576, intermediate width 1536,
  Q heads 9, KV heads 3, head width 64. Device partitions Q=(6,3), KV=(2,1).
- Prompt IDs: [1,42,30,198]. This is a fixed token-ID smoke test, not a language
  benchmark or representative prompt suite.
- Compared all layers and 49,152 logits for 12 sequential input tokens.
  Both models received the same prefix, extending it with reference argmax IDs.
- Separately generated 8 tokens autoregressively from the four-token prompt,
  and repeated NDP generation after history reset.

```sh
make -C tools/gem5/pim_kernel fp16
cmake --build build-python
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel_model.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16 --steps 8
```

The checkpoint and Transformers package must be installed separately. The
example prints JSON metrics. It asserts finite values, exact repeated token
sequences and greedy-token agreement. It reports the existing elementwise
numerical criterion separately (`numerical_allclose`); a successful script
exit does not imply that numerical criterion passed.

## Initial baseline before the RMSNorm correction

| Check | Result |
|---|---|
| Native regression suite | 26/26 CTest entries passed, including Python contracts |
| Next-token argmax on common-prefix inputs | 12/12 matched |
| Independently generated new tokens | 8/8 matched |
| Repeated generation after reset | Exact token-ID match |
| Maximum absolute logit difference | 0.203125 |
| Logit RMSE over 589,824 values | 0.0264796511 |
| Logits outside atol=0.02, rtol=0.02 | 14,992 / 589,824 (about 2.54%) |
| Maximum layer relative L2 error | 0.60682% |
| Layer elements outside the same tolerance | 3,211 / 207,360 |

Both generated the new IDs `[198,504,808,2775,288,536,314,288]`.
The full-model elementwise numerical criterion **did not pass**. This result
establishes the execution flow and agreement on this short greedy trajectory;
it does not establish general numerical equivalence, perplexity or accuracy.
Tolerance was not relaxed to turn the result into a pass.

Layer errors below are aggregated over all 12 input tokens. Relative L2 is
`sqrt(sum((actual-reference)^2) / sum(reference^2))`, accumulated in FP32.
Each layer comparison contains 6,912 values. Reference maximum and maximum
error can occur at different coordinates; their ratio is not a pointwise
relative-error bound.

| Layer (zero-based) | Max absolute error | Max absolute reference | Relative L2 error | Elements outside tolerance |
|---|---|---|---|---|
| 0 | 0.03125 | 39 | 0.0692% | 0 |
| 1 | 0.0625 | 132.25 | 0.0666% | 0 |
| 2 | 1.5 | 757.5 | 0.1627% | 0 |
| 3 | 1.5 | 756.5 | 0.1629% | 0 |
| 4 | 1.5 | 754.5 | 0.1634% | 0 |
| 5 | 1.5 | 750.5 | 0.1647% | 0 |
| 6 | 1.5 | 746.5 | 0.1657% | 0 |
| 7 | 1.5 | 740 | 0.1677% | 0 |
| 8 | 1.5 | 675.5 | 0.1876% | 0 |
| 9 | 1.5 | 313.5 | 0.3077% | 0 |
| 10 | 1.5 | 95.875 | 0.3758% | 5 |
| 11 | 16 | 20432 | 0.0785% | 77 |
| 12 | 16 | 20416 | 0.0786% | 80 |
| 13 | 16 | 20400 | 0.0786% | 78 |
| 14 | 16 | 20384 | 0.0787% | 97 |
| 15 | 16 | 20368 | 0.0788% | 87 |
| 16 | 16 | 20352 | 0.0788% | 105 |
| 17 | 16 | 20336 | 0.0789% | 91 |
| 18 | 16 | 20336 | 0.0789% | 90 |
| 19 | 16 | 20336 | 0.0790% | 116 |
| 20 | 16 | 20336 | 0.0790% | 128 |
| 21 | 16 | 20336 | 0.0791% | 131 |
| 22 | 16 | 20336 | 0.0792% | 168 |
| 23 | 16 | 20336 | 0.0793% | 176 |
| 24 | 16 | 20304 | 0.0796% | 186 |
| 25 | 16 | 20304 | 0.0797% | 214 |
| 26 | 16 | 20304 | 0.0798% | 216 |
| 27 | 16 | 20192 | 0.0806% | 293 |
| 28 | 32 | 4296 | 0.6068% | 303 |
| 29 | 4 | 794 | 0.2884% | 570 |

## Execution and lifecycle coverage

The stack passes root-device FP16 rows directly between layers. CPU embedding
feeds the first layer; final RMSNorm runs on device 0 before downloading its
output for the CPU LM head. FP32 projection reductions still run on the host.
The verification hooks download intermediate layer rows for observation;
these downloads are absent in normal model execution.

Native unit tests cover two-layer composition, a single-device reference,
one Python upload/download pair at the stack boundary, source checkpoint keys
and reload, generation capacity, zero new tokens, EOS stopping, out-of-range
IDs, history reset, a late-layer MLP failure, and failure after the decoder
in final normalization. A downstream failure invalidates the whole stack.

## Remaining precision work

The implementation rounds resident intermediate operations to FP16 and sums
TP projection partials in FP32 before FP16 conversion. The Hugging Face
reference uses its own operator ordering and intermediate precision. These
are possible contributors to accumulated differences; this test does not
isolate which operation causes the largest drift. Layer 28 has large hidden
values (reference maximum 4,296) as well as an absolute difference of 32.

The follow-up below compares one-device and two-device runs and isolates
normalization on identical inputs. Further work must localize the remaining
projection, RoPE and attention differences without accumulated input drift. Longer contexts and multiple realistic prompts also remain
necessary. The current result is not evidence of a gem5 device implementation
error, since only native numerical workers were used.

## RMSNorm correction and device-count control

The full-model test was repeated with one device and two devices, before and
after the adapter correction. The one-device run uses the same adapter with
unsplit projections and the same FP32 result path. It is a device-count control,
not a separate hardware reference. All runs use the same 12 input tokens and
compare to the same Hugging Face eager model.

| Execution | Logit max absolute error | Logit RMSE | Logits outside tolerance / 589,824 |
|---|---|---|---|
| One device, fused norm | 0.375 | 0.056502109 | 19129 (3.243%) |
| Two devices, fused norm | 0.203125 | 0.026479651 | 14992 (2.542%) |
| One device, corrected norm | 0.15625 | 0.024516599 | 1532 (0.260%) |
| Two devices, corrected norm | 0.15625 | 0.025704637 | 1365 (0.231%) |

Every run retained 12/12 reference next-token choices, matched all 8 independently
generated tokens and reproduced the generated IDs after reset. Both corrected
runs still fail the original elementwise numerical criterion; no tolerance
was relaxed. Significant error exists even without splitting a projection,
so the original drift cannot be attributed solely to TP host reduction.
These comparisons do not quantify each individual operator's contribution.

### Isolated normalization finding

Hugging Face Llama computes the normalization in FP32, converts the normalized
values to the input dtype, then multiplies by the learned weight. The old
adapter fused the weight multiplication before the final FP16 conversion.
On identical inputs captured from all 61 norms over four prompt tokens
(140,544 values):

| Formula | Values unequal to reference | Maximum absolute error |
|---|---|---|
| Fused normalize and weight, then FP16 | 36,306 | 0.015625 |
| Normalize, FP16, weight, FP16 | 24 | 0.0001220703125 |

The adapter now uses the second formula: L2 RMSNorm with resident unit weights,
followed by resident multiplication by gamma. General L2 RMSNorm semantics
remain unchanged. The remaining isolated differences were measured, but their
precise cause was not established by this experiment. FP32 reduction ordering
is a possible contributor, not a verified explanation.

```sh
PYTHONPATH=build-python/python python examples/python/llama_norm_precision.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel_model.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16 --devices 1 --steps 8
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel_model.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16 --devices 2 --steps 8
```

A deterministic native regression uses non-unit gamma values to distinguish
the fused and staged formulas, and verifies checkpoint reload. The full native
CTest suite passed 26/26 after the correction. The model validation example
accepts --devices for repeatable device-count controls and reports the device
count with its JSON metrics.

### Remaining work

The next precision investigation should compare projection outputs on
identical reference activations, including complete and input-sharded dot
products. This avoids mistaking propagated upstream errors for local errors.
RoPE and attention should then be checked with the same fixed Q/K/V inputs.
Greedy agreement on this short trajectory remains a smoke test, not proof of
full numerical equivalence or general generation quality.
