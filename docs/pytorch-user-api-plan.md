# PyTorch dispatcher design and verification

For model conversion, shared Q/K/V inputs, and checkpoint results, see the [PyTorch integration guide](pytorch-integration.md).

## Operator and state contract

The CPU dispatch kernels for `ndp::linear_pim` and `ndp::qkv_pim` take CPU FP16 tensors and explicit `LinearPackedParams` objects. Each packed object holds its Session and prepared weight. Fake execution reads the logical input and output widths. A real launch checks Session lifetime and thread ownership. Q/K/V requires three weights from the same Session and returns separate CPU tensors.

The module serializes the CPU weight in its state dictionary and rebuilds the packed representation after checkpoint loading. Direct edits require refresh before execution. A backend failure propagates to the caller.

## Implementation status

U1–U4 passed their native acceptance checks. The functional and native libpim backends passed the two-step tiny Llama checkpoint comparison.

U3 passed all four torch.library.opcheck tests on PyTorch 2.9.1: schema, autograd registration, FakeTensor and dynamic AOT dispatch. Packed handles compare by native identity and raise TypeError when probed for sequence length. The fake kernel uses immutable shape metadata, including after the runtime closes.

On macOS arm64 with Python 3.11.18 and PyTorch 2.9.1, torch.compile(fullgraph=True, backend="eager") captured the fixed-shape linear_pim operator and matched eager execution for two inputs on both the functional and native libpim backends. Torch warns during capture that it retains the original session-bound handle when deepcopy is unavailable. Coverage applies to operator graph capture; whole-model compilation and Inductor code generation require separate validation.

Run the regression with the same Python interpreter used to build the extension:

```sh
PYTHONPATH=build-python/python NDP_GEM5_PIM_WORKER="$PWD/build/pim-kernel-worker-fp16" \
    python -m unittest discover -s tests/python -p test_dispatch.py -v
```

## Proposed model conversion API

Section 5.3.1 of the AiMX Platform Whitepaper (pp. 18–20) describes two entry points. Developers can define modules that call the runtime during initialization and forward, or convert an existing PyTorch model with to_aim(). Figure 5-12 illustrates flattening nested Sequential containers and combining eligible adjacent layers. The specialized attention module illustrates a second level of conversion that knows the semantics of an entire model component.

For NDP, add one Python entry point over the current dispatcher and prepared weights. The following is a proposed API:

```python
model = model.eval()  # CPU FP16 checkpoint already loaded
with ndp.Session(backend="ndp") as runtime:
    with ndp.to_ndp(model, runtime, target="model.layers.0") as accelerated:
        with torch.inference_mode():
            output = accelerated(input_ids, use_cache=True)
```

The context yields the same model instance and restores the original modules on exit. A serving process opens this scope once around its request loop. Use a qualified module path to select a subtree and allow an explicit empty path for the root. Require target in the first release so a small integration experiment cannot accidentally prepare the entire checkpoint. Keep transport configuration in Session. Use the existing from_float interface for a single explicitly selected projection.

1. **Discover before mutation.** Resolve the subtree and its registered children. Build a conversion plan listing paths, matched rules, estimated weight storage where available, and reasons for skipped modules. Expose the same discovery through a read-only preview. Count a shared module or parameter only when its semantics and storage ownership are supported. Initially skip such cases and parametrized or hooked modules whose behavior a replacement could bypass. Report an empty match explicitly.
2. **Plan device placement for the region.** For the recognized Llama block, lower projections, functional operations and residual branches to logical operations. Query L2 planning with concrete engine, shape, dtype, layout and memory requirements. Plan both prefill and decode. Record placement as PIM, PNM or host, including the reason for any host boundary. The ndp command backend models both engines; gem5_pim currently exposes PIM Matvec and single-token bias-free Linear. A full-device request must report a missing kernel or unsupported cache contract before installation. Allow partial coverage explicitly for projection-only targets.
3. **Prepare and install a resident implementation.** Use resident L2 tensors for intermediate activations and L3 for weight preparation and request-owned KV caches. For compatible shapes use PIM GEMV for decode and PNM GEMM for prefill. Initially map RMSNorm, RoPE composition, attention score/value products, softmax, SwiGLU elementwise operations and residual additions to PNM where supported. Keep shared bank storage, PNM scratchpad transfers, layout conversion and synchronization within the runtime/backend contract. Upload a CPU activation at entry and download the requested output at exit. Retain the existing projection adapter for partial coverage. Prepare replacements before mutation and roll back on failure.
4. **Validate incrementally.** First implement a resident SwiGLU MLP on the command emulator. Verify its two input projections reuse one device activation, their outputs feed PNM SiLU and multiply, and the down projection consumes the resident product. Compare with CPU on the pinned checkpoint and count host transfers separately from device relayout and scratchpad copies. Then extend to attention and a full block. Match Hugging Face RoPE convention, mask, cache positions, grouped query heads and cache lifetime per request before comparing intermediate values, logits and greedy tokens. Test checkpoint keys and original-module restoration on normal and exceptional exit. Use the measured five uploads/seven downloads as a projection-only baseline. A fully resident region with prepared weights and cache should require one activation upload and one output download per CPU-boundary call.
