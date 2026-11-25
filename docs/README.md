# NDP Library documentation

Start with the [project quickstart](../README.md) for the native build and decoder example.

[NDP 라이브러리 기능 안내 (한국어)](runtime-overview.ko.md)

## Library and serving APIs

- [Using NDP Library: Generic L2 APIs and a Decoder Layer](using-ndp-library.md)
- [Using NDP Library from PyTorch](pytorch-integration.md)
- [NDP Library Object Abstractions](library-abstractions.md)
- [Data flow across library layers](layer-data-flow.md)
- [Multi-device execution and tensor parallelism](parallelism.md)
- [Multi-token PNM prefill](prefill.md)
- [Sequential request scheduling](request-scheduling.md)
- [Overlapping tensor-parallel stages](pipeline-execution.md)
- [Tensor-parallel full-model validation](tensor-parallel-model-validation.md)

## Extended gem5 and libpim

- [Extended gem5 validation record](extended-gem5-validation-plan.md)
- [V0: libpim build and FP16 profile integration](extended-gem5-v0.md)
- [Extended gem5 PIM: one-command Docker verification](extended-gem5-smoke-verification.md)
- [External Python client and a persistent SE worker](gem5-se-external-client.md)
- [External Python client on extended gem5 FS](extended-gem5-fs-client.md)
- [Qwen3 single-layer decode timing](qwen3-layer-timing.md)
- [Resume extended gem5 development on Linux](linux-server-handoff.md)
- [Kernel launch plan for the RISC-V gem5 PIM backend](gem5-pim-kernel-plan.md)
- [Separate PNM core launch ABI (N0 proposal)](pnm-launch-abi.md)

## Backend implementation and FPGA requirements

- [Backend integration handoff](backend-handoff.md)
- [Command-driven NDP backend](ndp-backend.md)
- [NDP command contract and implementation status](ndp-command-contract.md)
- [Single-device execution and layout specification](execution-layout-spec.md)
- [Native CENT conformance results](cent-conformance-results.md)
- [Reproduce the CENT conformance checks](cent-user-verification.md)
- [CENT-based execution/layout validation plan](cent-validation-plan.md)
- [VHK158 FPGA integration requirements](vhk158-integration-requirements.md)

## Stock gem5 and command transport

- [Stock gem5 FS verification](stock-gem5-verification.md)
- [Stock gem5 FS baseline (F0)](gem5-fs.md)
- [F2: metadata-only guest IPC](gem5-f2.md)
- [F3: persistent emulator execution](gem5-f3.md)
- [F4: Session integration with an owned receiver](gem5-f4.md)
- [F5: guest decoder integration](gem5-f5.md)
- [K4 사용자 검증 — 외부 PyTorch에서 stock gem5 게스트의 GEMV 실행](gem5-k4-stock-fs-user-verification.md)
- [Command transport and codec (F1)](command-transport.md)
- [NDP–gem5 communication specification, version 1](gem5-communication-spec.md)
- [NDP–gem5 packet specification v2](gem5-extension-handoff.md)
- [NDP–gem5 패킷 규격 v2](gem5-extension-handoff.ko.md)
- [NDP–gem5 PIM and memory packet specification v2](gem5-pim-packets.md)
- [NDP–gem5 PIM 및 메모리 패킷 규격 v2](gem5-pim-packets.ko.md)
