# LLM serving을 위한 NDP Library L2/L3 설계 초안

상태: 구현 전 제안. 실제 FPGA 사양이 확인되지 않은 항목은 capability와 backend 계약으로 격리한다.

현재 단일 device 구현의 범위·객체 계약·작업 순서는 [Single-device 구현 계획](single-device-implementation-plan.md)을 따른다. 이 문서의 multi-device 및 mutable tensor 일반화는 장기 설계 방향으로 남긴다.

## 1. 하드웨어 모델과 원칙

- NDP device 하나는 PNM(RISC-V + VPU), PNM scratchpad(SPM), off-chip PIM memory, PIM compute engine을 포함한다.
- PIM memory는 device의 주 저장 공간이다. PNM이 접근할 수 있는 범위와 방법은 backend가 선언한다. 직접 load/store 가능 여부는 아직 미확정이다.
- PIM은 matrix-vector MAC과 지원되는 LUT activation 중심으로 모델링한다. 임의 elementwise/reduction, dtype, accumulator 정밀도, 연산 중 memory 접근 가능 여부를 가정하지 않는다.
- PNM은 GEMM 및 general-purpose kernel을 담당한다. 실제 지원 연산은 설치된 kernel 목록에 의해 결정한다.
- 각 device는 독립적인 주소 공간을 가진다. Unified memory, coherence, peer DMA, device 간 직접 통신을 기본 가정으로 두지 않는다.
- 저장 위치(memory space), 실행 엔진(engine), 물리 표현(layout)을 독립 속성으로 둔다.

예: `device=0, memory=PIM_DRAM, layout=DenseRowMajor`인 tensor를 PNM GEMM이 tile 단위로 읽을 수 있다. `PIM_DRAM`에 저장되어 있다는 사실만으로 PIM GEMV에 적합한 것은 아니다.

## 2. 계층 책임

| 계층 | 소유하는 책임 | 하위 계층에 위임하는 책임 |
|---|---|---|
| L3 | Tensor, 연산 API, PreparedWeight, DeviceMesh/DistributedTensor, KVCache, 사용자 실행 정책 | 검증·배치·전송·실행·수명 관리 |
| L2 | 메모리 pool, tensor 표현/버전 관리, 실행 계획, 의존성, SPM workspace, multi-device 전송·collective 조율 | 실제 allocate/copy/launch/wait와 ISA lowering |
| L1 | capability, layout/kernel requirement 조회, native allocation, transfer, pack 및 연산 실행, completion/error | driver 또는 simulator 호출 |
| L0 | RVV kernel, PIM command sequence | 실제 device execution |

L3의 serving 유틸리티는 L2의 메모리·실행 primitive 위에 작성한다. 요청 admission, batching 정책, tokenizer, 전체 serving scheduler는 초기 library 범위 밖이다.

## 3. 핵심 객체

### L2 내부 객체

- `Device`: device id, topology, capabilities, backend instance.
- `Session`: device와 allocator, 실행 queue, 표현 cache, 오류 상태의 소유자.
- `Buffer`: `(device_id, memory_space, allocation_handle, byte_size, alignment)`. 주소는 opaque handle이며 host pointer로 해석하지 않는다.
- `TensorDesc`: logical shape, dtype. 필요 시 quantization metadata를 포함하되 v1에서 미지원 가능.
- `TensorStorage`: tensor의 논리 데이터 identity, 현재 version, 유효한 physical representation 목록.
- `Representation`: buffer, layout descriptor, valid version, readiness event. Layout descriptor는 physical size, padding, alignment, format id/version과 target compatibility를 포함한다.
- `TensorView`: storage에 대한 논리 slice/shape 정보. Dense strides는 dense 표현에만 적용하며 packed layout은 별도의 mapping을 갖는다.
- `ExecutionPlan`: 선택 kernel, operand representation, 이동·변환, workspace, 의존성 및 backend command 목록.
- `Event`: pending/success/failed 상태. 완료는 command 제출이 아닌 결과의 후속 접근 가능 상태를 뜻한다.

### L3 사용자 객체

- `Tensor`: logical tensor와 preferred placement를 노출한다. 내부 physical representation의 구체적인 bank 주소는 숨긴다.
- `PreparedWeight`: immutable weight의 미리 준비한 표현들을 보유한다. prefill/decode 양쪽용 표현을 선택적으로 유지한다.
- `DeviceMesh`, `DistributedTensor`: device 집합, 전역 shape, 각 local tensor의 shard/replica 정보를 보유한다.
- `KVCache`: layer/request/head/token의 논리 indexing, page table과 page pool, append 및 release 정책을 관리한다.

`DistributedTensor`를 단일 device buffer처럼 취급하지 않는다. 단일-device 연산에 전달하면 명시적인 distributed 연산 경로를 거쳐야 한다.

## 4. 메모리 관리

### PIM memory

Weight, KV cache, activation, temporary buffer를 저장한다. Capacity는 logical element 수가 아니라 padding과 layout packing을 반영한 physical byte 수로 계산한다.

L2는 suballocation/pool을 제공할 수 있지만, PIM 연산용 bank 배치 제약은 L1이 반환하는 allocation requirement를 따라야 한다. 단순 연속 byte allocation으로 충분하다고 가정하지 않는다.

### SPM

기본적으로 PNM kernel의 제한된 workspace로 사용한다. L2는 kernel의 tile/workspace requirement를 받아 실행 중 SPM을 예약한다. 실제 DMA 및 double buffering은 L1/L0의 실행 구현에 맡긴다.

작은 persistent tensor의 SPM 상주는 명시적인 선택으로 제공하되, workspace 용량을 잠식하므로 예산과 실패 조건을 둔다. 전체 weight나 KV cache를 SPM으로 옮기는 것을 기본 경로로 삼지 않는다.

### 수명

Python/C++ 객체가 해제되어도 in-flight command가 참조하는 buffer는 event 완료까지 재사용하지 않는다. 실패한 command가 접근을 중단했는지 확인되지 않으면 해당 allocation을 격리한다. Session 종료는 drain하거나 오류를 보고하며 강제 재사용하지 않는다.

초기 allocator는 capacity 초과 시 필요한 physical bytes와 가용량을 포함한 OOM을 반환한다. Cache eviction은 선택 기능이며 최신 mutable 데이터를 버려서는 안 된다.

## 5. Layout 변환 및 공유

Layout은 `DenseRowMajor`, `DenseColumnMajor`, backend가 제공하는 `PimMatrix(format_id, ...)`처럼 정의한다. Channel/bank/row mapping을 L3 enum 하나에 고정하지 않는다. L1이 shape, dtype, operation role 및 장치 제약으로부터 layout requirement와 pack/unpack 방법을 제공한다.

세 동작을 구별한다.

1. `view/reshape`: 가능한 경우 데이터 이동 없이 논리 metadata만 변경한다. Packed tensor의 임의 view는 거부하거나 materialization을 요구한다.
2. `copy/to`: device 또는 memory space 사이의 이동. 의미상 element 값은 유지한다.
3. `materialize_layout`: 같은 논리 값의 다른 physical representation 생성. Packing/unpacking 비용과 추가 allocation이 발생할 수 있다.

`materialize_layout`은 동일 TensorStorage에 표현을 추가한다. `copy`는 별도 storage를 만들며 이후 변경을 자동 공유하지 않는다. dtype 변환은 별도 `cast` 연산이며 layout 변환에 암묵적으로 포함하지 않는다.

PNM/PIM 간 표현 공유 정책:

- 같은 표현을 두 kernel이 모두 읽을 수 있으면 그대로 공유한다.
- PNM이 packed layout을 직접 읽는 kernel을 가진다면 별도 dense 표현 없이 실행할 수 있다.
- 호환되지 않으면 다른 표현을 materialize한다. 변환은 destination allocation에 수행하고 원본은 유지한다.
- immutable weight는 dense/PIM-packed 표현을 재사용한다. 두 표현 유지 여부는 device memory budget으로 결정한다.
- mutable tensor의 쓰기는 모든 alias에 공유되는 storage version을 갱신하고 다른 표현을 stale 처리한다. Stale 표현은 읽기 전에 다시 생성한다.
- v1은 mutable storage 단위로 읽기/쓰기를 보수적으로 직렬화한다. Pending write에는 새 version과 completion event를 함께 예약하고, 실패하면 해당 version을 유효 데이터로 노출하지 않는다.
- Slicing된 alias의 쓰기도 전체 storage 표현을 무효화한다. 초기에는 packed alias에 대한 in-place 쓰기를 제한한다.

PNM/PIM의 cache flush, memory visibility, mode switch는 변환과 별개다. L2는 hazard를 관리하고 L1이 필요한 barrier를 수행한다. 동시 접근이 안전하다는 capability가 없으면 공유 memory 접근을 직렬화한다.

## 6. 실행 계약

L3 operation → shape/dtype/device 검증 → engine 및 kernel 선택 → representation 검증/준비 → workspace 예약 → dependency 연결 → L1 제출 → completion/event 처리.

- `engine=PNM` 또는 `PIM`은 strict: 해당 engine에서 구현할 수 없으면 명확한 오류를 반환한다.
- `engine=AUTO`만 device 내 fallback을 허용한다. Host compute fallback은 별도 opt-in이며 기본 비활성화한다.
- Strict engine은 수학 연산의 실행 위치를 지정한다. 준비 과정의 copy/pack은 다른 엔진 또는 host를 사용할 수 있으므로 `allow_host_staging`과 `layout_policy`로 별도 제어한다.
- `layout_policy=require_prepared`이면 실행 도중 암묵적 layout 생성은 오류다. 초기 serving 실행은 이 모드가 적합하다.
- AUTO는 v1에서 capability와 명시적 규칙으로 선택한다. 성능 우위는 profiling 전까지 보장하지 않는다.
- Stream은 사용자 수준의 순서와 의존성이다. 실제 hardware queue 또는 병렬 실행을 보장하지 않는다.
- 연산은 asynchronous 제출을 기본으로 하되 Tensor가 readiness를 보유한다. 후속 연산은 자동 의존성을 연결하고 host read는 완료를 기다린다.
- v1에서는 device당 실행을 보수적으로 직렬화해도 API는 async 형태를 유지한다.

오류는 unsupported operation/dtype/layout, invalid shape/device, capacity 부족, backend execution failure를 구분한다. 실행 오류는 wait/host read 및 관련 후속 event에서 관찰 가능해야 한다.

## 7. 연산 범위

| API | 우선 실행 위치 | 조건 |
|---|---|---|
| matmul / linear | PNM | GEMM kernel 및 tile workspace 필요 |
| matvec | PIM | matrix layout, vector 공급, partial sum 회수 조건 확인 |
| add / mul / reduction | PNM | PIM의 MAC tree가 임의 reduction API를 의미하지 않음 |
| RMSNorm / LayerNorm / softmax / RoPE | PNM | 초기 primitive 조합, 이후 fused kernel |
| activation | PNM, 조건부 PIM | 지원 함수/오차/입출력 경로를 capability로 확인 |
| KV append / attention | 초기 PNM | PIM attention은 KV layout 및 증분 갱신 검증 후 추가 |

PNM GEMM과 PIM GEMV는 동일 수학 연산의 서로 다른 구현이지만, 모든 GEMM이 PIM에서 효율적으로 실행된다고 가정하지 않는다. LUT activation의 근사 방식과 accumulator 정밀도도 연산 계약의 일부다.

## 8. L1에 필요한 최소 계약

아래는 의미상의 계약이며 최종 ABI가 아니다.

```cpp
query_device_caps(device);
query_kernel_requirements(op, tensor_descs, engine);
query_layout_requirements(op, operand_role, tensor_desc, engine);
allocate(memory_space, physical_requirements);
release(allocation_handle);
submit_copy(src, dst, dependencies);
submit_layout_transform(src_repr, dst_repr, dependencies);
submit_operation(op, representations, workspace, dependencies);
query_event(event);
wait_event(event);
```

L2는 공통 operation semantics를 넘기고 L1은 해당 backend의 kernel/command sequence로 변환한다. Requirement 조회 결과와 실행은 같은 capability/kernel revision을 사용해야 한다. 미지원 transfer/pack 경로도 명시적으로 응답한다.

Capabilities에는 engine별 op/dtype/accumulation, memory capacity/accessibility, layout formats, SPM requirement, transfer routes, coherence/barrier, PNM/PIM concurrency, peer connectivity를 포함한다.

## 9. L3 API 사용 예시

아래 코드는 제안 API이며 실행 가능한 구현은 아직 없다. PyTorch binding은 L4에서 제공한다.

```python
ctx = ndp.Session(backend="simulator", devices=[0, 1])
d0 = ctx.device(0)

# from_host는 기본적으로 독립 storage로 복사한다.
w = ndp.from_host(host_weight, device=d0, memory="pim_dram")
weight = ndp.prepare_weight(
    w, uses=[("linear", "pnm"), ("matvec", "pim")],
    memory_budget_bytes=weight_budget,
)
x = ndp.from_host(host_input, device=d0, memory="pim_dram")

y = ndp.linear(x, weight, engine="pnm",
               layout_policy="require_prepared")
z = ndp.rms_norm(y, norm_weight, epsilon=1e-5, engine="pnm")
host_result = ndp.to_host(z)  # readiness를 기다리고 dense host 값을 반환
```

추가 primitive는 `empty`, `copy`, `materialize_layout`, `cast`, `wait`를 둔다. `prepare_weight`는 요청한 모든 표현을 예산 내에서 생성하거나 실패하며 조용히 예산을 초과하지 않는다. Kernel 선택에는 operand role 및 transpose semantics도 포함한다.

## 10. 다중 device

L2에는 초기부터 device id와 explicit transfer를 포함한다. 상위 분산 실행 기능은 단일 device 검증 후 확장한다.

- Peer copy가 가능하면 사용하고, 없으면 host staging을 사용한다. 둘 다 없거나 정책상 금지이면 오류다.
- Host staging buffer는 async DMA가 끝날 때까지 유지하고 크기를 제한한다.
- `DeviceMesh`는 device 집합과 연결 경로, `ShardSpec`은 축/범위/replica/partial-sum 상태를 표현한다.
- v1 분산 연산은 explicit `all_reduce(sum)`, `all_gather`, device copy로 한정한다. 고정 순서/shape/dtype을 검증하고 실패 시 그룹 내 관련 작업에 오류를 전달한다.
- Collective의 host fallback과 누적 dtype을 명시한다. 경로별 실행 결과와 비용을 trace에 남긴다.

예: `W[N,K]`를 output dimension N으로 분할하면 `y_i=W_i x`는 output shard다. 다음 연산이 shard를 유지할 수 있으면 gather하지 않는다. K로 분할하면 `y_i=W_i x_i`는 partial sum이므로 all-reduce가 필요하다. Shard와 partial sum을 구분해야 잘못된 결과 조합을 막을 수 있다.

## 11. LLM serving 데이터 정책

- Prefill은 PNM GEMM, decode의 batch가 작은 linear는 PIM GEMV를 우선 후보로 삼는다. 실제 경계는 batch와 shape, 변환/통신 비용을 포함한 측정으로 정한다.
- Weight 준비는 모델 로딩 시 수행한다. Dense/packed 이중 상주의 메모리 비용을 보고하고, 부족하면 지원되는 공유 표현 또는 명시적인 단계 전환 시 재변환을 선택한다.
- KV cache는 고정 token capacity/page pool, append, valid length, page reclamation을 제공한다. 초기에는 PNM-accessible layout 하나를 canonical로 삼는다.
- K와 V는 attention에서 접근 방향이 다르므로 동일 PIM layout을 강제하지 않는다. PIM attention은 page/block 단위 packing과 incremental append 비용을 확인한 후 도입한다.
- v1 KV page는 request별로 소유한다. Prefix sharing/COW는 이후 확장하며 mutable page의 암묵적 공유를 금지한다.
- Request 종료 후에도 attention이 참조 중인 page는 completion 이후 반환한다.
- Hot path에서 weight 재packing, token마다 전체 KV repacking, 불필요한 host round-trip을 trace로 검출한다.

## 12. 구현 순서와 완료 조건

1. **계약과 functional backend**: C++ core의 desc/handle/capability/error 정의 및 host functional backend. 할당·복사·event·오류·수명 검증. 이 backend는 simulator 성능을 대변하지 않는다.
2. **단일 device 실행**: L2 allocator/plan/dependency와 L3 Tensor, PNM matmul, PIM matvec, explicit layout materialization. Reference 값 비교, layout 왕복, padding, alias invalidation, in-flight free 검증.
3. **Serving 기본 경로**: PreparedWeight, PNM norm/softmax/RoPE, KV append 및 한 decoder block. Prefill 후 여러 decode step을 reference와 비교하고 실제 physical memory 사용량 확인.
4. **다중 device**: mesh/shard/transfer/collective 및 두 device linear. Output shard와 partial sum 경로, 제한된 host staging, collective 실패 전파 검증.
5. **실제 simulator 및 FPGA**: 동일 계약의 L1을 연결하고 visibility, dtype 정밀도, memory mapping, resource conflict를 검증. Backend 전환 시 L3 의미가 유지되는지 확인.

실제 simulator 연동이 가능해지는 대로 2~4와 병행한다. 각 dtype/accumulator/LUT 특성에 따라 수치 tolerance를 명시하며 단일 공통 tolerance로 통과시키지 않는다.

## 13. KETI 확인 우선순위

1. PNM이 PIM memory를 직접 load/store할 수 있는가? 접근 가능 영역, bandwidth, cache 유무와 flush/fence 방식은 무엇인가?
2. PIM 연산과 PNM/DMA memory access는 병행 가능한가? 어느 channel/bank/device 단위에서 배타적인가?
3. PIM matrix/vector/result의 위치와 bank mapping, padding/alignment, partial sum 합산 경로는 무엇인가?
4. PIM native dtype/accumulator, LUT 함수 및 정확도, command 단위는 무엇인가?
5. SPM 용량/주소 범위, DMA 경로, RVV 및 toolchain/kernel loading 방법은 무엇인가?
6. Host allocation/transfer/launch/completion/error API와 여러 board의 식별 방법은 무엇인가?
7. Device 간 peer transfer 또는 collective 지원이 있는가? 없다면 host staging이 가능한가?

회신 전에는 위 값을 설정 가능한 capability로 두며 실제 FPGA와 일치한다고 주장하지 않는다.

## 참고

AiM의 matrix-vector, MAC accumulator 및 LUT 기반 activation 개념 확인: SK hynix, *System architecture and software stack for GDDR6-AiM*, Hot Chips 34 (2022): https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf

이 자료는 구조 참고이며 대상 FPGA의 지원 사양을 확정하는 근거로 사용하지 않는다.
