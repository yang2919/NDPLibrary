# Single-device NDP Library L2/L3 구현 계획

상태: 구현을 시작하기 위한 설계 제안. 이 문서의 범위와 결정은 기존 `l2-l3-design.md`의 일반 설계보다 우선한다. 아래 C++ 선언은 책임과 계약을 나타내는 interface 초안이며 컴파일 가능한 헤더가 아니다.

## 1. 구현 목표와 범위

**완료 목표:** 하나의 Session가 하나의 NDP device를 관리하고, device-resident tensor를 PNM/PIM 연산에서 사용하며, 필요한 layout 변환과 메모리 수명을 runtime이 관리한다. 고정 크기 decoder block의 prefill과 반복 decode를 수행할 수 있어야 한다.

하드웨어 모델은 PNM(RISC-V + VPU), scratchpad(SPM), off-chip PIM memory, PIM compute engine이다. PNM의 PIM memory 접근 방식은 backend capability로 확인한다.

초기 구현 결정을 다음과 같이 고정한다.

| 항목 | 결정 |
|---|---|
| Core 언어/빌드 | C++20, CMake. L2/L3는 PyTorch에 의존하지 않음 |
| 실행 단위 | Session 하나에 device 하나, 내부 FIFO 실행 queue 하나 |
| API 제출 | host thread 하나에서 제출. thread-safe 동시 제출은 초기 범위 밖 |
| 동작 | 비동기 반환, Tensor/Event를 통한 대기. 실제 device 작업은 직렬 실행 |
| 메모리 | 일반 Tensor는 PIM_DRAM에 저장. SPM은 kernel workspace 전용 |
| 엔진 선택 | 호출자가 PNM 또는 PIM을 지정. AUTO/성능 기반 배치는 후속 기능 |
| Tensor 수정 | 공개 Tensor는 연산으로 수정하지 않음. 연산은 새 Tensor 반환 |
| 특수 mutable 객체 | KVCache 내부 저장소만 제한적인 append/reset 지원 |
| Layout | dense row-major + backend가 정의하는 packed layout |
| View | public strided/sliced alias는 미지원. 필요 시 명시적인 copy |
| 수치 | 기본 Tensor 저장 dtype은 FP16(IEEE binary16). 향후 연산 reference의 누적은 FP32로 정의하고 BF16은 후속 지원. 실제 device 지원은 capability에 따름 |
| Host 연동 | contiguous host buffer import/export. PyTorch custom op 및 binding은 별도 L4 작업 |
| 분산 | DeviceMesh, sharding, collective, device 간 전송은 구현하지 않음 |

Tensor의 공개 불변성으로 일반 tensor의 stale layout 문제를 우선 제거한다. 일반 in-place 연산과 복잡한 alias/version coherence는 이후 확장한다. 단, 작업 중인 output의 readiness와 KVCache의 변경 상태는 처음부터 관리한다.

FunctionalBackend는 host에서 NDP 의미를 검증하는 테스트용 backend다. 실제 simulator나 FPGA의 처리량/지연시간을 재현하지 않으며, 실제 backend 미지원 연산을 숨기는 host fallback으로 사용하지 않는다.

## 2. 코드 구조와 의존 방향

```text
include/ndp/
  types.hpp              # shape, dtype, engine, status
  session.hpp            # L3 Session 및 Options
  tensor.hpp             # L3 Tensor, HostTensor
  ops.hpp                # L3 수학 연산
  prepared_weight.hpp    # L3 불변 weight 준비
  kv_cache.hpp            # L3 제한적 mutable cache
  backend/backend.hpp    # L1과 L2 사이 계약
src/runtime/
  runtime_state.*
  memory_manager.*
  tensor_record.*
  layout_manager.*
  planner.*
  executor.*
  event.*
src/api/
  session.* tensor.* ops.* prepared_weight.* kv_cache.*
src/backends/functional/
  backend.* layouts.* kernels.*
tests/
  contracts/ memory/ layout/ execution/ ops/ serving/
examples/
  single_device_linear.cpp
  decoder_block.cpp
```

의존 방향은 `L3 → L2 → IBackend`로 고정한다. L2는 PyTorch 타입, RVV opcode, MMIO 주소 및 FPGA driver header를 include하지 않는다. Backend 선택은 Session 생성 시 주입한다.

## 3. 공통 값 타입

```cpp
enum class Engine { PNM, PIM };
enum class MemorySpace { PimDram, Scratchpad };
enum class DType { F16, F32, BF16 };
enum class LayoutPolicy { AllowMaterialize, RequirePrepared };

struct Shape { std::vector<uint64_t> dims; };
struct TensorDesc { Shape shape; DType dtype; };
struct OpOptions {
  Engine engine;
  LayoutPolicy layout_policy = LayoutPolicy::AllowMaterialize;
  bool allow_host_staging = false;
};
```

- v1 shape는 양의 dimension만 지원하고 rank-0/zero-size tensor는 명확한 unsupported 오류로 반환한다.
- element 수, physical byte 수 및 offset 계산은 overflow를 검사한다.
- `Status/Result<T>`는 프로젝트 내부 타입으로 작성한다. C++20에 `std::expected`가 있다고 가정하지 않는다.
- 오류 코드: InvalidArgument, SessionMismatch, UnsupportedOperation, UnsupportedDType, UnsupportedLayout, LayoutNotPrepared, OutOfMemory, BackendFailure, DependencyFailed, SessionClosed.
- shape/dtype/engine 오류는 제출 전에 반환하고, 실제 실행 실패는 Event에서 반환한다.
- 정밀도 계약에는 입력/출력 dtype, 누적 dtype, 반올림 및 LUT 근사 여부를 포함한다. 지원하지 않는 dtype을 암묵적으로 cast하지 않는다.

## 4. L3 객체 abstraction

### 4.1 Session

**의미:** device 하나를 사용하는 세션 및 모든 API의 진입점.

```cpp
class Session {
 public:
  static Result<Session> create(std::unique_ptr<IBackend>, SessionOptions);
  Result<Tensor> from_host(HostTensorView);
  Result<HostTensor> to_host(const Tensor&); // blocking
  Result<void> synchronize();
  Result<void> close();                    // drain 후 제출 차단
  MemoryStats memory_stats() const;
  DeviceCaps capabilities() const;
};
```

Session은 move-only wrapper이고 RuntimeState를 보유한다. Tensor의 BufferLease는 backend와 메모리 통계를 가진 Resources만 공유하며 worker나 RuntimeState를 유지하지 않는다. Session이 파괴되어도 살아 있는 Tensor의 buffer가 먼저 해제되어서는 안 된다. 명시적인 `close()`에서 drain/error를 확인하며 destructor는 예외를 던지지 않는다. Close 후 새로운 실행 요청은 거부한다.

### 4.2 Tensor

**의미:** 변경되지 않는 논리적 n차원 데이터. 저장된 모든 physical representation은 동일 값을 나타낸다.

```cpp
class Tensor {
 public:
  const TensorDesc& desc() const;
  Result<void> wait() const;
  TensorStatus status() const;
  Result<Tensor> clone(Session&) const;
 private:
  std::shared_ptr<TensorRecord> record_;
};
```

- Tensor 복사는 handle 공유다. `clone()`은 별도 TensorRecord/Buffer를 만든다.
- raw device pointer, mutable data pointer, 공개 free를 제공하지 않는다.
- 다른 Session의 Tensor와 연산하면 동일 device 번호여도 SessionMismatch를 반환한다.
- Public `empty()`는 제공하지 않는다. 미초기화 tensor는 runtime의 output allocation 내부 상태로만 존재한다.
- Tensor는 readiness event를 가진다. Kernel 실행 전 반환되어도 후속 연산이 해당 event에 의존한다.
- Shape나 stride를 임의 변경하는 public view는 v1에서 제외한다. 연산 descriptor가 transpose 의미를 표현한다.

### 4.3 PreparedWeight

**의미:** 특정 연산 용도를 위해 physical representation을 미리 준비한 immutable weight.

```cpp
struct WeightUse { OpKind op; OperandRole role; Engine engine; };
struct WeightPrepareOptions {
  std::vector<WeightUse> uses;
  uint64_t max_resident_bytes; // 원본을 포함한 고유 buffer 총량 상한
  bool allow_host_staging;
};
Result<PreparedWeight> prepare_weight(Session&, const Tensor&,
                                      WeightPrepareOptions);
```

PreparedWeight는 원본 TensorRecord와 준비한 LayoutKey 집합을 보유한다. 동일 layout은 중복 저장하지 않는다. 준비 중 실패하면 새로 만든 임시 표현을 공개하지 않고 정리한다. 원래 Tensor의 유효 표현은 보존한다. 이 API는 초기 버전에서 blocking으로 완료 후 반환한다.

원본과 PIM layout을 둘 다 유지할 때 실제 physical bytes를 예산에 포함한다. 이미 존재하던 표현도 해당 weight의 resident 사용량으로 계산한다. v1은 자동 eviction을 하지 않는다.

### 4.4 KVCache

**의미:** decoder layer 하나, request 하나의 K/V 저장소. v1에서는 고정 capacity의 dense buffer를 사용한다.

```cpp
struct KVCacheDesc {
  uint64_t capacity_tokens, kv_heads, head_dim;
  DType dtype;
};
class KVCache {
 public:
  static Result<KVCache> create(Session&, KVCacheDesc);
  Result<Event> append(const Tensor& k, const Tensor& v);
  uint64_t scheduled_length() const;
  uint64_t committed_length() const;
  Result<Event> reset();
};
```

- K/V logical shape는 각각 `[capacity_tokens, kv_heads, head_dim]`, append 입력은 `[Tnew, kv_heads, head_dim]`이다.
- 새 영역 예약은 제출 시, committed length 갱신은 K와 V 쓰기 모두 성공한 후 수행한다.
- 연속 append는 scheduled length를 사용해 겹치지 않는 범위를 예약한다. Capacity 초과는 제출 전에 실패하며 상태를 바꾸지 않는다.
- Attention은 호출 시점의 visible length와 last mutation Event를 캡처한다. 이후 append가 과거 attention의 범위를 바꾸지 않는다.
- 초기 실행 queue의 순서로 read/append/reset을 직렬화한다. Cache 내부 Buffer/일반 mutable Tensor를 사용자에게 노출하지 않는다.
- Append가 부분 실패하면 cache를 Poisoned로 표시하고 dependent attention을 실패시킨다. backend가 계속 사용 가능하면 `reset()` 후 재사용할 수 있다.
- Reset은 앞선 접근 종료 후 generation과 길이를 초기화한다. Fatal backend 오류가 있으면 Session를 다시 생성한다.
- Page allocator, prefix sharing, eviction, PIM KV layout은 후속 단계다. 고정 capacity의 실제 사용 bytes를 stats에 표시한다.

## 5. L2 객체 abstraction

### 5.1 RuntimeState

**소유:** IBackend, MemoryManager, LayoutManager, Planner, Executor, session id, shutdown/error 상태.

TensorRecord는 BufferLease를 통해 Resources를 shared ownership으로 유지한다. RuntimeState는 TensorRecord 전체를 강하게 보관하지 않는다. 실행 queue가 TensorRecord를 잡고 영구 cycle을 만들지 않도록 command는 BufferLease/EventState 같은 실행 자원만 유지한다. Worker는 shared RuntimeState 자체를 캡처하지 않으며 종료 시 drain/join한다.

### 5.2 Buffer와 MemoryManager

**Buffer의 의미:** shape나 dtype이 없는 native allocation 소유권.

```cpp
struct AllocationRequest {
  MemorySpace space;
  uint64_t bytes, alignment;
  AllocationClass allocation_class; // dense 또는 backend placement 제약
};
class MemoryManager {
 public:
  Result<BufferLease> allocate(const AllocationRequest&);
  MemoryStats stats() const;
};
```

BufferLease는 allocation의 shared RAII handle이다. BackendAllocationHandle, physical bytes, memory space 및 backend lifetime을 가진다. Operation이 완료/접근 중단될 때까지 command가 lease를 잡는다.

v1 allocator는 L1 native allocation을 그대로 사용하고 suballocation/pool은 구현하지 않는다. 이는 임의 byte range 분할로 bank mapping 제약을 깨는 것을 방지한다. Stats는 live tensor bytes, pending temporary bytes, workspace bytes, peak 및 backend allocated bytes를 구분한다. 단순 합산이 가능한 항목과 중복 집계 항목을 문서화한다.

실패 후 device 접근 중단이 확인되지 않으면 lease를 quarantine에 보관하고 Session를 fatal 상태로 전환한다. Timeout은 완료로 간주하지 않는다. Backend teardown으로 접근 중단을 확인하기 전에는 allocation을 재사용하지 않는다.

### 5.3 TensorRecord와 Representation

```cpp
struct Representation {
  LayoutKey key;
  LayoutDesc layout;
  BufferLease buffer;
  Event ready;
};
struct TensorRecord {
  SessionId session;
  TensorId id;
  TensorDesc logical;
  std::shared_ptr<Resources> resources; // backend 및 메모리 자원; 실행 queue 제외
  std::map<LayoutKey, Representation> representations;
  Event value_ready;
};
```

TensorRecord는 일반 Tensor용이며 mutable KVCache와 분리한다. Cache 저장소에는 별도 CacheState와 generation/last-access event를 사용한다. 이 선택으로 일반 tensor의 version 전파 구현을 v1에서 생략할 수 있다.

### 5.4 LayoutDesc와 LayoutManager

**LayoutDesc:** logical index를 physical storage로 해석하는 규칙. Dense의 경우 stride 정보를, packed의 경우 backend format과 packing parameter를 가진다.

```cpp
struct LayoutDesc {
  LayoutKey key; // format/version/device profile/shape/dtype/packing parameters
  uint64_t physical_bytes, alignment;
  AllocationClass allocation_class;
  BackendLayoutPayload payload;
};
class LayoutManager {
 public:
  Result<RepresentationRef> find_compatible(TensorRecord&, LayoutRequirement);
  Result<RepresentationRef> ensure(TensorRecord&, LayoutRequirement,
                                   PreparationPolicy);
};
```

- LayoutRequirement는 특정 operand 역할에서 kernel이 허용하는 표현이다. 단순 `PNM layout`/`PIM layout` 두 종류로 환원하지 않는다.
- 호환성/physical size는 backend가 판단한다. Runtime이 packed data를 dense pointer로 해석하지 않는다.
- Pending representation도 map에 등록해 중복 요청을 하나로 합치고 ready event를 공유한다. 변환 실패 시 해당 표현을 제거/실패 표시한다.
- 변환은 원본을 덮어쓰지 않고 새 buffer를 생성한다. DType은 유지한다.
- 새 표현은 TensorRecord가 살아 있는 동안 보존한다. 자동 cache eviction은 v1에서 제외한다.
- FunctionalBackend의 test packed format은 K tile과 padding을 가진 가역 format으로 정의한다. 실제 AiM bank mapping으로 주장하지 않는다.
- Host 기반 pack이 필요하면 명시적인 host-staging 정책과 경로가 있어야 한다. FunctionalBackend의 host 계산 자체는 staging 정책과 구분한다.

일반 Tensor도 명시적으로 layout을 준비할 수 있도록 고급 L3 API를 제공한다.

```cpp
Result<Event> materialize_layout(Session&, const Tensor&,
                                 const LayoutDesc& target,
                                 PreparationPolicy);
```

Target은 backend의 operation requirement 조회로 얻은 opaque descriptor다. 사용자가 임의 bank 주소를 만들지 않는다. Session/backend profile, logical shape, dtype 일치 여부를 검증한다. 이 함수는 같은 TensorRecord에 표현을 추가하며 논리 값이나 Tensor identity를 바꾸지 않는다. 동일 표현이 이미 있으면 기존 readiness를 반환한다. 일반 연산 사용자는 AllowMaterialize 또는 PreparedWeight를 통해 이 세부 interface를 직접 사용하지 않아도 된다.

### 5.5 Planner와 ExecutionPlan

**Planner:** 수학적 연산 요청을 실제 실행 가능한 순서로 변환한다. Kernel을 실행하지 않는다.

```cpp
struct OpRequest {
  OpKind kind;
  OpAttributes attributes;  // typed variant: epsilon/axis/transpose 등
  std::vector<TensorRef> inputs;
  OpOptions options;
};
struct ExecutionPlan {
  std::vector<PlanStep> steps; // Copy, Transform, Kernel
  std::vector<OutputSpec> outputs;
  uint64_t persistent_bytes, transient_upper_bound;
};
Result<ExecutionPlan> Planner::build(const OpRequest&);
```

구현 알고리즘:

1. Session/shape/dtype/attributes를 검증하고 출력 logical shape를 계산한다.
2. 지정 engine의 kernel candidate를 조회한다. 없으면 오류다.
3. 각 operand의 허용 layout과 필요한 output layout을 조회한다.
4. 기존 표현을 우선 선택한다. 없으면 AllowMaterialize에서만 transform step을 추가한다.
5. Output 및 변환용 persistent buffer를 할당한다. 계획 준비 실패 시 새 자원을 회수한다.
6. Workspace 크기와 capacity를 검증하고 의존 event를 연결한다.
7. 전체 작업을 FIFO에 넣고 output TensorRecord/Event를 반환한다.

`RequirePrepared`는 입력의 추가 materialization을 금지한다. Output allocation 및 kernel 내부의 정상적인 tile DMA는 허용한다. Strict engine은 수학 연산 위치를 뜻하며 preparation 경로는 별도 정책으로 검사한다.

### 5.6 Executor, Event와 Workspace

Executor는 PlanStep들을 하나씩 실행한다. v1에서는 이전 backend ticket이 terminal 상태이고 접근 중단이 확인된 뒤 다음 step을 시작한다. 미래의 parallel scheduler용 DAG 구현은 하지 않는다.

Event 상태는 `Pending → Running → Succeeded/Failed`다. Dependency 실패 시 해당 step은 backend에 제출하지 않고 DependencyFailed로 종료한다. `wait()`는 성공 또는 구체적인 Status를 반환한다.

SPM은 session 시작 시 backend가 사용 가능한 workspace arena를 예약한다. Plan에는 arena 요구량만 기록하고 실행 직전에 slice를 빌린다. 직렬 실행이므로 완료 후 같은 arena를 재사용한다. Queue에 계획이 많이 쌓였다는 이유로 SPM을 계획마다 할당하지 않는다. Firmware/kernel이 이미 점유한 영역은 backend가 가용량에서 제외한다.

Kernel이 요구하는 SPM이 가용량보다 크면 L1의 더 작은 tile candidate를 선택할 수 있다. 모든 candidate가 실패하면 제출 전에 OOM/unsupported를 반환한다. DRAM 임시 workspace는 실행 step에서 할당할 수 있으며 실패는 Event로 전달한다.

PNM/PIM mode 전환, cache flush/invalidate, DMA 완료 및 결과의 memory visibility는 L1의 completion 계약에 포함한다. FIFO만으로 coherence를 보장한다고 가정하지 않는다.

## 6. L1 최소 interface

```cpp
class IBackend {
 public:
  virtual Result<DeviceCaps> capabilities() = 0;
  virtual Result<KernelPlan> query_operation(const OperationSpec&) = 0;
  virtual Result<TransformPlan> query_transform(const LayoutDesc& src,
                                               const LayoutDesc& dst) = 0;
  virtual Result<BackendAllocation> allocate(const AllocationRequest&) = 0;
  virtual Status release(BackendAllocationHandle) = 0;
  virtual Result<Ticket> submit_copy(const CopyCommand&) = 0;
  virtual Result<Ticket> submit_transform(const TransformCommand&) = 0;
  virtual Result<Ticket> submit_kernel(const KernelCommand&) = 0;
  virtual Completion poll(Ticket) = 0;
  virtual Completion wait(Ticket) = 0;
  virtual Status shutdown() = 0;
  virtual ~IBackend() = default;
};
```

KernelPlan에는 stable kernel id, engine, 입력별 LayoutRequirement, output LayoutDesc, allocation 제약, workspace 크기/정렬, numerical contract가 들어간다. Operand layout들이 서로 제약을 가지면 kernel candidate 전체를 함께 선택한다.

Completion에는 실행 status와 `accesses_quiesced`를 포함한다. 성공은 후속 PNM/PIM/DMA/host transfer가 결과를 읽을 수 있는 상태를 뜻한다. 실패해도 접근 중단 여부를 별도로 알려야 한다. Shutdown은 in-flight 접근 중단/장치 분리를 완료해야 한다.

CopyCommand는 H2D/D2H/D2D를 명시한다. Host buffer 수명을 command가 보유하도록 owned staging을 사용한다. `from_host()`는 호출 중 host 입력을 owned staging으로 복사하므로 반환 후 caller가 원본을 변경해도 안전하다. `to_host()`는 결과 준비와 D2H 완료 후 소유권 있는 HostTensor를 반환한다. Pending staging 사용량에 상한을 둔다.

Backend implementation은 KV append의 범위 쓰기와 attention의 strided cache 읽기에 필요한 bounded buffer region descriptor를 해석한다. 이는 public mutable TensorView와 별개인 내부 계약이다.

## 7. L3 연산 API와 수학적 계약

```cpp
Result<Tensor> matmul(Session&, const Tensor& a, const Tensor& b, OpOptions);
Result<Tensor> matvec(Session&, const Tensor& w, const Tensor& x, OpOptions);
Result<Tensor> linear(Session&, const Tensor& x, const PreparedWeight& w,
                      const std::optional<Tensor>& bias, OpOptions);
Result<Tensor> add(Session&, const Tensor&, const Tensor&, OpOptions);
Result<Tensor> mul(Session&, const Tensor&, const Tensor&, OpOptions);
Result<Tensor> rms_norm(Session&, const Tensor&, const Tensor& gamma,
                        float epsilon, OpOptions);
Result<Tensor> softmax(Session&, const Tensor&, int axis, OpOptions);
Result<Tensor> silu(Session&, const Tensor&, OpOptions);
```

| 연산 | v1 입력/출력 계약 | 엔진 |
|---|---|---|
| matmul | `[M,K] × [K,N] → [M,N]`, rank-2 | PNM |
| matvec | `W[N,K] × x[K] → y[N]` | PIM 우선, PNM도 reference 경로 제공 |
| linear | `x[M,K], W[N,K] → y[M,N]`, 의미는 xWᵀ | PNM, PIM은 M=1만 |
| add/mul | 동일 shape 또는 마지막 축 `[D]` broadcast만 | PNM |
| rms_norm | x `[...,D]`, gamma `[D]`, epsilon > 0, 마지막 축 | PNM |
| softmax | 마지막 축만, max subtraction을 사용 | PNM |
| silu | 동일 shape의 elementwise 결과 | PNM |

Linear의 bias는 `[N]`이다. PIM fused bias가 없으면 초기 PIM linear는 bias 없는 경우만 허용한다. 사용자가 PNM add를 별도로 호출할 수 있다. 연산 위치를 조용히 바꾸지 않는다.

RoPE와 attention은 serving 단계에서 추가한다.

- RoPE: Q `[T,Hq,D]`, K `[T,Hkv,D]`, 명시적인 token positions 및 sin/cos table. v1은 even D, interleaved pair convention을 고정하며 다른 convention은 거부한다.
- Attention: Q `[Tq,Hq,D]`, cache의 K/V 및 query 시작 position. Hq는 Hkv의 배수여야 한다. Scale `1/sqrt(D)`, causal mask와 GQA head mapping을 명시한다.
- Prefill은 K/V를 append한 뒤 Q token별 causal mask를 적용한다. Decode는 새 K/V를 append한 뒤 해당 token까지 접근한다.
- PNM용 primitive 합성 또는 전용 kernel로 구현한다. 초기 PIM attention/LUT activation은 필수가 아니다. PIM activation은 실제 capability와 수치 계약이 확인된 뒤 추가한다.

Projection의 `[T,H*D]` 출력과 attention의 `[T,H,D]` 입력 연결은 runtime 내부의 검증된 dense reshape descriptor로 처리한다. Public alias API를 추가하지 않으며, packed representation은 바로 reshape하지 않는다. 최종 attention 출력의 head 병합도 같은 규칙을 따른다. LayerNorm 및 범용 reduction은 후속 연산으로 두고 초기 decoder 검증은 RMSNorm을 사용하는 구조로 한정한다.

## 8. 대표 실행 흐름

### 8.1 Weight 공유 및 layout 준비

1. `from_host(W)`가 dense buffer와 TensorRecord를 만든다.
2. `prepare_weight()`가 PNM linear와 PIM matvec의 requirement를 조회한다.
3. PNM이 dense를 사용할 수 있으면 재사용한다. PIM에 packing이 필요하면 별도 allocation/transform을 수행한다.
4. 두 표현은 동일 TensorRecord에 연결한다. PreparedWeight가 이 record를 유지한다.
5. 이후 PNM/PIM 실행은 호환 표현을 선택하고 변환 없이 실행한다.

### 8.2 PIM 결과를 PNM에서 사용

1. PIM matvec가 결과용 representation과 ready event를 가진 Tensor를 반환한다.
2. PNM rms_norm planner가 output layout을 검사한다.
3. 호환되면 같은 buffer를 사용하고 PIM completion에 의존한다.
4. 호환되지 않으면 AllowMaterialize에서 transform을 넣는다. RequirePrepared이면 준비 필요 오류를 반환한다.
5. Host를 경유해야 하는 경로는 allow_host_staging=false에서 실패한다.

### 8.3 반복 decode

Weight는 미리 준비한다. Cache는 한번 할당한다. 각 step은 embedding 입력 → RMSNorm → Q/K/V projection → RoPE → cache append → causal attention → output projection/residual → RMSNorm → SwiGLU FFN/residual 순서로 구성한다.

Output projection/FFN의 M=1 linear는 준비된 PIM 경로를 선택할 수 있다. Intermediate는 device에 유지하며 마지막 검증/필요 출력에만 to_host를 호출한다. 실제 SPM/PIM memory 경로 미지원은 capability 오류로 드러낸다.

## 9. 구현 작업 분해와 완료 조건

| 단계 | 산출물/작업 | 선행 | 완료 조건 |
|---|---|---|---|
| S0 계약/빌드 | CMake, types/status, public header, IBackend descriptor | 없음 | 빈 library와 API smoke test 빌드, shape overflow 및 dtype 오류 검사 |
| S1 메모리/host I/O | FunctionalBackend allocate/copy, MemoryManager, Session, Tensor, owned staging | S0 | host→device→host 일치, OOM/session mismatch, caller buffer 수명 검사 |
| S2 비동기 실행 | Event, FIFO Executor, BufferLease, close/failure/quarantine 계약 | S1 | pending tensor 연결, 조기 handle 해제, 지연 backend 및 실패 전파 검증 |
| S3 Layout | LayoutDesc/Manager, test pack/unpack, representation dedup | S2 | tile 경계 전후 shape의 왕복 일치, padding/physical bytes, 중복 변환 방지 |
| S4 기본 compute | Planner, PNM matmul/PIM matvec/linear, SPM arena, PreparedWeight | S3 | reference 비교, PNM→PIM→PNM 연결, RequirePrepared 및 예산 검증 |
| S5 LLM primitive | add/mul/RMSNorm/softmax/SiLU/RoPE, FP16 저장/FP32 누적 reference | S4 | 각 연산 수학 계약 및 누적/반올림 오차 검증 |
| S6 KV/decoder | 고정 capacity KVCache, attention, decoder example | S5 | prefill+최소 3 decode step reference 비교, append/reset/failure 검사 |
| S7 실제 backend 검증 | simulator adapter와 동일 contract test | S4 이후 시작 가능 | 실제 simulator에서 S4 및 지원되는 S5/S6 실행, 미지원 목록 명시 |

각 단계는 헤더 계약 → 구현 → 해당 단계의 의미 있는 검증 → example 순서로 진행한다. 실제 backend 준비가 늦어져도 S0~S6는 FunctionalBackend에서 진행할 수 있다. S0~S6 통과는 L2/L3 기능 검증이며 실제 NDP device 검증 완료는 S7을 요구한다. FPGA adapter는 사양 확보 후 별도 작업이다.

권장 첫 구현 작업 묶음은 S0+S1이다. 첫 PR/변경의 목표를 **Session 생성 → host tensor 업로드 → 다운로드 → 안전한 종료**로 두고, 다음 작업에서 async와 layout을 추가한다.

## 10. 검증 시나리오와 계측

수치 reference는 backend kernel 구현을 재사용하지 않는 독립적인 scalar 계산으로 작성한다. FP16은 저장 round-trip 및 FP32 누적을 명시하고, 연산별 tolerance를 근거와 함께 둔다.

필수 시나리오:

1. 비정렬 K/N shape에서 pack→unpack 값 보존과 padding 영역 비노출.
2. 같은 immutable weight를 PNM/PIM이 반복 사용하며 두 번째 호출부터 weight packing이 발생하지 않음.
3. 출력 Tensor가 pending인 상태에서 다음 연산을 호출해도 결과가 올바름.
4. 입력/출력 handle을 일찍 파괴해도 backend completion 이전에 allocation이 해제되지 않음.
5. Queue에 여러 GEMM을 넣어도 SPM arena가 요청 수에 비례해 증가하지 않음.
6. Copy/transform/kernel 각각에 실패를 주입해 dependent operation이 실행되지 않고 오류를 반환함.
7. K 또는 V 쓰기 중 실패하면 committed length가 증가하지 않고 cache가 poisoned 상태가 됨.
8. Prefill+decode 결과가 동일 convention의 reference와 일치하고 KV capacity를 넘으면 제출 전에 실패함.
9. 반복 request 실행/종료 시 cache·activation allocation이 회수되고 weight만 남음.
10. Host staging 금지 시 device 내 경로가 없는 변환을 명확히 거부함.

Trace 필드: op id, engine, tensor/layout id, allocation bytes, transform/copy bytes, host staging bytes, workspace 요구량, submit/completion/error. FunctionalBackend의 시간은 hardware 성능 지표로 해석하지 않는다.

## 11. 사양 미확정 항목의 처리

| 미확정 정보 | 구현 시 처리 |
|---|---|
| 실제 PIM format/bank mapping | opaque LayoutDesc와 backend pack/unpack에 격리 |
| PNM의 PIM memory 접근 | memory accessibility/transfer route capability, 미지원이면 오류 |
| SPM 크기/예약 영역 | backend가 usable arena와 kernel workspace requirement 제공 |
| 실제 dtype/LUT 함수 | per-operation numerical capability, 암묵적 대체 금지 |
| PNM/PIM coherence | backend completion/barrier 계약, 실제 adapter에서 검증 |
| 연산 제출 및 완료 API | IBackend ticket으로 감싸고 L2/L3는 유지 |

위 값이 미정이라는 이유로 상위 abstraction 구현을 중단하지 않는다. 대신 FunctionalBackend의 설정값과 실제 하드웨어 사양을 명시적으로 구분한다.

## 구현 현황: S2

- `Context` 명칭은 `Session`으로 변경했다. Tensor 기본값과 현재 지원 dtype은 F16이다.
- S0/S1과 S2의 FIFO worker, Event, dependency failure, owned staging 및 drain-on-close를 구현했다.
- Host 입력은 native-endian uint16_t 단위의 IEEE binary16 bit pattern이다. Float32에서의 암묵적 변환은 없다.
- `from_host()`는 staging 복사/용량 예약 후 비동기 반환한다. 실제 backend allocation/upload 오류는 Event로 전달한다.
- `to_host_async()`는 HostTransfer를 반환하며 `wait()`는 독립적인 HostTensor 사본을 반환한다. `to_host()`는 blocking 편의 API다.
- live_bytes/allocations에는 worker가 아직 native allocation을 만들지 않은 예약도 포함한다. Staging 통계는 진행 중인 upload/download buffer를 센다. 완료된 HostTransfer/HostTensor의 사용자 보유 메모리는 제외한다.
- synchronize/close는 첫 실행 오류를 유지하여 반환한다. 독립 작업은 계속 실행하고 실패한 Tensor를 읽는 dependent 작업만 건너뛴다. 반복 close도 기존 종료 오류를 반환한다.
- 현 IBackend는 worker가 호출하는 blocking adapter다. 성공/실패/예외 모두 device 접근이 중단된 뒤 반환해야 한다. Native asynchronous ticket 및 불확실한 완료의 quarantine은 실제 adapter 도입 시 별도 계약 확장이 필요하다.
- Worker command는 BufferLease/Resources만 보유하며 Session 실행 상태를 참조하지 않는다. Session 종료 후 Tensor는 backend 자원만 유지한다.
- S3 layout 변환 및 S4 compute는 아직 구현하지 않았다.
