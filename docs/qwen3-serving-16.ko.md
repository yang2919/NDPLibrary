# CXL-PIM 16개에서 Qwen3의 문맥 길이별 TPOT 비교

실험 단위는 **모델 × (문맥 길이, 상주 요청 수) × (PP, TP)**다. 가중치와 KV 캐시가 이미 디바이스에 있는 decode 정상 상태에서, 같은 요청이 다음 토큰을 돌려받기까지 걸리는 시간(TPOT)을 비교한다. 모델 복제본은 한 개이며 모든 구성에서 디바이스 16개를 사용한다. [실험 설정 JSON](../configs/qwen3-serving-16.json)

## 부하를 세 쌍으로 고정한다

| Sequence length S | Batch size B | 상주 KV 토큰 슬롯 S × B |
|---:|---:|---:|
| 1,024 | 128 | 131,072 |
| 2,048 | 64 | 131,072 |
| 4,096 | 32 | 131,072 |

S는 이번 decode에서 attention이 읽는 **현재 토큰 포함 KV 길이**, B는 닫힌 요청 집합에서 계속 디코딩하는 **동시 상주 시퀀스 수**다. 현재 pipeline worker는 한 번에 한 요청의 한 토큰을 처리하므로 stage microbatch는 1이다. 단계 사이에서는 서로 다른 요청을 겹쳐 실행한다. 세 쌍은 논리 KV 용량을 같게 유지하면서 가장 깊은 PP16에서도 단계당 평균 두 요청 이상을 공급하도록 정한 기준 부하다. 길이와 부하가 함께 변하므로 길이 단독 효과는 추론할 수 없다. [실제 스케줄러 정책](pipeline-execution.md)

FP16 KV의 장치당 하한은 `B × S × (layers/PP) × (KV heads/TP) × head_dim × 2(K,V) × 2 bytes`다. 아래 유효 배치에서 dense는 2 GiB, MoE는 768 MiB로 일정하다. 장치당 예산은 32 GiB, 16채널이다. 실험 전 각 장치에 대해 실제 packed weight(모든 expert 포함), KV 예약, 패딩, workspace를 합산한다. 초과한 행은 capacity failure로 남기고 그 행만 B를 낮추지 않는다.

## PP와 TP를 전수 비교한다

각 stage는 연속된 레이어를 균등하게 맡고, stage 안의 TP rank는 온전한 KV head를 분담한다. 표는 **실험 설계상 가능한 배치**다. 실험 runner는 TP별 대표 rank를 측정한다. 실제 체크포인트와 16개 worker를 연결하는 Qwen용 TP 어댑터는 별도 구현 대상이다.

| PP × TP | Dense: 레이어/stage, KV head/rank | MoE: 레이어/stage, KV head/rank |
|---|---|---|
| 16 × 1 | 4, 8 | 3, 4 |
| 8 × 2 | 8, 4 | 6, 2 |
| 4 × 4 | 16, 2 | 12, 1 |
| 2 × 8 | 32, 1 | 제외 · KV head가 4개 |
| 1 × 16 | 제외 · KV head가 8개 | 제외 · KV head가 4개 |

Dense 12개, MoE 9개로 총 **21개** 비교점이다. KV 복제와 expert parallelism은 이 sweep에서 사용하지 않는다. MoE는 레이어마다 router가 선택한 8개 expert 각각을 TP 분할하고 128개 전체 가중치를 상주시킨다. Dense Q 폭은 8192, MoE Q 폭은 4096으로 각각 hidden 폭과 다르다. shard 크기는 실제 projection 차원에서 산출한다. [모델 및 단일 레이어 구성](qwen3-layer-timing.md)

## CENT 코드의 재사용 경계

기존 CENT 검증과 같은 커밋 `4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2`를 고정한다. **gem5가 계산 시간을 공급하고, CENT가 CXL 메시지 비용을 공급하며, 요청과 stage 큐를 재생하는 이벤트 모델이 두 비용을 하나의 시계에서 결합한다.** 이 경로의 결과는 모델 기반 추정치로 기록한다.

| 원본 Python 코드 | 가져올 기능 | Qwen 실험에서 적용하는 방식 |
|---|---|---|
| [cxl_latency.py](https://github.com/Yufeng98/CENT/blob/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/cxl_latency.py) | `vector_latency(size, lanes)`의 메시지 패킹, flit 직렬화, 고정 왕복 지연. 반환 단위는 ms | 외부 checkout에서 함수를 직접 import한다. FP16 N개에는 size=N, FP32 부분합 N개에는 size=2N을 전달한다. 전송 방향과 rank별 실제 메시지 수는 NDP 실행 경로에서 얻는다. |
| [run_sim.py](https://github.com/Yufeng98/CENT/blob/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/run_sim.py) | sequence별 trace 생성·결과 캐시, `update_csv`의 병렬 배치 열거, `load_data_point`의 PIM·CXL·연산 시간 분해 | 같은 키에 모델·TP·프로파일을 추가하고 Qwen shard별 gem5 ROI를 읽는다. PP=16/TP를 명시해 pure PP도 같은 규칙으로 집계한다. CENT의 균형 파이프라인 처리량 `1000 × PP / token_latency_ms`는 무경합 이상 조건의 교차 점검값이다. |
| [scaling_study_various_DP.py](https://github.com/Yufeng98/CENT/blob/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/scaling_study_various_DP.py) | 단일 대표 길이와 장치 그룹별 계산·통신 항목을 비교하는 별도 scaling study | 결과 분해를 대조하는 참고 경로다. 기본 모델 목록이 Llama2이고 DP는 이번 실험에서 1로 고정한다. |

CENT의 `llama_latency`는 Llama용 collective 횟수와 multicast 가정을 내장한다. Qwen 결과에는 더 낮은 수준의 `vector_latency`를 사용한다. 예를 들어 endpoint x4를 가정하면 원본 함수의 dense hidden 5120개 FP16 전송 한 번은 **0.984663 µs**, MoE hidden 2048개는 **0.507826 µs**다. 이는 무경합 메시지 한 건의 모델값이다. 원본 checkout에서 다음과 같이 확인할 수 있다.

```sh
PYTHONPATH="$CENT_ROOT/cent_simulation" python3 -c 'from cxl_latency import vector_latency; print([1000 * vector_latency(n, 4) for n in (5120, 2048)])'
```

**공유 링크 모델.** x4는 본 실험의 장치당 lane 수다. 호스트↔스위치의 방향별 공유 상한은 64 GB/s로 가정한다. 호스트 broadcast는 rank별 unicast로, FP32 reduction은 rank별 read → 호스트 합산 → 필요한 rank로 결과 전송으로 풀어낸다. PP 경계에서는 실제 소유 위치에 따라 host staging한 hidden을 다음 그룹으로 전달한다. 각 메시지의 wire bytes는 `256 × ceil(payload_bytes / 192)`이다. 이벤트 모델은 endpoint와 공유 uplink의 가용 대역폭 중 작은 값으로 직렬화 시간을 계산하고, CENT의 고정 지연을 한 번만 더한다. 무경합에서 `vector_latency`와 일치하는지 확인한다. CPU 합산 시간과 stage 대기는 별도 이벤트다.

## Docker 실행 절차

저장소 루트에서 Docker를 시작한다. 호스트에는 Python 3.9 이상이 필요하다. SE 실행과 CSV 생성은 터미널 하나에서 끝난다. 먼저 MoE의 1K·TP4 한 건으로 빌드, ROI 수집, 보고서 경로를 확인한다.

```sh
python3 tools/gem5/pim_kernel/qwen_serving.py list
python3 tools/gem5/pim_kernel/qwen_serving.py run --model moe --context 1024 --tp 4
```

첫 실행은 필요한 Docker 이미지와 RISC-V gem5를 준비한다. 각 실행이 시작되면 기존 runner가 절대 경로를 `Results:`로 출력한다. 정상 종료하면 `NDP_QWEN_SERVING_REPORT cases=1`과 CSV 경로가 출력된다. 이어서 같은 디렉터리에 전체 실험을 진행한다.

```sh
python3 tools/gem5/pim_kernel/qwen_serving.py run
```

완료 조건은 `NDP_QWEN_SERVING_REPORT cases=21`이다. 기본 결과표는 `build/extended-gem5/qwen-serving/serving.csv`이다. 위에서 완료한 한 건은 재사용한다. 중단되면 같은 명령을 다시 실행한다. runner는 모델·길이·TP와 코드/설정 지문이 일치하는 완료 기록을 읽고 그 원본 로그를 다시 검사한다. 같은 출력 폴더에서는 한 번에 하나의 sweep을 실행한다. gem5 fork, 컴파일 옵션, Docker 이미지가 바뀌면 새 `--output` 디렉터리를 지정한다. `JOBS`, `CPUSET`, `ROI_TIMEOUT`은 기존 run.sh와 같다.

## 결과 읽기와 재집계

| 결과 | 위치 | 해석 |
|---|---|---|
| gem5 직접 측정 | CSV의 `run` 경로 아래 `roi.json`, `sim/stats.txt`, `sim/config.ini` | 한 토큰, 한 레이어, 대표 rank의 attention·FFN ticks, instruction, CPU cycle과 opcode별 PIM 요청 수. 종료 시 dump는 제외한다. |
| 16장치 모델 추정 | `serving.csv`의 `estimated_*` | 정상 상태 관측창에서 같은 요청의 연속 토큰 완료 간격(mean/p50/p95), 전체 tokens/s, stage별 점유율(연산 및 통신 대기 포함). |
| 조건과 트래픽 | 같은 CSV | S, B, PP, TP, rank당 논리 KV bytes, 토큰당 host-staged payload bytes와 TP reduction 횟수, 적용한 host 비용. |

고정 길이로 초기화된 KV를 사용한다. B개의 요청을 stage별 FIFO에 넣고, 각 요청의 토큰이 마지막 stage를 빠져나오면 같은 요청의 다음 토큰을 제출한다. B×2회 토큰 완료를 예열하고 이후 B×8회 완료까지의 간격을 기록한다. 출력 길이와 EOS에 의존하지 않는다. p95는 이 결정적 스케줄의 요청 간격 분포를 의미한다.

호스트 embedding·logits·토큰 선택의 합계 비용과 FP32 벡터 한 번의 호스트 reduction 시간은 기본값 0 µs다. 기본 CSV는 이 비용을 제외한 decoder 추정치다. 호스트에서 측정한 값을 넣으면 gem5 재실행 없이 다시 집계할 수 있다. `report`는 같은 필터의 CSV를 갱신하므로 기존 비교표를 보관하려면 먼저 복사한다.

```sh
python3 tools/gem5/pim_kernel/qwen_serving.py report \
  --host-token-us "$HOST_TOKEN_US" --host-reduce-us "$HOST_REDUCE_US"
```

고정 커밋의 외부 CENT checkout을 직접 사용하려면 `run` 또는 `report`에 `--cent-root "$CENT_ROOT"`를 추가한다. 파일 해시를 확인한 후 원본 `vector_latency`를 호출한다. 기본 경로는 같은 flit 패킹·lane 대역폭·고정 지연을 Python 표준 라이브러리로 계산한다.

## 추정 범위

- guest는 실제 TP별 Q/K/V head 및 FFN 폭으로 한 레이어를 실행한다. 측정 시 한 rank의 zero-weight shard는 독립적으로 계산하며, TP collective는 이후 이벤트 모델에서 처리한다. Attention·FFN 두 ROI를 단계별로 반복하고 같은 그룹 내 rank 계산 시간은 같다고 가정한다. 실제 다중 rank 출력을 교환하는 기능 검증이나 공유 시스템의 16장치 실측은 별개다.
- MoE는 모든 expert shard를 할당하고 고정 top-8을 실행한다. Router를 수행하는 대표 rank 시간을 임계 경로로 사용한다. 이벤트 모델은 routing 결과를 다른 rank에 전달하고, 각 rank가 expert 결과를 가중합한 뒤 레이어당 attention·FFN 각각 한 번씩 FP32 reduction한다고 가정한다. 통신과 계산의 세부 중첩 및 실제 expert 선택 편차는 측정하지 않는다.
- 장치 간 통신은 방향별 FIFO uplink와 endpoint 전송의 cut-through 근사다. CPU 비용은 한 공유 호스트 자원에서 직렬화하며 토큰당 embedding/output 합계를 마지막 stage에 부과한다. 실제 요청의 텍스트 생성, 분포 기반 라우팅, OS·런타임 지연, 구성별 전체 모델의 실할당량은 별도로 검증해야 한다.
