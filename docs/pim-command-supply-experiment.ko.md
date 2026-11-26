# PIM 명령 공급과 컨트롤러 발행 병목을 구분하는 실험

**질문은 “같은 GEMV 명령열을 더 빨리 공급하면 마지막 결과가 더 일찍 준비되는가?”이다.** 16개 디바이스의 서빙 큐를 재생할 필요 없이, 확장 gem5 SE Timing에서 디바이스 하나와 RISC-V 코어 하나로 검증한다.

## A 단계 실행 · Docker와 터미널 하나

저장소 루트에서 Docker를 시작하고 작은 GEMV를 먼저 실행한다. runner가 기존 빌드 캐시를 사용해 RISC-V ELF와 PIM trace 계측을 포함한 gem5를 준비한다.

```sh
STAGE=SUPPLY SUPPLY_CASE=small ./tools/gem5/pim_kernel/run.sh
```

`NDP_SUPPLY_A_PASS case=small trials=3`가 나오면 같은 조건으로 gate 하나를 측정한다.

```sh
STAGE=SUPPLY SUPPLY_CASE=gate ./tools/gem5/pim_kernel/run.sh
```

두 번째 완료 표시는 `NDP_SUPPLY_A_PASS case=gate trials=3`이다. 각 명령이 출력하는 `Results:` 경로가 해당 실험 디렉터리다. `ROI_TIMEOUT`은 선택적인 wall-clock 제한(초)이며 기본값은 무제한이다. gate는 모든 가중치를 초기화하므로 첫 통계가 나오기까지 시간이 걸릴 수 있다. 예열은 양수 입력으로 수행하고 측정 입력의 부호를 번갈아 바꿔 매번 전체 출력을 확인한다.

| 저장 파일 | 확인할 값 |
|---|---|
| `supply.json` | 세 반복의 input copy, GEMV, output copy 시간과 CPU 명령 수. GEMV의 채널별 AbMac 컬럼 증분, 요청 수, 유효 GMAC/s, 평균 미완료 MAC 요청 수, 컨트롤러 PIM 큐 retry 카운터. |
| `mac-events.csv` (small) | AbMac 요청별 코어 최초 시도, 컨트롤러 accept, dispatch, 마지막 명령 통지, done의 gem5 tick과 코어 retry 횟수. 시간 차이는 해당 반복의 sim_freq로 나누어 초로 변환한다. |
| `sim/pim.trace` (small) | WrGb와 레지스터 작업을 포함한 원본 요청 순서. 기본 256-bit VLEN에서는 full-width AbMac 한 요청이 32바이트 컬럼 하나이며 CSV의 columns 필드에서 확인한다. |
| `sim/stats.txt`, `sim/*ramulator_stats.*.yaml`, `artifacts.sha256` | ROI별 원본 통계, 16개 컨트롤러 누적 카운터의 스냅샷, 빌드 산출물 식별값. |

small은 각 측정에서 채널당 128개, gate는 채널당 32,000개의 실제 AbMac 컬럼을 요구한다. GEMV 이전과 이후의 채널별 누적값 차이로 검증한다. small의 `mac_trace_by_trial`에서 요청이 하나도 accept 상태에 없던 시간 비율, 코어 시도 간격, 각 경로의 평균 지연을 비교한다. 공백에는 WrGb, reg 작업, 일반 메모리 접근과 의존성도 포함된다. 마지막 명령 통지는 sequencer가 보고하는 시점이며, 여러 컬럼을 가진 요청의 내부 발행 시점 전체를 나타내지는 않는다. gate는 파일 크기를 제한하기 위해 이벤트 trace를 끄고 같은 요청 경로의 집계 통계만 수집한다.

## 기존 결과로 확인할 수 있는 범위

Dense TP1·4K의 두 번째 stats dump(FFN)는 82.205 ms다. 16개 채널에 걸쳐 AbMac 1,536,000건, WrGb 1,536,000건이며 AbMac의 평균 accept→done은 45.75 ns다. AbMac 요청의 체류 시간 합계를 ROI 시간으로 나누면 디바이스 전체 평균 동시 체류량은 약 0.855건이다(ROI 경계를 가로지르는 요청은 오차 요인). 이는 16채널에 MAC 요청이 고르게 분포했다는 카운터와 양립한다. 평균 체류량에는 큐 대기도 포함된다. 45.75 ns는 요청 하나의 지연이고, 3.75 ns는 열린 행에서 연속된 MAC 컬럼의 최소 발행 간격이다. 두 값만으로 공급자를 판정할 수 없다. [기존 실험](qwen3-serving-16.ko.md) · [통계 생성 코드](../gem5-ParallelPIM-main/src/mem/ramulator2/pim_ramulator2.cc)

## 대상과 통제 조건

- 대표 연산은 Qwen3-32B gate projection 하나로 고정한다. W[25600, 5120] × x[5120], TP1, FP16 입력·가중치, FP32 누산, 16채널 LPDDR6, VLEN 256, 기존 3 GHz Timing CPU와 DRAM 375 ps를 사용한다. 유효 연산량은 131,072,000 MAC, 순수 MAC 하한은 120 µs다. 정상 배치의 AbMac 컬럼은 채널당 32,000개다.
- 상세 trace 개발에는 W[256, 2048]을 먼저 사용한다. 이는 16채널 전체에 걸쳐 한 출력 행 그룹과 여러 입력 타일을 포함한다. 주소와 데이터는 모든 실험군에서 동일하게 유지한다. 예를 들어 모든 가중치와 입력을 1로 초기화하면 모든 출력 원소가 각각 5120 또는 2048인지 확인할 수 있다.
- 할당·가중치 초기화·백엔드 초기화는 측정 전에 마친다. 동일한 예열 1회 후 같은 버퍼로 3회 반복한다. 이전 작업 완료, accumulator 초기화, 같은 행/캐시 시작 조건을 확인한다. 시간을 입력 복사, pim_gemv, 출력 복사로 나누고 마지막 읽기 응답과 검증까지 확인한다. posted write 응답은 완료 시점으로 쓰지 않는다.
- 산출물에 ELF·layout 해시, CPU/DRAM 클록, 타이밍·큐 설정, 형상, 명령 개수와 payload, 결과 검사, ROI 경계를 남긴다. 계측 전후 기준 실행의 simTicks와 명령 수가 일치하는지도 확인한다.

## 비교군과 실행 순서

| 순서 | 무엇을 바꾸는가 | 무엇을 판정하는가 |
|---|---|---|
| A. 기준 실행 | 기존 libpim과 CPU로 GEMV. 작은 형상은 상세 trace, 실제 gate는 채널별 요약 통계 수집 | 시간이 GEMV 호출 안에 있는지, 채널이 번갈아 동작하는지, 공급 대기가 어느 구간에 집중되는지 확인한다. |
| B. 코어 속도 민감도 | 같은 ELF로 CPU만 1.5 / 3 / 6 GHz. DRAM·컨트롤러·버스·캐시 클록은 기준값 고정 | CPU 속도에 따라 최초 요청 도착 간격과 GEMV 시간이 함께 줄어드는지 본다. 응답 대기에 묶인 코어는 주파수만 높여도 효과가 작을 수 있다. |
| C1. 기록 시각 재생 | A에서 메모리 포트로 들어온 요청을 동일한 주소·크기·순서·도착 조건으로 Timing 경로에 재생 | CPU를 제거한 비교기가 기준 실행의 컬럼 수, 결과, 채널별 분포, first accept→last done 시간을 재현하는지 검증한다. |
| C2. 조기 공급 | 같은 재생기에서 독립 요청 사이의 소프트웨어 공백만 제거한다. 채널별 순서, 데이터 의존성과 컨트롤러 backpressure는 보존한다. | 같은 명령열로 얻을 수 있는 최대 개선 폭을 측정한다. A/C1 대비 속도 향상이 크면 소프트웨어 및 전달 경로에 개선 여지가 있다. |
| D. 컨트롤러 민감도 | C2에서만 nCCD_AB를 10→20클록으로 변경하고 다른 타이밍은 고정한다. 열린 행의 독립 MAC 스트림으로 실제 발행 간격 변화도 확인한다. | 공급이 충분할 때 성능이 DRAM 컬럼 간격을 따르는지 확인한다. 열린 행 스트림의 피크와 WrGb·결과 회수를 포함한 GEMV 상한을 따로 보고한다. |

C1/C2는 동일한 timing 포트의 accept/retry 경로로 주입한다. GEMV가 의존하는 일반 메모리 접근, WrGb, accumulator reset, register read와 결과 저장도 보존한다. 읽기 결과로 생성되는 후속 쓰기는 실제 응답 이후에 발행한다. 재생기의 outstanding 한도는 기록하고 늘려 보아 결과가 수렴하는지 확인한다. 실제 gate에서 C1의 기준 시간 차이가 5%를 넘으면(실험 허용 기준) C2의 속도 향상을 원인 판정에 사용하기 전에 누락된 트래픽이나 초기 상태를 찾는다.

## 관측할 이벤트

| 위치 | 이벤트와 단위 | 필요한 이유 |
|---|---|---|
| CPU측 timing 포트와 메모리 입구 | 최초 send 시도, 거절, retry, accept, 응답 tick. 요청 ID와 채널·주소·연산 기록 | CPU가 늦게 요청한 경우와 이미 보낸 요청이 막힌 경우를 구분한다. retry는 새 요청으로 세지 않는다. |
| gem5 PIM 큐 | seq별 accept, dispatch, done. 채널·unit별 미완료 수, 선행 요청·hazard·busy로 인한 대기 | 큐에 들어갔지만 sequencer로 넘어가지 못한 시간을 분리한다. unit 단위 직렬화도 후보에 포함한다. |
| Ramulator sequencer | job ID, 채널, 실제 ACT/PRE/GB/MAC 컬럼 발행 clock, 완료 clock | 한 요청이 여러 컬럼일 수 있다. MAC 간격은 실제 컬럼 발행 시점으로 계산한다. |

기존 PIM debug trace에 accept/dispatch/done이 있고 sequencer에 컬럼 누적 카운터가 있다. A는 guest microbenchmark, 코어 최초 시도·retry, 마지막 명령 통지와 ROI 집계를 사용한다. B는 현재 SE 보드에 고정된 3 GHz와 코어 클록 도메인을 분리하는 설정 작업이 필요하다. 다중 컬럼 요청 내부의 개별 발행 시각 및 C의 timing 재생기는 후속 계측·구현 대상이다. 두 clock의 차이는 기록된 DRAM 주기와 기준 tick으로 환산한다. [SE 설정](../gem5-ParallelPIM-main/configs/example/gem5_library/riscv-pim.py) · [PIM 요청 처리](../gem5-ParallelPIM-main/src/mem/ramulator2/pim_ramulator2.cc) · [실제 컬럼 발행 훅](../gem5-ParallelPIM-main/ext/ramulator2/src/ramulator/pim/pim_sequencer.cpp)

## 결과를 읽는 기준

1. **공급 병목**은 코어측 최초 시도 자체가 드물고, 같은 시간에 컨트롤러가 다음 의존성 없는 요청을 받을 수 있었으며, C2에서 GEMV 시간이 줄어들 때 지지된다. B까지 빨라지면 코어 실행 비용이 원인이라는 근거가 강해진다. C2만 빨라지면 libpim의 채널 순회·요청 직렬화 등도 남는다.
2. **컨트롤러·의존성 병목**은 retry 또는 accept 이후의 대기가 길고 C2에서도 GEMV 시간이 비슷할 때 지지된다. dispatch 이전이면 unit busy·hazard를, 이후면 WrGb/ACT/PRE 경합·행 상태·컬럼 제약을 확인한다. D만으로 CPU 병목을 배제하지 않는다.
3. **채널 병렬성**은 각 채널의 동작 시간 합계와 모든 채널 동작 구간의 합집합을 함께 본다. 같은 MAC 개수를 처리한 16개 채널도 순차로 일할 수 있다. 채널별 MAC 간격 p50/p95, 열린 행에서 연속 가능한 MAC 간격, 시간대별 동시 활성 채널 수를 보고한다.
4. **최종 보고**에는 A/B/C1/C2/D별 GEMV 시간, 유효 GMAC/s, 최초 시도→accept·accept→dispatch·dispatch→first column·last column→done 분포, 큐 공백 및 retry 시간, MAC/GB 컬럼 수와 출력 일치 여부를 싣는다. 다른 채널에서 겹친 지연은 합산해 전체 시간으로 환산하지 않는다. 먼저 원인을 검증한 후 해당 구간의 비용만 기존 서빙 재생 모델에 반영한다.
