# K4 사용자 검증 — 외부 PyTorch에서 stock gem5 게스트의 GEMV 실행

이 절차는 Mac에서 실행한 Python 프로세스가 게스트에 weight를 한 번 준비하고, 서로 다른 두 입력으로 FP16 GEMV를 실행한 뒤 결과를 회수하는지 확인합니다. 게스트는 기존 x86 Ubuntu 이미지에서 공급사의 libpim reference model로 빌드한 worker 한 개를 세션 종료까지 유지합니다.

| 구간 | 전달 방식 |
|---|---|
| 외부 PyTorch → gem5 실행 프로세스 | 호스트의 loopback TCP 포트로 연결합니다. NDP Session이 기존 v2 worker 요청을 전송합니다. |
| gem5 실행 프로세스 ↔ 게스트 Linux | 게스트가 `m5 exit`로 대기하면 gem5의 Python 설정 코드가 요청을 readfile에 기록하고 시뮬레이션을 재개합니다. 게스트는 `m5 readfile`로 요청을 가져오고 `m5 writefile`로 응답을 내보냅니다. |
| 게스트 Linux ↔ libpim worker | 게스트 중계 프로그램이 동일한 바이너리 프레임을 worker의 표준 입력과 출력으로 전달합니다. worker가 tensor handle, weight 배치, GEMV 실행과 결과를 관리합니다. |

## 1. 호스트 바인딩 준비

Docker Desktop을 시작합니다. 두 터미널 모두 저장소 루트에서 실행합니다. 호스트에서는 PyTorch를 설치한 Python을 사용합니다.

```sh
python3 -c 'import torch'
cmake -S . -B build-python -DNDP_BUILD_PYTHON=ON -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-python --target ndp_python -j4
```

## 2. 터미널 A에서 FS 시작

```sh
STAGE=K4 BOOT_TIMEOUT=7200 ./tools/gem5/fs/run.sh
```

기존 F0의 gem5 이미지와 커널·디스크 캐시를 재사용합니다. 게스트 부팅과 worker 준비가 끝나면 터미널 A에 다음 줄이 표시됩니다.

```text
K4_HOST_READY port=45240
```

이때 터미널 B를 실행합니다. 실행 한 번당 클라이언트 연결 한 개를 받으므로 포트 확인용 접속도 이 연결을 소비합니다.

## 3. 터미널 B에서 두 결과 검증

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_remote.py
```

호스트는 bias 없는 2×3 weight를 재사용합니다. 각 결과를 CPU PyTorch 결과와 원소별로 비교하며, 세션 종료 응답까지 받으면 다음과 같이 출력합니다.

```text
GEMV [[6.0, 1.0]]
GEMV [[-1.0, -4.0]]
K4_CLIENT_PASS
```

터미널 A에서도 다음 줄을 확인하면 통과입니다.

```text
K4 PASS: stock x86 guest reference worker session closed. Verify K4_CLIENT_PASS on host.
```

## 결과 확인 및 재실행

```sh
run_dir="build-gem5-fs/$(basename "$(cat build-gem5-fs/latest-run.txt)")"
cat "$run_dir/result.txt"
grep 'K4_HOST_' "$run_dir/console.log"
grep -E 'K4_GUEST_WORKER_CLOSED|NDP_FS_SCRIPT_COMPLETE' "$run_dir/board.pc.com_1.device"
```

포트가 사용 중이면 터미널 A에서 `K4_PORT=45241`을 추가하고 터미널 B의 명령 끝에 `--port 45241`을 추가합니다. 준비 로그는 컨테이너 내부 포트인 45240을 표시합니다. 연결이 중단되면 해당 실행을 종료하고 두 터미널의 절차를 새 세션으로 반복합니다.

별도 클라이언트에서 연결할 때는 blocking TCP socket의 `fileno()`를 `ndp.Session(backend="gem5_pim", worker_fd=...)`에 전달합니다. Session이 descriptor를 복제하므로 생성 직후 호출자의 socket을 닫고 해당 스트림은 Session 전용으로 사용합니다. 완전한 예시는 [검증 클라이언트](../tools/gem5/pim_kernel/verify_remote.py)에 있습니다.

이 절차가 확인하는 범위는 외부 PyTorch부터 게스트 reference worker까지의 명령 전달, FP16 결과, 세션 수명입니다. RISC-V PIM 커널, 실제 장치 주소 매핑 및 PIM 지연시간 검증은 확장 gem5 단계에서 수행합니다.
