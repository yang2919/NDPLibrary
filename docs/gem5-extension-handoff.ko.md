# NDP–gem5 패킷 규격 v2

이 규격은 `gem5` 백엔드의 명령 패킷에 적용됩니다. 확장 gem5의 `gem5_pim` 백엔드는 [libpim 커널 인터페이스](gem5-pim-kernel-plan.md)를 사용합니다.

[English](gem5-extension-handoff.md)

NDP Library가 보내는 요청과 그에 대한 응답은 같은 패킷 구조를 사용합니다. 송신 측과 수신 측이 주고받을 데이터와 완료 조건을 아래에 정의합니다. 수신 측의 프로그램 구성과 명령 실행 방식은 구현팀이 선택합니다.

## 1. 패킷 구조

```text
|<------------ 32 bytes ------------>|<------- payload_bytes ------->|
| NIP1 header                        | request or response payload   |
```

각 표의 첫 바이트 번호는 0입니다. 예를 들어 헤더 표의 시작 바이트 4, 크기 2는 헤더의 4번과 5번 바이트를 뜻합니다. 여러 바이트로 된 정수는 낮은 자리부터 저장합니다(little-endian). 예를 들어 256은 2바이트로 `00 01`입니다. 필드 사이에 빈 바이트는 없습니다.

모든 요청·응답 헤더의 첫 4바이트는 고정 문자열 `NIP1`입니다. 수신 측은 이 값이 일치하는지 확인한 뒤 나머지 필드를 이 표에 따라 해석합니다.

|바이트|비트|필드|값 또는 의미|
|---|---|---|---|
|0–3|0–31|패킷 형식 식별자|NIP1 = 4e 49 50 31|
|4–5|32–47|version|1|
|6–7|48–63|kind|4 요청 · 5 성공 · 6 오류|
|8–15|64–127|epoch|연결 시작 시 양측이 공유하는 세션 번호(0 제외)|
|16–23|128–191|sequence|요청 번호. 첫 요청은 1, 이후 1씩 증가|
|24–27|192–223|payload_bytes|헤더 다음에 오는 바이트 수. 최대 8,388,608|
|28–31|224–255|payload_crc32|본문의 전송 오류를 확인하는 값|

현재 전송 방식은 Unix 스트림 소켓입니다. 한 번의 읽기로 패킷 전체가 도착한다는 보장이 없으므로, 헤더 32바이트를 모은 뒤 헤더에 적힌 길이만큼 본문을 읽습니다. 다음은 본문 검사 예시입니다.

```python
import zlib

received_crc = int.from_bytes(header[28:32], "little")
assert zlib.crc32(payload) == received_crc
```

**CRC 검사 예시.** Submit 성공 응답의 본문은 한 바이트 `04`입니다. 응답을 보내는 쪽은 이 한 바이트의 CRC32를 계산하고 결과 `0xD56F2B94`를 헤더의 28–31번 바이트에 `94 2b 6f d5` 순서로 씁니다(little-endian). 응답을 받은 쪽은 받은 본문으로 CRC32를 다시 계산합니다. 헤더의 값과 다르면 응답을 폐기합니다.

**응답을 잃었을 때.** 요청자는 한 번에 요청 하나를 보내고 그 응답을 기다립니다. 예를 들어 장치가 MAC 누산을 끝낸 직후 연결이 끊기면, 요청자는 완료 응답을 받지 못해도 장치의 누산값은 이미 바뀌었을 수 있습니다. 이때 같은 요청을 다시 보내면 같은 곱셈 결과가 두 번 더해질 수 있습니다. 헤더나 CRC 오류, 연결 종료, 시간 초과로 유효한 응답을 받지 못하면 실행 여부를 미확정으로 처리하고 요청을 자동 재전송하지 않습니다.

## 2. 요청과 응답

본문의 첫 바이트가 요청 번호(op)입니다. 응답은 요청과 같은 epoch와 sequence를 사용합니다. 아래 성공 응답은 헤더를 제외한 본문이며, 바이트 값은 16진수로 표시했습니다.

|op|요청|성공 응답|응답 시점|
|---|---|---|---|
|1|세션 설정 확인|요청 본문 63바이트 그대로|첫 요청에서 규격·장치 구성이 모두 일치할 때|
|2|Upload|02|쓰기 완료 후 다음 명령이 값을 읽을 수 있을 때|
|3|Download|03 + 읽은 데이터|요청한 바이트를 모두 읽었을 때|
|4|Submit|04|배치의 모든 명령과 쓰기가 완료됐을 때|
|5|Close|05|장치의 해당 세션 접근이 모두 끝났을 때|

**데이터 이동 경로**

|동작|원본 → 목적지|패킷으로 전달하는 내용|
|---|---|---|
|Upload 요청|라이브러리 → 장치 메모리|요청에 목적지 주소, 길이, **실제 데이터**를 담습니다.|
|Download 요청|장치 메모리 → 라이브러리|요청에 원본 주소와 길이를 담고, **응답으로 실제 데이터**를 받습니다.|
|Submit 안의 Transfer 명령|장치 메모리 → 장치 메모리|명령에 원본 주소, 목적지 주소, 길이를 담습니다. 데이터 복사는 장치 안에서 수행하고 배치 완료 시 Submit 성공 응답을 보냅니다.|

예를 들어 입력 벡터를 장치에 처음 기록할 때는 Upload를 사용합니다. 이미 Shared에 있는 벡터 타일을 GPR로 옮겨 연산에 사용할 때는 Transfer를 사용합니다. 최종 결과를 라이브러리에서 읽을 때는 Download를 사용합니다.

### 요청 1 · 세션 설정 확인

기존 코드에서 이 요청의 이름은 `Hello`입니다. 패킷에는 이름 대신 정수 1을 넣습니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|1|op|1|
|1|2|codec_version|2|
|3|2|numeric_profile|1 = FP16 저장|
|5|2|mapping_id|1 = 아래 Shared 주소 규칙|
|7|8|channels|채널 수|
|15|8|banks|채널당 뱅크 수|
|23|8|rows|뱅크당 행 수|
|31|8|row_bytes|행 하나의 크기|
|39|8|gb_bytes|채널별 Global Buffer 크기|
|47|8|gpr_bytes|GPR 공간 크기|
|55|8|scratchpad_bytes|Scratchpad 크기|

### 요청 2–5 · 데이터와 명령 전달

|요청|시작 바이트|크기(바이트)|필드 및 값|
|---|---|---|---|
|Upload / Download|0|1|op = 2 / 3|
|Upload / Download|1|9|목적지 / 원본 주소(아래 Address 표)|
|Upload / Download|10|8|N = 쓰거나 읽을 바이트 수|
|Upload|18|N|기록할 데이터. Download 요청은 바이트 17에서 끝남|
|Submit|0|1|op = 4|
|Submit|1|가변|NDP1 명령 배치(3절)|
|Close|0|1|op = 5|

Upload 데이터는 최대 8,388,590바이트, Download 결과 데이터와 Submit의 NDP1 배치는 각각 최대 8,388,607바이트입니다.

### 오류 응답 · kind = 6

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|1|op|실패한 요청 번호. 요청 번호가 없으면 0|
|1|2|error_id|1 InvalidArgument · 2 UnsupportedOperation · 3 OutOfMemory · 4 BackendFailure|
|3|0–256|message|선택적 오류 설명. 길이는 payload_bytes − 3이고 종료 문자는 없음|

오류 응답은 해당 요청의 최종 결과입니다. 실행 도중 오류가 났다면 이미 쓴 데이터가 남을 수 있습니다. 버퍼를 재사용하기 전에 장치 접근이 끝나야 합니다.

## 3. 명령 배치 구조

```text
Submit payload
[04] [NDP1 header: 16 bytes] [command header: 8 bytes] [command body] ...
```

**NDP1 헤더** · 아래 표의 바이트 0은 문자 N입니다.

Submit 본문의 요청 번호 바로 뒤에는 고정 문자열 `NDP1`이 옵니다. 이 값은 이어지는 데이터가 아래 형식의 명령 배치임을 표시합니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|4|명령 배치 형식 식별자|NDP1 = 4e 44 50 31|
|4|2|version|2|
|6|2|numeric_profile|1|
|8|4|total_bytes|16바이트 헤더를 포함한 배치 전체 길이|
|12|4|command_count|0–65,536|

**명령 헤더** · 첫 번째 헤더는 NDP1의 바이트 16에서 시작합니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|2|opcode|4절의 명령 번호|
|2|2|reserved|0|
|4|4|body_bytes|이 헤더 바로 뒤에 오는 명령 본문의 길이|

배치 전체의 길이, 개수, ID, 예약값, 필드 범위를 검증한 뒤 기록된 순서로 실행합니다. 명령 사이에 패딩은 없으며 마지막 명령 뒤에 남는 바이트도 허용하지 않습니다. 상태와 데이터는 다음 Submit에서도 유지합니다.

**공통 필드 형식**

|형식|시작 바이트|크기(바이트)|의미|
|---|---|---|---|
|Address|0|1|1 Shared · 2 GPR · 3 Scratchpad · 4 Global Buffer (Transfer destination)|
|Address|1|8|선택한 공간의 시작점부터 센 바이트 수|
|Operand|0|9|Address|
|Operand|9|8|FP16 원소 개수|

```text
Shared byte offset = (((row * channels + channel) * banks + bank) * row_bytes) + byte_in_row
mask bit i = (mask >> i) & 1
```

현재 Shared 주소 매핑(`mapping_id = 1`)은 메모리를 `[row][channel][bank][byte_in_row]` 순서로 배치한다고 가정합니다. 한 뱅크의 한 행 안에서 바이트가 연속하고, 그다음으로 뱅크, 채널, 행 순으로 인덱스가 바뀝니다. gem5 연동 시 대상 메모리 컨트롤러의 주소 매핑에 맞춰 이 규칙을 변경할 수 있습니다.

마스크에서 1인 비트가 해당 채널 또는 뱅크를 선택합니다. 예를 들어 채널 마스크 `0x05`는 채널 0과 2를 선택합니다. 텐서 데이터는 little-endian FP16입니다.

## 4. 명령 목록과 본문

각 표의 시작 바이트는 해당 명령 본문의 첫 바이트를 기준으로 합니다.

### 1 · Transfer

장치 내부에서 바이트를 복사합니다. 지원 경로는 Shared ↔ GPR, Shared ↔ Scratchpad 및 Shared/GPR → 단일 채널 Global Buffer입니다.

Global Buffer 목적지의 space ID는 4이고 offset은 `채널 번호 × gb_bytes + 채널 내 바이트 위치`입니다. 원본·목적지 위치와 전송량은 32바이트 배수이며 전송량은 양수입니다. 한 전송은 한 채널 안에서 끝나야 합니다. 예를 들어 gb_bytes=64일 때 채널 1의 시작 주소는 {4, 64}입니다. GEMV는 준비한 GPR 타일을 채널별로 한 번씩 전송합니다. 명령 코덱 v2에서 opcode 3은 사용하지 않습니다. 외부 NIP1 헤더 버전은 1입니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|9|src|원본 Address|
|9|9|dst|목적지 Address|
|18|8|bytes|처리할 바이트 수|

### 2 · InitializeMac

GPR 값으로 선택한 누산기 초기화

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|8|channel_mask|선택할 채널|
|8|8|bank_mask|선택할 뱅크|
|16|8|gpr_offset|GPR 내 바이트 위치|

### 4 · MacAllBanks

선택한 채널의 뱅크 데이터와 Global Buffer 데이터를 곱해 누산

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|8|channel_mask|선택할 채널|
|8|8|row|뱅크의 행 번호|
|16|8|first_burst_column|시작 열(32바이트 burst 단위)|
|24|8|burst_count|처리할 32바이트 burst 개수|

### 5 · ReadMac

누산 결과를 FP16으로 GPR에 기록

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|8|channel_mask|선택할 채널|
|8|8|gpr_offset|GPR 내 바이트 위치|

### 6 · LaunchPnmProgram

미리 적재한 RISC-V 프로그램을 program_id로 지정하고 인수와 스택을 전달합니다. v2 소켓 규격에는 프로그램 적재 요청이 없습니다. 기준 구현은 알 수 없는 ID를 InvalidArgument로 거부하며, 유효한 실행 요청도 바이너리 실행기가 없어 UnsupportedOperation으로 반환합니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|8|program_id|프로그램 식별자|
|8|9|arguments|인수 블록 Address|
|17|8|argument_bytes|인수 블록 크기|
|25|9|stack|스택 Address|
|34|8|stack_bytes|스택 크기|
|42|8|instruction_limit|최대 실행 명령 수|

### 7 · PnmCompute

kernel_id로 미리 정의된 연산을 선택합니다. 예를 들어 Matmul은 입력·출력 주소와 행렬 차원을 받아 에뮬레이터에서 직접 계산합니다. 현재 decoder의 PNM 연산은 이 방식을 사용하며 프로그램 바이너리가 필요하지 않습니다.

```text
I = input_count
J = index_count
B = 6 + 17 × I
body_bytes = 60 + 17 × I + 8 × J
```

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|2|kernel_id|1 Matmul · 2 Add · 3 Mul · 4 RMSNorm · 5 Softmax · 6 SiLU · 7 Copy · 8 Gather · 9 Scale · 10 Transpose|
|2|4|input_count|입력 Operand 개수|
|6|17 × I|inputs|순서대로 나열한 입력 Operand|
|B|17|output|출력 Operand|
|B + 17|8|rows|행 수|
|B + 25|8|width|출력 너비|
|B + 33|8|inner|행렬곱의 내부 차원 K|
|B + 41|1|transpose_b|0 = 그대로, 1 = 두 번째 행렬 전치|
|B + 42|4|factor|배율(IEEE 754 binary32)|
|B + 46|4|epsilon|정규화 상수(IEEE 754 binary32)|
|B + 50|4|index_count|인덱스 개수|
|B + 54|8 × J|indices|인덱스 배열. 원소당 8바이트|

### 8 · PnmRelayout

행렬의 물리 배치 변환

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|9|src|원본 Address|
|9|9|dst|목적지 Address|
|18|8|rows|행 수|
|26|8|columns|열 수|
|34|8|banks|배치에 참여하는 총 뱅크 수|
|42|8|tile_elements|타일당 FP16 원소 수|
|50|8|row_bytes|물리 행 크기|
|58|1|pack|1 = dense → banked, 0 = banked → dense|

### 9 · CopyBytes

원본과 목적지 범위가 유효하면 Shared, GPR, Scratchpad 안에서 또는 이들 사이에서 복사합니다. 같은 공간에서 범위가 겹쳐도 원본 바이트가 보존되는 memmove 방식입니다. 현재 L1은 Shared 버퍼의 구간 복사에 사용합니다. Transfer(1)는 Shared ↔ GPR 또는 Shared ↔ Scratchpad 및 Shared/GPR → 단일 채널 Global Buffer 경로를 허용하며 GEMV 명령 생성기가 데이터를 옮길 때 사용합니다. 두 명령 모두 본문은 원본 주소, 목적지 주소, 바이트 수로 구성됩니다.

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|9|src|원본 Address|
|9|9|dst|목적지 Address|
|18|8|bytes|처리할 바이트 수|

### 10 · FillBytes

지정한 범위를 한 바이트 값으로 채우기

|시작 바이트|크기(바이트)|필드|의미|
|---|---|---|---|
|0|9|dst|목적지 Address|
|9|8|bytes|처리할 바이트 수|
|17|1|value|반복해서 쓸 한 바이트 값|

## 5. 전체 패킷 예시

세션 설정 확인이 끝난 뒤, Shared의 바이트 256과 257을 `5a`로 채우고 다시 읽는 예입니다. 마지막 응답의 본문 `03 5a 5a`에서 첫 바이트는 Download 요청 번호이고 뒤의 두 바이트가 읽은 값입니다.

① Submit 요청 · FillBytes 한 개

```text
4e 49 50 31 01 00 04 00 01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 2b 00 00 00 88 25 8d 90
04 4e 44 50 31 02 00 01 00 2a 00 00 00 01 00 00 00 0a 00 00 00 12 00 00 00 01 00 01 00 00 00 00 00 00 02 00 00 00 00 00 00 00 5a
```

② Submit 성공 응답

```text
4e 49 50 31 01 00 05 00 01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 01 00 00 00 94 2b 6f d5
04
```

③ Download 요청 · 2바이트

```text
4e 49 50 31 01 00 04 00 01 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 12 00 00 00 f9 05 e5 f1
03 01 00 01 00 00 00 00 00 00 02 00 00 00 00 00 00 00
```

④ Download 성공 응답

```text
4e 49 50 31 01 00 05 00 01 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 03 00 00 00 7f 6a ed 36
03 5a 5a
```

각 예시의 첫 줄은 헤더, 둘째 줄은 본문입니다.

[NIP1](../src/l1/gem5/metadata.cpp) · [Request / response](../src/l1/gem5/execution.cpp) · [NDP1](../src/l1/ndp/codec.cpp)
