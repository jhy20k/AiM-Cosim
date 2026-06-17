# Verilog RTL Modules

이 디렉터리는 GDDR6-AiM functional path를 Verilator로 실행하기 위한
SystemVerilog RTL을 담고 있다. 현재 기본 빌드는 `ACC_MAG_WIDTH=32`이며
`pim_mac_tree_mag32` top과 `accumulator_param #(.MAG_WIDTH(32), .Q_POINT(24))`
를 사용한다.

## 빌드 시 선택되는 RTL top

상위 CMake 옵션 `ACC_MAG_WIDTH`가 RTL top과 accumulator 구현을 선택한다.

| `ACC_MAG_WIDTH` | RTL top | Accumulator | 내부 누산 포맷 |
|---:|---|---|---|
| `32` | `pim_mac_tree_mag32` | `accumulator_param #(.MAG_WIDTH(32), .Q_POINT(24))` | sign + 32-bit magnitude + 10-bit exponent |
| `37` | `pim_mac_tree_mag37` | `accumulator_param #(.MAG_WIDTH(37), .Q_POINT(30))` | sign + 37-bit magnitude + 10-bit exponent |
| 그 외/legacy | `pim_mac_tree` | `accumulator.v` | sign + 50-bit Q30 magnitude + 10-bit exponent |

`MAG32/Q24`는 IEEE FP32 accumulator가 아니다. BF16 결과로 내보내기 전의
magnitude를 32-bit fixed-point 형태로 보관하고, 별도 exponent와 함께 정규화/누적한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `pim_mac_tree.v` | legacy top. scalar MAC, all-bank vector16 MAC, EWMUL, activation path 포함 |
| `pim_mac_tree_mag32.v` | 기본 top. `pim_mac_tree.v`와 같은 구조이며 accumulator만 `MAG32/Q24`로 교체 |
| `pim_mac_tree_mag37.v` | `MAG37/Q30` accumulator variant |
| `adder_tree.v` | 16-lane BF16 product magnitude를 공통 exponent로 정렬한 뒤 reduction |
| `accumulator.v` | legacy 50-bit Q30 accumulator |
| `accumulator_param.v` | parameterized accumulator. `MAG_WIDTH`, `Q_POINT`로 내부 magnitude 폭과 binary point 위치 선택 |
| `activation_unit.v` | BF16 ReLU/LeakyReLU. 내부 Q8.16 변환 후 BF16 RNE 출력 |
| `bf16_to_q816.v` | BF16 to Q8.16 변환 |
| `q816_to_bf16_rne.v` | Q8.16 to BF16 RNE 변환 |
| `pu.v` | BF16 multiplier reference module. 현재 top은 inline multiplier logic 사용 |
| `without_multilatch_accumulator.v` | legacy accumulator wrapper. latch selection을 고정한 단순 wrapper |

## Top-level 동작 모드

`pim_mac_tree*` top은 `i_start`와 mode control 신호로 실행 path를 나눈다.

| Mode | 조건 | 출력 |
|---|---|---|
| scalar MAC | `i_af_en=0`, `i_ewmul_en=0`, `i_all_bank_vector_en=0` | `o_result` |
| all-bank vector16 MAC | `i_af_en=0`, `i_ewmul_en=0`, `i_all_bank_vector_en=1` | `o_mac_result_flat`, `o_result_flat` |
| scalar activation | `i_af_en=1`, `i_all_bank_vector_en=0` | `o_result`, `o_af_result_flat[15:0]` |
| all-bank vector16 activation | `i_af_en=1`, `i_all_bank_vector_en=1` | `o_af_result_flat`, `o_result_flat` |
| element-wise multiply | `i_ewmul_en=1` | `o_result_flat` |

주요 입력은 scalar path용 `i_wgt_flat`/`i_vec_flat` 각 256-bit
(16 x BF16), all-bank path용 `i_wgt_bank_flat` 4096-bit
(16 banks x 16 BF16), 그리고 bank별 bias용 `i_bias_bank_flat` 256-bit이다.

## MAC datapath

1. BF16 입력을 sign, exponent, mantissa로 unpack한다.
2. Top 내부 multiplier logic이 16개 lane의 BF16 product sign/exponent/magnitude를 만든다.
3. `adder_tree`가 최대 exponent를 기준으로 product magnitude를 정렬한다.
   정렬 전 magnitude는 `{8'b0, i_man[i], 8'b0}` 형태로 32-bit에 배치된다.
4. `adder_tree`는 signed reduction 후 sign, 32-bit magnitude, common exponent를 출력한다.
5. accumulator가 L/R lane 상태와 optional latch checkpoint를 사용해 누적하고,
   마지막에 BF16 RNE로 반올림한다.

## Accumulator 포맷

`accumulator_param`의 내부 상태는 fixed-point magnitude와 exponent를 따로 갖는 구조다.
따라서 “FP32로 누적”이라기보다 “확장 fixed-point magnitude + exponent”에 가깝다.

`MAG32/Q24`의 32-bit magnitude는 다음처럼 이해하면 된다.

```text
[31:25] headroom | [24] hidden/reference bit | [23:0] fractional field
```

여기서 `Q_POINT=24`는 정규화 목표 bit 위치다. 정상 정규화된 non-zero
magnitude는 bit 24가 leading 1 위치가 되도록 맞춰지고, bit 23 아래가 fraction
precision으로 쓰인다. BF16 RNE 출력 시에는 다음 bit를 사용한다.

```text
in_mag[23:17] : BF16 fraction 후보 7 bits
in_mag[16]    : guard bit
in_mag[15:0]  : sticky field
```

즉 `MAG32/Q24`는 “32-bit fixed-point magnitude 안에서 binary point/hidden-bit
기준 위치가 24”인 accumulator variant다. 전체 accumulator 상태에는 이 magnitude
외에도 sign과 10-bit exponent가 함께 저장된다.

legacy `accumulator.v`는 50-bit Q30 magnitude를 사용한다. 이 경우 BF16 RNE는
Q30 기준으로 fraction 후보 `in_mag[29:23]`, guard `in_mag[22]`, sticky
`in_mag[21:0]`를 사용한다.

## Activation datapath

`activation_unit`은 BF16 입력을 Q8.16 two's-complement 값으로 변환한 뒤 ReLU 또는
LeakyReLU를 수행하고, `q816_to_bf16_rne`로 BF16 결과를 만든다. 내부 fixed latency는
4 cycle이며, slope 입력이 0이면 기본 LeakyReLU slope로 약 0.01을 사용한다.

## 타이밍 메모

- `adder_tree`는 start 이후 stage 1 register와 stage 2 output register를 거친다.
- Top은 `i_acc_en`, bias, latch control을 accumulator 입력 타이밍에 맞추기 위해
  delay register로 전달한다.
- `activation_unit`은 자체 fixed latency 4 cycle을 갖는다.
- EWMUL path는 top 내부 BF16 RNE multiplier array 결과를 register한 뒤 출력한다.
- 정확한 command-level latency는 C++ co-simulation wrapper와 trace mode 설정까지
  포함해 확인해야 한다.

## 신호 명명 규칙

| 접두사 | 의미 |
|---|---|
| `i_` | 입력 포트 |
| `o_` | 출력 포트 |
| `c_` | 조합 논리 |
| `s_` | 순차 논리/register |
| `w_` | weight 관련 |
| `v_` | vector 관련 |
| `abk_` | all-bank vector path 관련 |
| `dbg_` | debug/verification 포트 |
