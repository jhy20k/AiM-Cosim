# AiM-CoSim: GDDR6-AiM Timing-Functional Co-Simulation 프레임워크

2026 대한전자공학회 하계학술대회

`AiM-Cosim`은 SK하이닉스 GDDR6-AiM timing-functional co-simulation을 위한 시뮬레이션 프레임워크 제안

## Key Contribution
- Timing simulator의 PU 모델링 한계를 보완하기 위해 Verilator 기반 RTL Co-Simulation Engine을 Ramulator2.0 기반 AiM Simulator와 통합
- SK하이닉스 GDDR6-AiM dataflow와 PU를 반영하여 RTL 모델 설계
- PU 동작 지연시간을 AiM Simulator의 DRAM timing model에서 정의된 지연시간과 별도로 분석하며 RTL PU latency가 AiM simulator에서 추상화된 PU 동작 지연시간보다 크다면 RTL PU latency로 동기화

## 저장소 구조

```text
AiM-Cosim/
├── CMakeLists.txt
├── Makefile
├── pyproject.toml
├── README.md
├── LICENSE
├── docker/
├── extern/                 # submodule / 외부 의존성
├── src/
│   ├── core/               # C++ simulator / RTL co-simulation core
│   ├── rtl/                # Verilator 대상 RTL
│   ├── configs/            # Ramulator2 / AiM 실행 설정
│   └── tools/              # 독립 실행형 C++ helper
├── scripts/
└── tests/
    └── result/
```

## 빌드 옵션

기본 빌드는 `MAG32` accumulator RTL을 사용한다. accumulator 폭은 CMake 옵션
`ACC_MAG_WIDTH`로 선택한다.

```bash
cmake -S . -B build_mag37 -GNinja -DACC_MAG_WIDTH=37
cmake --build build_mag37 -j$(nproc)
```

기본값은 `32`이며, CMake 설정은 `32`, `37`, 그 외 값에 대해 각각 대응 RTL top을 선택한다.

## 현재 검증 범위

현재 기본 build/test flow는 GDDR6-AiM RTL co-simulation을 기준으로 한다. `src/rtl`의
기본 RTL top은 `pim_mac_tree_mag32`이며, accumulator는
`accumulator_param #(.MAG_WIDTH(32), .Q_POINT(24))`를 사용한다. 자세한 RTL 구성과
accumulator 포맷은 `src/rtl/README.md`를 참고한다.

`src/core/*hbmpim*`, `src/core/hbmpim`, `src/configs/hbmpim_*.yaml`,
`src/tools/hbmpim_*` 경로의 HBM-PIM 관련 코드는 future work scaffold이다. 현재
검증된 기본 경로에는 포함하지 않으며, GDDR6-AiM 결과와 혼동하지 않는다.

## 처음 환경 세팅부터 실행까지

### Dependencies

기본 개발 환경은 Docker를 기준으로 한다. Host 쪽 도구는 다음만 사용한다.

- `git`
- Docker Engine
- Docker Compose plugin

컨테이너 이미지는 Verilator `v5.024`, GCC 12, CMake, Ninja, Python 3.10, test/plot/LLM
실험용 Python package를 포함한다. Llama3.2 WikiText처럼 CUDA를 쓰는 실험을 컨테이너에서
실행할 때는 NVIDIA Container Toolkit 기반 GPU passthrough 구성을 사용한다.

### Clone

저장소와 submodule을 함께 받는다.

```bash
git clone --recursive <repo-url> AiM-Cosim
cd AiM-Cosim
```

### Docker 환경 실행

프로젝트 루트에서 Docker image를 빌드하고 컨테이너를 시작한다.

```bash
make docker-build
make docker-run
make docker-shell
```

컨테이너 안의 작업 디렉토리는 `/workspace`이며, host의 저장소가 그대로 mount된다.

### Build

컨테이너 안에서 기본 빌드를 실행한다. 기본값은 `MAG32` accumulator RTL이다.

```bash
cd /workspace
make build
```

빌드가 끝나면 simulator binary는 다음 위치에 생성된다.

```text
build_mag32/extern/aim_simulator/ramulator2
```

### Test

컨테이너 안에서 기본 unit/regression test를 실행한다.

```bash
make test
```

### Run

Mode 1은 timing-only 실행이다.

```bash
make run-timing TRACE=tests/data/tc1_single_mac16.trace
```

Mode 2는 timing simulation과 Verilator RTL functional path를 함께 실행한다.

```bash
make run-rtl TRACE=tests/data/tc1_single_mac16.trace
```

직접 실행 형식은 다음과 같다.

```bash
./build_mag32/extern/aim_simulator/ramulator2 \
  -f <config.yaml> \
  -t tests/data/tc1_single_mac16.trace
```

기본 Mode 2 설정(`src/configs/aim_rtl.yaml`)은 RTL functional path를 실행하지만,
RTL latency를 이용해 producer/consumer command issue를 지연시키는 scoreboard gating은
켜지지 않는다. scoreboard gating을 확인하려면 `verilator.scoreboard_gating: true`가
설정된 debug config를 사용한다.

```bash
./build_mag32/extern/aim_simulator/ramulator2 \
  -f src/configs/aim_rtl_stage4_debug_mac.yaml \
  -t tests/data/tc1_single_mac16.trace
```

NOTE: 현재 AiM simulator 설정은 32 channels를 기준으로 한다.

## 추가 실험

Mode 1과 Mode 2 cycle 분석은 matmul regression으로 확인한다.

```bash
bash scripts/run_matmul_regression.sh
```

LLM projection별 Mode 1/Mode 2 cycle, Mode 2 RTL 결과 오차, 시뮬레이션 속도 비교는
HF projection runner와 요약 스크립트로 생성한다. Mode 1은 timing-only이므로 projection 오차는
Mode 2 RTL 결과와 CPU/GPU BF16 reference 비교 기준이다.

```bash
python3 scripts/run_summer26_hf_stage5_7.py \
  --models Llama3.2-1B-Instruct \
  --projection-groups self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.gate_proj mlp.up_proj mlp.down_proj

python3 scripts/consolidate_summer26_results.py
python3 scripts/generate_summer26_llm_figures.py --force
```

`generate_summer26_llm_figures.py`는 기본적으로 여러 모델의 coverage를 확인한다. `Llama3.2-1B`
결과만 있는 경우 마지막 coverage check에서 non-zero로 종료될 수 있지만, GPU functional
comparison summary는 먼저 생성된다.

주요 요약 파일은 다음과 같다.

- `tests/result/result_summer26/summary/hf_weight_mode_summary.csv`: projection별 Mode 1/Mode 2 cycle.
- `tests/result/result_summer26/summary/hf_weight_projection_aggregate_summary.csv`: projection별 오차 요약.
- `tests/result/result_summer26/summary/hf_weight_actual_pre_activation_projection_aggregate.csv`: 실제 decoder projection 입력 기반 pre-activation 오차 요약.
- `tests/result/result_summer26/summary/pre_activation_projection_comparison_summary.md`: 기존 random BF16 aggregate와 실제 pre-activation 결과 비교.
- `tests/result/result_summer26/summary/hf_weight_execution_time_comparison.csv`: projection별 Mode 1/Mode 2 wall-time 비교.
- `tests/result/result_summer26/figures/supplemental_torch_functional_aggregate.csv`: torch CUDA BF16 functional 비교 요약.
- `tests/result/result_summer26/figures/supplemental_cupy_cuda_aggregate.csv`: CuPy CUDA timing 보조 요약.

## 검증 결과

현재 `tests/result/result_summer26`에는 `Llama3.2-1B-Instruct` 기준 projection 검증 결과가
포함되어 있다.

| 검증 항목 | 결과 | 요약 파일 |
|---|---:|---|
| Mode 1 / Mode 2 cycle | 24/24 PASS | `summary/hf_weight_execution_time_comparison.csv` |
| CPU/GPU reference timing | CPU 24/24, GPU 24/24 OK | `summary/hf_weight_all_projection_cpu_gpu_summary.csv` |
| Mode 2 vs CPU BF16 functional | 7개 projection aggregate 생성 | `summary/hf_weight_projection_aggregate_summary.csv` |
| Mode 2 vs torch CUDA BF16 functional | 7/7 OK | `figures/supplemental_torch_functional_aggregate.csv` |
| 실제 decoder pre-activation projection | 7개 projection aggregate 생성 | `summary/hf_weight_actual_pre_activation_projection_aggregate.csv` |
| Mode 1 / Mode 2 simulation wall-time | 24개 projection chunk 비교 | `summary/hf_weight_execution_time_comparison.csv` |

비교 조건은 다음과 같이 구분한다.

- Activation source 3종 비교: `Llama3.2-1B-Instruct` layer 0에서 실제 HF safetensors weight와 `row_sharded` layout을 공통으로 사용하며, `pre_activation/RDMAC16` 결과를 Activation Function 없이 비교한다.
- 3종 입력 activation:
  - `random_bf16`: seed 42만으로 만든 랜덤 BF16 입력.
  - `prompt_hash_bf16`: prompt 파일과 seed 42로 만든 deterministic 랜덤 BF16 입력.
  - `actual hook`: 실제 Llama forward 중 projection module에 들어간 BF16 입력.
- CPU 기준: BF16 weight와 BF16 입력을 NumPy FP32로 변환해 GEMV를 수행한 뒤 BF16으로 반올림한 reference
- GPU 기준: 동일한 BF16 weight와 BF16 입력을 torch CUDA BF16 projection으로 실행한 reference이다. 본 실험에서는 수치 비교의 안정성을 우선하기 위해 `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`로 고정한다. 이 설정은 BF16 matmul의 reduction 과정에서 낮은 정밀도의 축약 누산 경로를 허용하지 않아 중간 누산 정밀도를 더 보수적으로 유지한다. 반대로 `True`로 설정하면 Tensor Core/cuBLAS 계열 backend가 shape에 따라 reduced-precision reduction을 사용할 수 있어 속도는 좋아질 수 있지만, GPU reference와 AiM RTL 사이의 delta가 backend 최적화 경로의 영향을 더 크게 받을 수 있다.
- 표기 형식: 각 cell은 `within_1_ulp, max ULP`를 의미한다.

| Projection | Random PIM-vs-CPU | Random PIM-vs-GPU | Prompt-hash PIM-vs-CPU | Prompt-hash PIM-vs-GPU | Actual hook PIM-vs-CPU | Actual hook PIM-vs-GPU |
|---|---:|---:|---:|---:|---:|---:|
| `self_attn.q_proj` | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 |
| `self_attn.k_proj` | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| `self_attn.v_proj` | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| `self_attn.o_proj` | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 1 |
| `mlp.gate_proj` | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8191/8192, max 8 |
| `mlp.up_proj` | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 0 | 8192/8192, max 1 | 8192/8192, max 1 |
| `mlp.down_proj` | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 |

GitHub에는 compact summary와 validation log만 포함한다. Raw 실행 산출물인 `.aimd`,
`rtl_results_ch*.csv`, `rtl_timing_ch*.csv`, chunk별 실행 디렉터리는 `.gitignore`로 제외한다.

## Llama3.2-1B WikiText Full PIM-Injected Decoder Layer Output Delta 실험

현재 PIM wrapper는 `model.model.layers`와 Llama 계열 projection 이름
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)을 갖는
decoder-only model을 기준으로 동작한다.

옵션은 다음과 같다.

- `--seq-len`: WikiText prompt token 길이.
- `--dataset-samples`: WikiText에서 prompt를 만들 때 사용할 sample 수.
- `--all-tokens`: 마지막 token만이 아니라 입력 prompt 전체 token의 decoder output delta를 기록한다.
- `--summary-output`: layer별 summary CSV 경로.
- `--elementwise-output`: element-wise delta CSV 경로.
- `--metadata-output`: 실행 metadata JSON 경로.
- `--model-id`: Hugging Face model id. online download, metadata, cache key에 사용한다.
- `--model-dir`: 로컬 model directory. 지정하지 않으면 `model-id` 기준 기본 cache 경로를 사용한다.
- `--ramulator`: 실행할 `ramulator2` binary 경로.
- `--resume`: 기존 PIM cache가 있으면 재사용하고, 없는 case만 Mode 2로 새로 실행한다.
- `--keep-run-artifacts`: `.aimd`, `rtl_results_ch*.csv` 등 case별 중간 산출물을 보존한다.
- `--cache-root`: PIM cache 저장 경로. 속도 비교 시 run별로 분리한다.
- `--mode2-token-batch`: 같은 projection의 여러 token을 하나의 Mode 2 trace로 묶어 실행한다.
- `--pim-workers`: 독립 Mode 2 실행을 병렬로 돌릴 worker 수.
- `--batch-token-chunk-size`: batch-token을 몇 token 단위 chunk로 나눌지 지정한다.
- `--offline`: Hugging Face model/dataset cache를 로컬에서만 사용한다.
- 출력 파일명에는 `model-id`와 `seq-len` tag가 자동으로 붙는다.

Mode 2 병렬화는 CPU core와 메모리 여유에 맞춰 조정한다. 현재 `seq_len=16` 기준으로
`--mode2-token-batch --pim-workers 4 --batch-token-chunk-size 4` 조합은 기존 결과와
동일한 summary/elementwise output을 내면서 Mode 2 구간을 약 10.21분에 완료했다.
8 physical core / 16 logical CPU 환경에서는 `--pim-workers 8 --batch-token-chunk-size 2`를
우선 시도하고, 최대 실험은 `--pim-workers 16 --batch-token-chunk-size 1`까지 고려한다.

```bash
python3 scripts/run_wikitext_full_pim_decoder_outputs.py \
  --model-id org/model-name \
  --seq-len 16 \
  --dataset-samples 64 \
  --all-tokens \
  --resume \
  --mode2-token-batch \
  --pim-workers 8 \
  --batch-token-chunk-size 2 \
  --summary-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/by_layer.csv \
  --elementwise-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/elementwise.csv \
  --metadata-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/metadata.json
```

로컬 모델 경로를 사용하는 실행은 다음과 같다.

```bash
python3 scripts/run_wikitext_full_pim_decoder_outputs.py \
  --model-id org/model-name \
  --model-dir /path/to/local-model \
  --seq-len 16 \
  --dataset-samples 64 \
  --all-tokens \
  --resume \
  --mode2-token-batch \
  --pim-workers 8 \
  --batch-token-chunk-size 2 \
  --summary-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/by_layer.csv \
  --elementwise-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/elementwise.csv \
  --metadata-output tests/result/llama32_error_accumulation/summary/decoder_layer_output/metadata.json
```

## Llama3.2-1B WikiText Full PIM-Injected Projection Output Delta 실험

decoder layer output 실험은 각 decoder block의 최종 hidden-state를 비교한다. 반면
projection output delta 실험은 full PIM injection 상태에서 각 decoder layer 내부의
7개 projection output을 GPU baseline과 직접 비교한다. 대상 projection은
`self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`,
`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`이다.

이 실험은 PIM output이 앞 layer부터 계속 주입된 상태에서 다음 layer/projection 입력이
달라지는 propagation 효과를 포함한다. 

summary CSV는 layer/projection 단위 평균으로 구성된다. Llama3.2-1B는 16개 decoder
layer와 7개 projection을 가지므로 summary는 총 112개 row가 된다.

주요 출력은 다음과 같다.

- `by_projection_*.csv`: 16 layer x 7 projection = 112개 summary row.
- `elementwise_*.csv`: 각 projection output element별 GPU/PIM BF16 hex, value, delta.
- `metadata_*.json`: model, seq_len, cache, Mode 2 실행 횟수, GPU/torch 설정.

실행 예시는 다음과 같다.

```bash
python3 scripts/run_wikitext_full_pim_projection_outputs.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --seq-len 16 \
  --dataset-samples 64 \
  --all-tokens \
  --resume \
  --mode2-token-batch \
  --pim-workers 8 \
  --batch-token-chunk-size 2 \
  --summary-output tests/result/llama32_error_accumulation/summary/projection_output/by_projection.csv \
  --elementwise-output tests/result/llama32_error_accumulation/summary/projection_output/elementwise.csv \
  --metadata-output tests/result/llama32_error_accumulation/summary/projection_output/metadata.json
```

## References

- GDDR6-AiM: "GDDR6-AiM - A 1ynm 1.25V 8Gb 16Gb/s/Pin GDDR6-Based Accelerator-in-Memory Supporting 1TFLOPS MAC Operation and Various Activation Functions for Deep Learning Application", JSSC 2023.
- AiM Simulator: [arkhadem/aim_simulator](https://github.com/arkhadem/aim_simulator)
- Verilator: [verilator/verilator](https://github.com/verilator/verilator)
- Ramulator2.0: [CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2)
