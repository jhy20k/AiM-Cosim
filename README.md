# AiM-CoSim: GDDR6-AiM Timing-Functional Co-Simulation Framework

Autumn Annual Conference of IEIE, 2026

`AiM-Cosim` proposes a simulation framework for SK hynix GDDR6-AiM timing-functional co-simulation.

## Key Contribution
- Integrates a Verilator-based RTL Co-Simulation Engine with the Ramulator2.0-based AiM Simulator to address the limitations of PU modeling in timing simulators.
- Designs an RTL model that reflects the SK hynix GDDR6-AiM dataflow and PU architecture.
- Analyzes PU execution latency separately from the latency defined in the AiM Simulator's DRAM timing model. If the RTL PU latency exceeds the abstracted PU operation latency in the AiM Simulator, execution is synchronized to the RTL PU latency.

## Repository Structure

```text
AiM-Cosim/
├── CMakeLists.txt
├── Makefile
├── pyproject.toml
├── README.md
├── LICENSE
├── docker/
├── extern/                 # submodules / external dependencies
├── src/
│   ├── core/               # C++ simulator / RTL co-simulation core
│   ├── rtl/                # RTL targets for Verilator
│   ├── configs/            # Ramulator2 / AiM runtime configurations
│   └── tools/              # standalone C++ helpers
├── scripts/
└── tests/
    └── result/
```

## Build Options

The default build uses the `MAG32` accumulator RTL. The accumulator width is selected using the CMake option
`ACC_MAG_WIDTH`.

```bash
cmake -S . -B build_mag37 -GNinja -DACC_MAG_WIDTH=37
cmake --build build_mag37 -j$(nproc)
```

The default value is `32`. The CMake configuration selects the corresponding RTL top for `32`, `37`, and other values, respectively.

## Current Validation Scope

The default build/test flow currently targets GDDR6-AiM RTL co-simulation. The default RTL top in `src/rtl` is
`pim_mac_tree_mag32`, and the accumulator uses
`accumulator_param #(.MAG_WIDTH(32), .Q_POINT(24))`. See `src/rtl/README.md` for detailed RTL configuration and accumulator format information.

The HBM-PIM-related code under `src/core/*hbmpim*`, `src/core/hbmpim`, `src/configs/hbmpim_*.yaml`, and
`src/tools/hbmpim_*` is a future-work scaffold. It is not part of the currently validated default path and should not be confused with the GDDR6-AiM results.

## From Initial Environment Setup to Execution

### Dependencies

The default development environment is Docker-based. Only the following tools are required on the host:

- `git`
- Docker Engine
- Docker Compose plugin

The container image includes Verilator `v5.024`, GCC 12, CMake, Ninja, Python 3.10, and Python packages for testing, plotting, and LLM experiments. For CUDA-based experiments such as Llama3.2 WikiText, use GPU passthrough configured with the NVIDIA Container Toolkit when running inside the container.

### Clone

Clone the repository together with its submodules.

```bash
git clone --recursive <repo-url> AiM-Cosim
cd AiM-Cosim
```

### Start the Docker Environment

Build the Docker image and start the container from the project root.

```bash
make docker-build
make docker-run
make docker-shell
```

The working directory inside the container is `/workspace`, where the host repository is mounted directly.

### Build

Run the default build inside the container. The default is the `MAG32` accumulator RTL.

```bash
cd /workspace
make build
```

After the build completes, the simulator binary is generated at:

```text
build_mag32/extern/aim_simulator/ramulator2
```

### Test

Run the default unit/regression tests inside the container.

```bash
make test
```

### Run

Mode 1 performs timing-only execution.

```bash
make run-timing TRACE=tests/data/tc1_single_mac16.trace
```

Mode 2 runs the timing simulation together with the Verilator RTL functional path.

```bash
make run-rtl TRACE=tests/data/tc1_single_mac16.trace
```

The direct execution format is:

```bash
./build_mag32/extern/aim_simulator/ramulator2 \
  -f <config.yaml> \
  -t tests/data/tc1_single_mac16.trace
```

The default Mode 2 configuration (`src/configs/aim_rtl.yaml`) executes the RTL functional path, but scoreboard gating that delays producer/consumer command issue according to RTL latency is disabled. To verify scoreboard gating, use a debug configuration with `verilator.scoreboard_gating: true`.

```bash
./build_mag32/extern/aim_simulator/ramulator2 \
  -f src/configs/aim_rtl_stage4_debug_mac.yaml \
  -t tests/data/tc1_single_mac16.trace
```

NOTE: The current AiM Simulator configuration assumes 32 channels.

## Additional Experiments

Mode 1 and Mode 2 cycle analysis is validated through the matmul regression.

```bash
bash scripts/run_matmul_regression.sh
```

Mode 1/Mode 2 cycles for each LLM projection, Mode 2 RTL result error, and simulation-speed comparisons are generated using the HF projection runner and summary scripts. Because Mode 1 is timing-only, projection error is evaluated by comparing the Mode 2 RTL results against CPU/GPU BF16 references.

```bash
python3 scripts/run_summer26_hf_stage5_7.py \
  --models Llama3.2-1B-Instruct \
  --projection-groups self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.gate_proj mlp.up_proj mlp.down_proj

python3 scripts/consolidate_summer26_results.py
python3 scripts/generate_summer26_llm_figures.py --force
```

`generate_summer26_llm_figures.py` checks coverage across multiple models by default. If only `Llama3.2-1B` results are available, the final coverage check may exit with a non-zero status, but the GPU functional comparison summary is generated beforehand.

The main summary files are:

- `tests/result/result_summer26/summary/hf_weight_mode_summary.csv`: Mode 1/Mode 2 cycles for each projection.
- `tests/result/result_summer26/summary/hf_weight_projection_aggregate_summary.csv`: Error summary for each projection.
- `tests/result/result_summer26/summary/hf_weight_actual_pre_activation_projection_aggregate.csv`: Pre-activation error summary using actual decoder projection inputs.
- `tests/result/result_summer26/summary/pre_activation_projection_comparison_summary.md`: Comparison between the existing random BF16 aggregate results and actual pre-activation results.
- `tests/result/result_summer26/summary/hf_weight_execution_time_comparison.csv`: Mode 1/Mode 2 wall-time comparison for each projection.
- `tests/result/result_summer26/figures/supplemental_torch_functional_aggregate.csv`: Summary of torch CUDA BF16 functional comparisons.
- `tests/result/result_summer26/figures/supplemental_cupy_cuda_aggregate.csv`: Supplemental summary for CuPy CUDA timing.

## Validation Results

`tests/result/result_summer26` currently contains projection validation results for `Llama3.2-1B-Instruct`.

| Validation Item | Result | Summary File |
|---|---:|---|
| Mode 1 / Mode 2 cycle | 24/24 PASS | `summary/hf_weight_execution_time_comparison.csv` |
| CPU/GPU reference timing | CPU 24/24, GPU 24/24 OK | `summary/hf_weight_all_projection_cpu_gpu_summary.csv` |
| Mode 2 vs CPU BF16 functional | Aggregates generated for 7 projections | `summary/hf_weight_projection_aggregate_summary.csv` |
| Mode 2 vs torch CUDA BF16 functional | 7/7 OK | `figures/supplemental_torch_functional_aggregate.csv` |
| Actual decoder pre-activation projection | Aggregates generated for 7 projections | `summary/hf_weight_actual_pre_activation_projection_aggregate.csv` |
| Mode 1 / Mode 2 simulation wall-time | 24 projection chunks compared | `summary/hf_weight_execution_time_comparison.csv` |

The comparison conditions are defined as follows.

- Three activation-source comparison: all cases use the same actual HF safetensors weights and `row_sharded` layout from layer 0 of `Llama3.2-1B-Instruct`, and compare the `pre_activation/RDMAC16` results without applying the activation function.
- Three input activation types:
  - `random_bf16`: Random BF16 input generated using seed 42 only.
  - `prompt_hash_bf16`: Deterministic random BF16 input generated from the prompt file and seed 42.
  - `actual hook`: BF16 input captured at the projection module during an actual Llama forward pass.
- CPU reference: BF16 weights and BF16 inputs are converted to NumPy FP32, GEMV is performed, and the result is rounded back to BF16.
- GPU reference: The same BF16 weights and BF16 inputs are evaluated using a torch CUDA BF16 projection. For numerical-comparison stability, this experiment fixes `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`. This setting disallows lower-precision reduced-accumulation paths during BF16 matmul reduction, thereby retaining more conservative intermediate accumulation precision. In contrast, when set to `True`, Tensor Core/cuBLAS-family backends may use reduced-precision reduction depending on the tensor shape, which can improve speed but can also make the delta between the GPU reference and AiM RTL more sensitive to backend optimization paths.
- Notation: Each cell represents `within_1_ulp, max ULP`.

| Projection | Random PIM-vs-CPU | Random PIM-vs-GPU | Prompt-hash PIM-vs-CPU | Prompt-hash PIM-vs-GPU | Actual hook PIM-vs-CPU | Actual hook PIM-vs-GPU |
|---|---:|---:|---:|---:|---:|---:|
| `self_attn.q_proj` | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 |
| `self_attn.k_proj` | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| `self_attn.v_proj` | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| `self_attn.o_proj` | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 1 |
| `mlp.gate_proj` | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8191/8192, max 8 |
| `mlp.up_proj` | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 0 | 8192/8192, max 1 | 8192/8192, max 1 |
| `mlp.down_proj` | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 |

Only compact summaries and validation logs are included on GitHub. Raw execution artifacts such as `.aimd`, `rtl_results_ch*.csv`, `rtl_timing_ch*.csv`, and per-chunk execution directories are excluded through `.gitignore`.

## Llama3.2-1B WikiText Full PIM-Injected Decoder Layer Output Delta Experiment

The current PIM wrapper targets decoder-only models that expose `model.model.layers` and Llama-style projection names
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

The available options are:

- `--seq-len`: WikiText prompt token length.
- `--dataset-samples`: Number of WikiText samples used to construct the prompt.
- `--all-tokens`: Records decoder output deltas for all tokens in the input prompt rather than only the final token.
- `--summary-output`: Path to the per-layer summary CSV.
- `--elementwise-output`: Path to the element-wise delta CSV.
- `--metadata-output`: Path to the execution metadata JSON.
- `--model-id`: Hugging Face model ID. Used for online download, metadata, and cache keys.
- `--model-dir`: Local model directory. If not specified, the default cache path derived from `model-id` is used.
- `--ramulator`: Path to the `ramulator2` binary to execute.
- `--resume`: Reuses an existing PIM cache when available and runs Mode 2 only for missing cases.
- `--keep-run-artifacts`: Preserves per-case intermediate artifacts such as `.aimd` and `rtl_results_ch*.csv`.
- `--cache-root`: Path for the PIM cache. Use separate directories per run when comparing execution speed.
- `--mode2-token-batch`: Batches multiple tokens from the same projection into a single Mode 2 trace.
- `--pim-workers`: Number of workers used to execute independent Mode 2 runs in parallel.
- `--batch-token-chunk-size`: Number of batch-tokens grouped into each chunk.
- `--offline`: Uses only the local Hugging Face model/dataset cache.
- Output filenames automatically include `model-id` and `seq-len` tags.

Adjust Mode 2 parallelism according to the available CPU cores and memory capacity. For `seq_len=16`, the current combination
`--mode2-token-batch --pim-workers 4 --batch-token-chunk-size 4` completed the Mode 2 section in approximately 10.21 minutes while producing the same summary/elementwise outputs as the existing results.
On a system with 8 physical cores / 16 logical CPUs, first try `--pim-workers 8 --batch-token-chunk-size 2`; for the most aggressive configuration, consider up to `--pim-workers 16 --batch-token-chunk-size 1`.

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

To run with a local model path:

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

## Llama3.2-1B WikiText Full PIM-Injected Projection Output Delta Experiment

The decoder-layer output experiment compares the final hidden state of each decoder block. In contrast, the projection-output delta experiment directly compares the seven projection outputs within each decoder layer against the GPU baseline under full PIM injection. The target projections are
`self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`,
`mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`.

This experiment includes the propagation effect in which PIM outputs are continuously injected from preceding layers, thereby changing the inputs to subsequent layers and projections.

The summary CSV contains averages at the layer/projection level. Because Llama3.2-1B has 16 decoder layers and 7 projections, the summary contains 112 rows in total.

The main outputs are:

- `by_projection_*.csv`: 16 layers x 7 projections = 112 summary rows.
- `elementwise_*.csv`: GPU/PIM BF16 hexadecimal values, numeric values, and deltas for each projection output element.
- `metadata_*.json`: Model, seq_len, cache, number of Mode 2 executions, and GPU/torch settings.

Example execution:

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
