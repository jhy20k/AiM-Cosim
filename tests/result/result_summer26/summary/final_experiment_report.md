# Summer 2026 Experiment Completion Report

## Executive Summary

- Functional baseline: 0 representative file-AIMD regression shapes completed with CPU BF16 ULP reports.
- Matmul timing regression: 0/0 official 36-case shapes PASS for Mode1/Mode2 `memory_system_cycles` identity.
- Simulation Time Tier1/perf_comparison has been removed from the official result set.
- LLM actual weights: 24 active projection rows in summary with HF safetensors values: 24 row-sharded, 0 legacy broadcast.
- Row-sharded LLM trace update: implemented, smoke validated, and full `active_rows_rowshard32` summaries are now official.
- Tier2 exact timing: 48/48 Mode1/Mode2 rows have matching `memory_system_cycles`: 48 row-sharded, 0 legacy broadcast.
- CPU/GPU/reference execution comparison: 24 projection rows combine CPU, GPU, Mode1, and Mode2 timing; GPU backend is torch_cuda_bf16_kernel.

## LLM Actual-Weight Results

| Model | Projection | Conceptual shape | Chunk rows | Mode2 cycle sum | <=1 ULP | Max ULP |
|---|---|---:|---:|---:|---:|---:|
| Llama3.2-1B-Instruct | self_attn.q_proj | `2048x2048` | 2 | 67044 | 2047/2048 | 3 |
| Llama3.2-1B-Instruct | self_attn.k_proj | `2048x512` | 1 | 16251 | 512/512 | 1 |
| Llama3.2-1B-Instruct | self_attn.v_proj | `2048x512` | 1 | 16251 | 512/512 | 1 |
| Llama3.2-1B-Instruct | self_attn.o_proj | `2048x2048` | 2 | 67044 | 2048/2048 | 1 |
| Llama3.2-1B-Instruct | mlp.gate_proj | `2048x8192` | 8 | 268176 | 8190/8192 | 2 |
| Llama3.2-1B-Instruct | mlp.up_proj | `2048x8192` | 8 | 268176 | 8191/8192 | 3 |
| Llama3.2-1B-Instruct | mlp.down_proj | `8192x2048` | 2 | 263652 | 2048/2048 | 1 |

## Caveats

- Historical Tier1 raw files may remain under `simulation_time/tier1_perf_comparison`, but Tier1 is no longer part of the official summary.
- Legacy `active_rows` broadcast directories may remain on disk. Official row-sharded summaries use `active_rows_rowshard32` or `active_rows_rowshard32_chunk*`.
- CPU/GPU timing was recorded in the `summer26` conda environment. GPU timing is kernel-time only and remains separate from AiM simulated time.
