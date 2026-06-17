# result_summer26 Stage Status

Status date: 2026-05-28 17:48:25 KST

This document summarizes execution status against `docs/experiment_plan.md`.
All generated artifacts are under `/home/jhlee/AiM-Cosim/tests/result/result_summer26`.

## Stage Summary

| Stage | Name | Status | Evidence |
|---:|---|---|---|
| 0 | Scope And Artifact Layout | DONE | `README.md`, `artifacts/model_sources.csv`, `artifacts/metadata_schema.csv` |
| 1 | Workload And Trace Specification | DONE | `artifacts/regression_matrix_36.csv`, `artifacts/llm_shapes.csv` |
| 2 | Functional Regression Baseline | DONE | 0 file-AIMD representative rows in `functional_regression/functional_summary.csv`; HF projection rows in `llm_shape/hf_weight_functional_summary.csv` |
| 3 | Matmul Timing Regression | DONE | 0/0 official 36-case Mode1/Mode2 rows PASS |
| 5 | LLM Shape Extension | DONE FOR ROW-SHARDED ACTIVE PROJECTION POLICY | 24 actual HF projection functional rows in summary: 24 row-sharded, 0 legacy broadcast |
| 6 | Tier 2 AiM Exact Simulation Time | DONE FOR ROW-SHARDED ACTIVE PROJECTION POLICY | 48/48 Mode1/Mode2 rows have matching cycles: 48 row-sharded, 0 legacy broadcast |
| 7 | CPU And GPU Reference Time | DONE | CPU measured for 24 projection rows with warmup 10; GPU measured for 24, unavailable for 0; execution comparison includes Mode1/Mode2 |
| 8 | Result Consolidation | DONE | `summary/final_experiment_report.md` and summary CSVs |

## Important Interpretation Notes

- Stage 3 is the official 36-case timing matrix. Pass/fail is based on Mode1 and Mode2 `memory_system_cycles`.
- Stage 2 official functional baseline uses file-mode AIMD and CPU BF16 reference. Random memory-init timing outputs are not used as functional correctness evidence.
- Simulation Time Tier1/perf_comparison is removed from the official Summer 2026 plan. Historical raw files may remain under `simulation_time/tier1_perf_comparison`, but they are not copied into `summary`.
- Stage 5-7 actual-weight execution covers seven module-level projections for the active model set: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`.
- The official LLM input activation is generated from `artifacts/prompt_inputs_maxnew16.json` using deterministic `prompt_hash_bf16`. This keeps all simulators on the same BF16 input vector; it is not a tokenizer/model-forward hidden-state extraction.
- The official LLM trace layout is `row_sharded`: each output row maps to `(channel, bank, physical_row)` across 32 channels and 16 banks, and Mode2 output is reconstructed from all `rtl_results_ch*.csv` files plus every `RDAF16` bank value. Legacy broadcast raw directories may remain on disk, but official summaries include the row-sharded rerun when available.
- CPU/GPU reference rows were generated in the `summer26` conda environment. Figure 2 uses CPU warmup10 and torch CUDA BF16 timing; CuPy CUDA measurements are kept as supplemental diagnostics only.

## Key Outputs

| Output | Path |
|---|---|
| Functional summary | `tests/result/result_summer26/summary/functional_summary.csv` |
| Matmul timing summary | `tests/result/result_summer26/summary/matmul_timing_summary.csv` |
| Row-sharded LLM trace update | `tests/result/result_summer26/summary/row_sharded_llm_trace_update.md` |
| LLM HF functional summary | `tests/result/result_summer26/summary/hf_weight_functional_summary.csv` |
| LLM HF aggregate summary | `tests/result/result_summer26/summary/hf_weight_projection_aggregate_summary.csv` |
| Tier2 Mode1/Mode2 summary | `tests/result/result_summer26/summary/hf_weight_mode_summary.csv` |
| CPU/GPU reference summary | `tests/result/result_summer26/summary/hf_weight_all_projection_cpu_gpu_summary.csv` |
| Execution time comparison | `tests/result/result_summer26/summary/hf_weight_execution_time_comparison.csv` |
| Execution time aggregate | `tests/result/result_summer26/summary/hf_weight_execution_time_aggregate.csv` |
