#!/usr/bin/env python3
"""Consolidate result_summer26 experiment outputs into summary artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import re
import shutil
import time


ROOT = Path("tests/result/result_summer26")
SUMMARY = ROOT / "summary"
FUNCTIONAL = ROOT / "functional_regression"
LLM = ROOT / "llm_shape"
TIER2 = ROOT / "simulation_time" / "tier2_aim_exact"
REFERENCE = ROOT / "reference_time" / "cpu_gpu"
MATMUL = ROOT / "matmul_regression"


MODEL_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama3.2-1B-Instruct",
    "LGAI-EXAONE/EXAONE-4.0-1.2B": "EXAONE-4.0-1.2B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini-Instruct",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}

MODEL_ORDER = {repo_id: i for i, repo_id in enumerate(MODEL_NAMES)}
GROUP_ORDER = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
    "q/o": 20,
    "k/v": 21,
    "gate/up": 22,
    "down": 23,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_functional_report(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    out = {
        "cycles": "",
        "n": "",
        "exact": "",
        "within_1": "",
        "within_2": "",
        "max_ulp": "",
        "avg_ulp": "",
    }
    if m := re.search(r"DRAM cycles=([0-9]+)", text):
        out["cycles"] = m.group(1)
    if m := re.search(r"Total output elements:\s+([0-9]+)", text):
        out["n"] = m.group(1)
    if m := re.search(r"Exact match \(0 ULP\):\s+([0-9]+/[0-9]+)", text):
        out["exact"] = m.group(1)
    if m := re.search(r"Within 1 ULP:\s+([0-9]+/[0-9]+)", text):
        out["within_1"] = m.group(1)
    if m := re.search(r"Within 2 ULP:\s+([0-9]+/[0-9]+)", text):
        out["within_2"] = m.group(1)
    if m := re.search(r"Max ULP difference:\s+([0-9]+)", text):
        out["max_ulp"] = m.group(1)
    if m := re.search(r"Avg ULP difference:\s+([0-9.]+)", text):
        out["avg_ulp"] = m.group(1)
    return out


def parse_cycles(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(errors="ignore").splitlines():
        if "memory_system_cycles:" in line:
            return line.split()[1]
    return ""


def parse_wall_csv(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return ""
    return rows[0].get("wall_time_s", "")


def model_name_from_repo(repo_id: str) -> str:
    return MODEL_NAMES.get(repo_id, repo_id.split("/")[-1])


def is_gpu_backend(backend: str) -> bool:
    return "cuda" in backend or backend == "gpu"


def parse_fraction(value: object) -> tuple[int, int]:
    text = str(value or "")
    if "/" not in text:
        return 0, 0
    num, den = text.split("/", 1)
    return int(num), int(den)


def parse_shape(shape: object) -> tuple[int, int]:
    q_text, r_text = str(shape or "0x0").split("x", 1)
    return int(q_text), int(r_text)


def parse_hf_report_comments(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("# ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        out[key.strip()] = value.strip()
    return out


def consolidate_hf_weight_functional() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for metadata_path in sorted(LLM.glob("*/metadata.json")):
        run_dir = metadata_path.parent
        report = run_dir / "functional_report.csv"
        if not report.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("weight_source") != "hf_safetensors":
            continue
        comments = parse_hf_report_comments(report)
        repo_id = metadata.get("repo_id", "")
        q = metadata.get("q", comments.get("q", ""))
        r = metadata.get("r", comments.get("r", ""))
        candidates.append(
            {
                "run_id": run_dir.name,
                "model": model_name_from_repo(repo_id),
                "model_source": f"https://huggingface.co/{repo_id}" if repo_id else "",
                "value_policy": metadata.get("value_policy", comments.get("value_policy", "")),
                "weight_source": metadata.get("weight_source", comments.get("weight_source", "")),
                "weight_seed": metadata.get("weight_seed", comments.get("weight_seed", "")),
                "projection_group": metadata.get("projection_group", comments.get("projection_group", "")),
                "tensor_name": metadata.get("tensor_name", comments.get("tensor_name", "")),
                "layer": metadata.get("layer", comments.get("layer", "")),
                "q": q,
                "r": r,
                "shape": f"{q}x{r}" if q and r else comments.get("shape", ""),
                "activation_seed": metadata.get("seed", comments.get("seed", "")),
                "activation_source": metadata.get("activation_source", comments.get("activation_source", "random_bf16")),
                "prompt_file": metadata.get("prompt_file", comments.get("prompt_file", "")),
                "max_new_tokens": metadata.get("max_new_tokens", comments.get("max_new_tokens", "")),
                "prompt_count": metadata.get("prompt_count", comments.get("prompt_count", "")),
                "prompt_digest": metadata.get("prompt_digest", comments.get("prompt_digest", "")),
                "channel_layout": metadata.get("channel_layout", "broadcast"),
                "num_channels": metadata.get("num_channels", "32"),
                "row_shard_policy": metadata.get("row_shard_policy", "full_broadcast"),
                "mode2_memory_system_cycles": comments.get("memory_system_cycles", ""),
                "mode2_wall_time_s": comments.get("wall_time_s", ""),
                "exact": comments.get("exact", ""),
                "within_1_ulp": comments.get("within_1_ulp", ""),
                "within_2_ulp": comments.get("within_2_ulp", ""),
                "max_ulp": comments.get("max_ulp", ""),
                "avg_ulp": comments.get("avg_ulp", ""),
                "result_dir": str(run_dir),
            }
        )

    by_case: dict[tuple[object, object, object, object, object, object, object], dict[str, object]] = {}
    for row in candidates:
        key = (
            row.get("model_source"),
            row.get("projection_group"),
            row.get("tensor_name"),
            row.get("q"),
            row.get("r"),
            row.get("activation_seed"),
            row.get("activation_source"),
            row.get("prompt_digest"),
            row.get("weight_source"),
        )
        prev = by_case.get(key)
        if prev is None:
            by_case[key] = row
            continue
        prev_is_row_sharded = prev.get("channel_layout") == "row_sharded"
        row_is_row_sharded = row.get("channel_layout") == "row_sharded"
        if row_is_row_sharded and not prev_is_row_sharded:
            by_case[key] = row

    rows = list(by_case.values())
    if any(row.get("channel_layout") == "row_sharded" for row in rows):
        rows = [row for row in rows if row.get("channel_layout") == "row_sharded"]
    if any(row.get("activation_source") == "prompt_hash_bf16" for row in rows):
        rows = [row for row in rows if row.get("activation_source") == "prompt_hash_bf16"]
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row.get("model_source", "")).replace("https://huggingface.co/", ""), 99),
            GROUP_ORDER.get(str(row.get("projection_group", "")), 99),
            int(row.get("q") or 0),
            int(row.get("r") or 0),
        )
    )
    fieldnames = [
        "run_id",
        "model",
        "model_source",
        "value_policy",
        "weight_source",
        "weight_seed",
        "projection_group",
        "tensor_name",
        "layer",
        "q",
        "r",
        "shape",
        "activation_seed",
        "activation_source",
        "prompt_file",
        "max_new_tokens",
        "prompt_count",
        "prompt_digest",
        "channel_layout",
        "num_channels",
        "row_shard_policy",
        "mode2_memory_system_cycles",
        "mode2_wall_time_s",
        "exact",
        "within_1_ulp",
        "within_2_ulp",
        "max_ulp",
        "avg_ulp",
        "result_dir",
    ]
    write_csv(LLM / "hf_weight_functional_summary.csv", rows, fieldnames)
    write_csv(SUMMARY / "hf_weight_functional_summary.csv", rows, fieldnames)
    consolidate_hf_weight_functional_aggregate(rows)
    return rows


def consolidate_hf_weight_functional_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in rows:
        key = (row.get("model"), row.get("model_source"), row.get("projection_group"))
        item = grouped.setdefault(
            key,
            {
                "model": row.get("model", ""),
                "model_source": row.get("model_source", ""),
                "projection_group": row.get("projection_group", ""),
                "q": int(row.get("q") or 0),
                "r_sum": 0,
                "execution_rows": 0,
                "cycles_sum": 0,
                "wall_sum": 0.0,
                "exact_num": 0,
                "within_1_num": 0,
                "within_2_num": 0,
                "den": 0,
                "max_ulp": 0,
                "avg_weighted_sum": 0.0,
            },
        )
        r = int(row.get("r") or 0)
        item["r_sum"] = int(item["r_sum"]) + r
        item["execution_rows"] = int(item["execution_rows"]) + 1
        item["cycles_sum"] = int(item["cycles_sum"]) + int(row.get("mode2_memory_system_cycles") or 0)
        item["wall_sum"] = float(item["wall_sum"]) + float(row.get("mode2_wall_time_s") or 0.0)
        exact_num, den = parse_fraction(row.get("exact"))
        within_1_num, _ = parse_fraction(row.get("within_1_ulp"))
        within_2_num, _ = parse_fraction(row.get("within_2_ulp"))
        item["exact_num"] = int(item["exact_num"]) + exact_num
        item["within_1_num"] = int(item["within_1_num"]) + within_1_num
        item["within_2_num"] = int(item["within_2_num"]) + within_2_num
        item["den"] = int(item["den"]) + den
        item["max_ulp"] = max(int(item["max_ulp"]), int(row.get("max_ulp") or 0))
        item["avg_weighted_sum"] = float(item["avg_weighted_sum"]) + float(row.get("avg_ulp") or 0.0) * den

    out: list[dict[str, object]] = []
    for item in grouped.values():
        den = int(item["den"])
        rows_n = int(item["execution_rows"])
        avg = (float(item["avg_weighted_sum"]) / den) if den else 0.0
        out.append(
            {
                "model": item["model"],
                "model_source": item["model_source"],
                "projection_group": item["projection_group"],
                "conceptual_shape": f"{item['q']}x{item['r_sum']}",
                "execution_rows": rows_n,
                "chunked": "yes" if rows_n > 1 else "no",
                "mode2_memory_system_cycles_sum": item["cycles_sum"],
                "mode2_wall_time_s_sum": f"{float(item['wall_sum']):.6f}",
                "exact": f"{item['exact_num']}/{den}" if den else "",
                "within_1_ulp": f"{item['within_1_num']}/{den}" if den else "",
                "within_2_ulp": f"{item['within_2_num']}/{den}" if den else "",
                "max_ulp": item["max_ulp"],
                "avg_ulp_weighted": f"{avg:.6f}" if den else "",
            }
        )
    out.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row.get("model_source", "")).replace("https://huggingface.co/", ""), 99),
            GROUP_ORDER.get(str(row.get("projection_group", "")), 99),
        )
    )
    fields = [
        "model",
        "model_source",
        "projection_group",
        "conceptual_shape",
        "execution_rows",
        "chunked",
        "mode2_memory_system_cycles_sum",
        "mode2_wall_time_s_sum",
        "exact",
        "within_1_ulp",
        "within_2_ulp",
        "max_ulp",
        "avg_ulp_weighted",
    ]
    write_csv(LLM / "hf_weight_projection_aggregate_summary.csv", out, fields)
    write_csv(SUMMARY / "hf_weight_projection_aggregate_summary.csv", out, fields)
    return out


def consolidate_functional() -> list[dict[str, object]]:
    host = platform.node()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows: list[dict[str, object]] = []
    for q, r in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]:
        report = FUNCTIONAL / f"comparison_p1_q{q}_r{r}_seed42.txt"
        if not report.exists():
            continue
        stats = parse_functional_report(report)
        rows.append(
            {
                "run_id": f"p1_q{q}_r{r}",
                "shape": f"1x{q}x{r}",
                "p": 1,
                "q": q,
                "r": r,
                "workload_source": "representative_functional_regression",
                "value_policy": "file_aimd_numpy_bf16_seed42",
                "activation_seed": 42,
                "comparison_basis": "AiMRTL Mode2 rtl_results_ch0.csv vs CPU BF16 reference",
                "mode2_memory_system_cycles": stats["cycles"],
                "total_output_elements": stats["n"],
                "exact": stats["exact"],
                "within_1_ulp": stats["within_1"],
                "within_2_ulp": stats["within_2"],
                "max_ulp": stats["max_ulp"],
                "avg_ulp": stats["avg_ulp"],
                "report": str(report),
                "status": "PASS_ULP_RECORDED",
                "host": host,
                "timestamp": timestamp,
                "command": "RESULT_DIR=tests/result/result_summer26/functional_regression bash scripts/run_functional_comparison.sh Q R 42",
            }
        )

    fieldnames = [
        "run_id",
        "shape",
        "p",
        "q",
        "r",
        "workload_source",
        "value_policy",
        "activation_seed",
        "comparison_basis",
        "mode2_memory_system_cycles",
        "total_output_elements",
        "exact",
        "within_1_ulp",
        "within_2_ulp",
        "max_ulp",
        "avg_ulp",
        "report",
        "status",
        "host",
        "timestamp",
        "command",
    ]
    write_csv(FUNCTIONAL / "functional_summary.csv", rows, fieldnames)
    write_csv(SUMMARY / "functional_summary.csv", rows, fieldnames)

    note = FUNCTIONAL / "random_init_functional_reports_note.md"
    note.write_text(
        """# Random-init Timing Outputs Are Not Official Functional Baseline

`matmul_regression/functional` and the older `functional_regression/regression_reports`
were extracted from Mode2 timing-regression runs that used random memory-manager
initialization. Those values are useful for checking that Verilator emitted data,
but they do not share the file-based AIMD tensor values used by the CPU BF16
reference flow.

The official Stage 2 functional summary for `result_summer26` is therefore
`functional_summary.csv`, generated from `run_functional_comparison.sh` file-mode
AIMD runs for the representative regression shapes.
""",
        encoding="utf-8",
    )
    return rows


def consolidate_tier2() -> list[dict[str, object]]:
    hf_rows = read_csv(LLM / "hf_weight_functional_summary.csv")
    out: list[dict[str, object]] = []
    for row in hf_rows:
        run_dir = Path(row["result_dir"])
        metadata = json.loads((run_dir / "metadata.json").read_text())
        repo_id = metadata["repo_id"]
        model = model_name_from_repo(repo_id)
        mode1_cycles = parse_cycles(run_dir / "mode1.txt")
        mode1_wall = parse_wall_csv(run_dir / "mode1_wall_time.csv")
        mode2_cycles = row["mode2_memory_system_cycles"]
        mode2_wall = row["mode2_wall_time_s"]
        result = "PASS" if mode1_cycles and mode1_cycles == mode2_cycles else "MODE1_MISSING_OR_MISMATCH"
        base = {
            "run_id": row["run_id"],
            "model": model,
            "model_source": row["model_source"],
            "value_policy": row["value_policy"],
            "weight_source": row["weight_source"],
            "weight_seed": row["weight_seed"],
            "projection_group": row["projection_group"],
            "tensor_name": row["tensor_name"],
            "layer": row["layer"],
            "q": row["q"],
            "r": row["r"],
            "shape": row["shape"],
            "aimd_layout": metadata.get("aimd_layout", "dense_max_rows"),
            "aimd_rows_per_bank": metadata.get("aimd_rows_per_bank", "16384"),
            "channel_layout": metadata.get("channel_layout", "broadcast"),
            "num_channels": metadata.get("num_channels", "32"),
            "row_shard_policy": metadata.get("row_shard_policy", "full_broadcast"),
            "activation_seed": metadata.get("seed", ""),
            "activation_source": metadata.get("activation_source", "random_bf16"),
            "prompt_file": metadata.get("prompt_file", ""),
            "max_new_tokens": metadata.get("max_new_tokens", ""),
            "prompt_count": metadata.get("prompt_count", ""),
            "prompt_digest": metadata.get("prompt_digest", ""),
            "result": result,
        }
        out.append(
            {
                **base,
                "mode": "Mode1",
                "memory_system_cycles": mode1_cycles,
                "simulated_time_ns_1ghz": mode1_cycles,
                "wall_time_s": mode1_wall,
                "slowdown_vs_mode1": "1.000000" if mode1_wall else "",
            }
        )
        slowdown = ""
        if mode1_wall and mode2_wall:
            slowdown = f"{float(mode2_wall) / float(mode1_wall):.6f}"
        out.append(
            {
                **base,
                "mode": "Mode2",
                "memory_system_cycles": mode2_cycles,
                "simulated_time_ns_1ghz": mode2_cycles,
                "wall_time_s": mode2_wall,
                "slowdown_vs_mode1": slowdown,
            }
        )
    fieldnames = [
        "run_id",
        "model",
        "model_source",
        "value_policy",
        "weight_source",
        "weight_seed",
        "projection_group",
        "tensor_name",
        "layer",
        "q",
        "r",
        "shape",
        "aimd_layout",
        "aimd_rows_per_bank",
        "channel_layout",
        "num_channels",
        "row_shard_policy",
        "activation_seed",
        "activation_source",
        "prompt_file",
        "max_new_tokens",
        "prompt_count",
        "prompt_digest",
        "mode",
        "memory_system_cycles",
        "simulated_time_ns_1ghz",
        "wall_time_s",
        "slowdown_vs_mode1",
        "result",
    ]
    write_csv(TIER2 / "hf_weight_mode_summary.csv", out, fieldnames)
    write_csv(SUMMARY / "hf_weight_mode_summary.csv", out, fieldnames)
    return out


def consolidate_reference_time(tier2_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    official_keys = {
        (
            str(row.get("model_source", "")),
            str(row.get("projection_group", "")),
            str(row.get("tensor_name", "")),
            str(row.get("shape", "")),
            str(row.get("activation_source", "")),
            str(row.get("prompt_digest", "")),
        )
        for row in tier2_rows
    }
    all_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for path in sorted(REFERENCE.glob("hf_weight_*_cpu_gpu.csv")):
        if "summary" in path.name or path.name.startswith("random_"):
            continue
        for row in read_csv(path):
            official_key = (
                row.get("model_source", ""),
                row.get("projection_group", ""),
                row.get("tensor_name", ""),
                row.get("shape", ""),
                row.get("activation_source", ""),
                row.get("prompt_digest", ""),
            )
            if official_keys and official_key not in official_keys:
                continue
            key = (
                row.get("model_source", ""),
                row.get("projection_group", ""),
                row.get("tensor_name", ""),
                row.get("shape", ""),
                row.get("activation_source", ""),
                row.get("prompt_digest", ""),
                row.get("backend", ""),
                row.get("status", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)

    all_rows.sort(
        key=lambda row: (
            MODEL_ORDER.get(row.get("repo_id", ""), 99),
            GROUP_ORDER.get(row.get("projection_group", ""), 99),
            row.get("backend", ""),
        )
    )
    fieldnames = list(all_rows[0]) if all_rows else [
        "repo_id",
        "model_source",
        "projection_group",
        "tensor_name",
        "layer",
        "q",
        "r",
        "shape",
        "value_policy",
        "weight_source",
        "weight_seed",
        "backend",
        "status",
        "warmup",
        "repeats",
        "mean_s",
        "median_s",
        "stddev_s",
        "min_s",
        "max_s",
    ]
    write_csv(REFERENCE / "hf_weight_all_projection_cpu_gpu_summary.csv", all_rows, fieldnames)
    write_csv(SUMMARY / "hf_weight_all_projection_cpu_gpu_summary.csv", all_rows, fieldnames)

    by_key_backend: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = {}
    for row in all_rows:
        key = (
            str(row.get("model_source", "")),
            str(row.get("projection_group", "")),
            str(row.get("tensor_name", "")),
            str(row.get("shape", "")),
            str(row.get("activation_source", "")),
            str(row.get("prompt_digest", "")),
        )
        by_key_backend.setdefault(key, {})[str(row.get("backend", ""))] = row

    modes: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = {}
    for row in tier2_rows:
        key = (
            str(row.get("model_source", "")),
            str(row.get("projection_group", "")),
            str(row.get("tensor_name", "")),
            str(row.get("shape", "")),
            str(row.get("activation_source", "")),
            str(row.get("prompt_digest", "")),
        )
        modes.setdefault(key, {})[str(row.get("mode", ""))] = row

    comparison: list[dict[str, object]] = []
    for key, backends in by_key_backend.items():
        cpu = backends.get("numpy_cpu", {})
        gpu = next((row for backend, row in backends.items() if is_gpu_backend(backend) and row.get("status") == "ok"), {})
        mode1 = modes.get(key, {}).get("Mode1", {})
        mode2 = modes.get(key, {}).get("Mode2", {})
        cpu_mean = str(cpu.get("mean_s", ""))
        gpu_mean = str(gpu.get("mean_s", ""))
        gpu_speedup = ""
        gpu_us = ""
        if cpu_mean and gpu_mean:
            gpu_s = float(gpu_mean)
            gpu_us = f"{gpu_s * 1e6:.3f}"
            gpu_speedup = f"{float(cpu_mean) / gpu_s:.6f}" if gpu_s else ""
        model_source, projection_group, tensor_name, shape, activation_source, prompt_digest = key
        comparison.append(
            {
                "model": model_name_from_repo(str(cpu.get("repo_id", "") or model_source.replace("https://huggingface.co/", ""))),
                "model_source": model_source,
                "projection_group": projection_group,
                "tensor_name": tensor_name,
                "shape": shape,
                "activation_source": activation_source,
                "prompt_digest": prompt_digest,
                "cpu_backend": cpu.get("backend", ""),
                "cpu_mean_s": cpu_mean,
                "gpu_backend": gpu.get("backend", ""),
                "gpu_mean_s": gpu_mean,
                "gpu_kernel_us": gpu_us,
                "gpu_speedup_vs_cpu": gpu_speedup,
                "mode1_memory_system_cycles": mode1.get("memory_system_cycles", ""),
                "mode1_simulated_time_ns_1ghz": mode1.get("simulated_time_ns_1ghz", ""),
                "mode1_simulated_time_s_1ghz": f"{float(mode1.get('simulated_time_ns_1ghz', '') or 0.0) / 1e9:.9f}",
                "mode1_wall_time_s": mode1.get("wall_time_s", ""),
                "mode2_memory_system_cycles": mode2.get("memory_system_cycles", ""),
                "mode2_simulated_time_ns_1ghz": mode2.get("simulated_time_ns_1ghz", ""),
                "mode2_simulated_time_s_1ghz": f"{float(mode2.get('simulated_time_ns_1ghz', '') or 0.0) / 1e9:.9f}",
                "mode2_wall_time_s": mode2.get("wall_time_s", ""),
                "mode2_slowdown_vs_mode1": mode2.get("slowdown_vs_mode1", ""),
                "mode_cycle_result": mode2.get("result", mode1.get("result", "")),
                "warmup": cpu.get("warmup", ""),
                "repeats": cpu.get("repeats", ""),
            }
        )

    comparison.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row.get("model_source", "")).replace("https://huggingface.co/", ""), 99),
            GROUP_ORDER.get(str(row.get("projection_group", "")), 99),
        )
    )
    comparison_fields = [
        "model",
        "model_source",
        "projection_group",
        "tensor_name",
        "shape",
        "activation_source",
        "prompt_digest",
        "cpu_backend",
        "cpu_mean_s",
        "gpu_backend",
        "gpu_mean_s",
        "gpu_kernel_us",
        "gpu_speedup_vs_cpu",
        "mode1_memory_system_cycles",
        "mode1_simulated_time_ns_1ghz",
        "mode1_simulated_time_s_1ghz",
        "mode1_wall_time_s",
        "mode2_memory_system_cycles",
        "mode2_simulated_time_ns_1ghz",
        "mode2_simulated_time_s_1ghz",
        "mode2_wall_time_s",
        "mode2_slowdown_vs_mode1",
        "mode_cycle_result",
        "warmup",
        "repeats",
    ]
    write_csv(REFERENCE / "hf_weight_execution_time_comparison.csv", comparison, comparison_fields)
    write_csv(SUMMARY / "hf_weight_execution_time_comparison.csv", comparison, comparison_fields)
    consolidate_reference_time_aggregate(comparison)
    return comparison


def consolidate_reference_time_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in rows:
        key = (row.get("model"), row.get("model_source"), row.get("projection_group"))
        q, r = parse_shape(row.get("shape"))
        item = grouped.setdefault(
            key,
            {
                "model": row.get("model", ""),
                "model_source": row.get("model_source", ""),
                "projection_group": row.get("projection_group", ""),
                "q": q,
                "r_sum": 0,
                "execution_rows": 0,
                "cpu_sum": 0.0,
                "gpu_sum": 0.0,
                "gpu_us_sum": 0.0,
                "mode1_cycles_sum": 0,
                "mode1_wall_sum": 0.0,
                "mode2_cycles_sum": 0,
                "mode2_wall_sum": 0.0,
                "slowdown_sum": 0.0,
                "mode_cycle_result": "PASS",
            },
        )
        item["r_sum"] = int(item["r_sum"]) + r
        item["execution_rows"] = int(item["execution_rows"]) + 1
        item["cpu_sum"] = float(item["cpu_sum"]) + float(row.get("cpu_mean_s") or 0.0)
        item["gpu_sum"] = float(item["gpu_sum"]) + float(row.get("gpu_mean_s") or 0.0)
        item["gpu_us_sum"] = float(item["gpu_us_sum"]) + float(row.get("gpu_kernel_us") or 0.0)
        item["mode1_cycles_sum"] = int(item["mode1_cycles_sum"]) + int(row.get("mode1_memory_system_cycles") or 0)
        item["mode1_wall_sum"] = float(item["mode1_wall_sum"]) + float(row.get("mode1_wall_time_s") or 0.0)
        item["mode2_cycles_sum"] = int(item["mode2_cycles_sum"]) + int(row.get("mode2_memory_system_cycles") or 0)
        item["mode2_wall_sum"] = float(item["mode2_wall_sum"]) + float(row.get("mode2_wall_time_s") or 0.0)
        item["slowdown_sum"] = float(item["slowdown_sum"]) + float(row.get("mode2_slowdown_vs_mode1") or 0.0)
        if row.get("mode_cycle_result") != "PASS":
            item["mode_cycle_result"] = row.get("mode_cycle_result", "")

    out: list[dict[str, object]] = []
    for item in grouped.values():
        rows_n = int(item["execution_rows"])
        cpu_sum = float(item["cpu_sum"])
        gpu_sum = float(item["gpu_sum"])
        out.append(
            {
                "model": item["model"],
                "model_source": item["model_source"],
                "projection_group": item["projection_group"],
                "conceptual_shape": f"{item['q']}x{item['r_sum']}",
                "execution_rows": rows_n,
                "chunked": "yes" if rows_n > 1 else "no",
                "cpu_mean_s_sum": f"{cpu_sum:.9f}",
                "gpu_mean_s_sum": f"{gpu_sum:.9f}",
                "gpu_kernel_us_sum": f"{float(item['gpu_us_sum']):.3f}",
                "gpu_speedup_vs_cpu_sum": f"{(cpu_sum / gpu_sum):.6f}" if gpu_sum else "",
                "mode1_memory_system_cycles_sum": item["mode1_cycles_sum"],
                "mode1_simulated_time_s_1ghz_sum": f"{int(item['mode1_cycles_sum']) / 1e9:.9f}",
                "mode1_wall_time_s_sum": f"{float(item['mode1_wall_sum']):.6f}",
                "mode2_memory_system_cycles_sum": item["mode2_cycles_sum"],
                "mode2_simulated_time_s_1ghz_sum": f"{int(item['mode2_cycles_sum']) / 1e9:.9f}",
                "mode2_wall_time_s_sum": f"{float(item['mode2_wall_sum']):.6f}",
                "mode2_slowdown_vs_mode1_sum": f"{float(item['slowdown_sum']):.6f}",
                "mode_cycle_result": item["mode_cycle_result"],
            }
        )
    out.sort(
        key=lambda row: (
            MODEL_ORDER.get(str(row.get("model_source", "")).replace("https://huggingface.co/", ""), 99),
            GROUP_ORDER.get(str(row.get("projection_group", "")), 99),
        )
    )
    fields = [
        "model",
        "model_source",
        "projection_group",
        "conceptual_shape",
        "execution_rows",
        "chunked",
        "cpu_mean_s_sum",
        "gpu_mean_s_sum",
        "gpu_kernel_us_sum",
        "gpu_speedup_vs_cpu_sum",
        "mode1_memory_system_cycles_sum",
        "mode1_simulated_time_s_1ghz_sum",
        "mode1_wall_time_s_sum",
        "mode2_memory_system_cycles_sum",
        "mode2_simulated_time_s_1ghz_sum",
        "mode2_wall_time_s_sum",
        "mode2_slowdown_vs_mode1_sum",
        "mode_cycle_result",
    ]
    write_csv(REFERENCE / "hf_weight_execution_time_aggregate.csv", out, fields)
    write_csv(SUMMARY / "hf_weight_execution_time_aggregate.csv", out, fields)
    return out


def copy_summary_inputs() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for legacy in ["tier1_perf_comparison_summary.csv", "tier1_workload_manifest.csv"]:
        (SUMMARY / legacy).unlink(missing_ok=True)
    for src, dst_name in [
        (MATMUL / "matmul_timing_summary.csv", "matmul_timing_summary.csv"),
        (LLM / "hf_weight_functional_summary.csv", "hf_weight_functional_summary.csv"),
        (LLM / "hf_weight_projection_aggregate_summary.csv", "hf_weight_projection_aggregate_summary.csv"),
        (REFERENCE / "hf_weight_all_projection_cpu_gpu_summary.csv", "hf_weight_all_projection_cpu_gpu_summary.csv"),
        (REFERENCE / "hf_weight_execution_time_comparison.csv", "hf_weight_execution_time_comparison.csv"),
        (REFERENCE / "hf_weight_execution_time_aggregate.csv", "hf_weight_execution_time_aggregate.csv"),
    ]:
        if src.exists():
            shutil.copyfile(src, SUMMARY / dst_name)


def write_stage_status(functional_rows: list[dict[str, object]], tier2_rows: list[dict[str, object]]) -> None:
    matmul_rows = read_csv(MATMUL / "matmul_timing_summary.csv")
    hf_rows = read_csv(LLM / "hf_weight_functional_summary.csv")
    cpu_rows = read_csv(REFERENCE / "hf_weight_all_projection_cpu_gpu_summary.csv")

    matmul_pass = sum(1 for r in matmul_rows if r.get("result") == "PASS")
    tier2_pass_modes = sum(1 for r in tier2_rows if r.get("result") == "PASS")
    hf_row_sharded = sum(1 for r in hf_rows if r.get("channel_layout") == "row_sharded")
    hf_broadcast = sum(1 for r in hf_rows if r.get("channel_layout") == "broadcast")
    tier2_row_sharded = sum(1 for r in tier2_rows if r.get("channel_layout") == "row_sharded")
    tier2_broadcast = sum(1 for r in tier2_rows if r.get("channel_layout") == "broadcast")
    if hf_row_sharded and not hf_broadcast:
        stage5_status = "DONE FOR ROW-SHARDED ACTIVE PROJECTION POLICY"
    elif hf_row_sharded:
        stage5_status = "PARTIAL ROW-SHARDED; FULL RERUN PENDING"
    else:
        stage5_status = "LEGACY BROADCAST COMPLETE; ROW-SHARDED FULL RERUN PENDING"
    if tier2_row_sharded and not tier2_broadcast:
        stage6_status = "DONE FOR ROW-SHARDED ACTIVE PROJECTION POLICY"
    elif tier2_row_sharded:
        stage6_status = "PARTIAL ROW-SHARDED; FULL RERUN PENDING"
    else:
        stage6_status = "LEGACY BROADCAST COMPLETE; ROW-SHARDED FULL RERUN PENDING"
    cpu_ok = sum(1 for r in cpu_rows if r.get("backend") == "numpy_cpu" and r.get("status") == "ok")
    gpu_rows = [r for r in cpu_rows if is_gpu_backend(r.get("backend", ""))]
    gpu_ok = sum(1 for r in gpu_rows if r.get("status") == "ok")
    gpu_unavail = sum(1 for r in gpu_rows if r.get("status", "").startswith("unavailable"))

    text = f"""# result_summer26 Stage Status

Status date: {time.strftime('%Y-%m-%d %H:%M:%S')} KST

This document summarizes execution status against `docs/experiment_plan.md`.
All generated artifacts are under `/home/jhlee/AiM-Cosim/tests/result/result_summer26`.

## Stage Summary

| Stage | Name | Status | Evidence |
|---:|---|---|---|
| 0 | Scope And Artifact Layout | DONE | `README.md`, `artifacts/model_sources.csv`, `artifacts/metadata_schema.csv` |
| 1 | Workload And Trace Specification | DONE | `artifacts/regression_matrix_36.csv`, `artifacts/llm_shapes.csv` |
| 2 | Functional Regression Baseline | DONE | {len(functional_rows)} file-AIMD representative rows in `functional_regression/functional_summary.csv`; HF projection rows in `llm_shape/hf_weight_functional_summary.csv` |
| 3 | Matmul Timing Regression | DONE | {matmul_pass}/{len(matmul_rows)} official 36-case Mode1/Mode2 rows PASS |
| 5 | LLM Shape Extension | {stage5_status} | {len(hf_rows)} actual HF projection functional rows in summary: {hf_row_sharded} row-sharded, {hf_broadcast} legacy broadcast |
| 6 | Tier 2 AiM Exact Simulation Time | {stage6_status} | {tier2_pass_modes}/{len(tier2_rows)} Mode1/Mode2 rows have matching cycles: {tier2_row_sharded} row-sharded, {tier2_broadcast} legacy broadcast |
| 7 | CPU And GPU Reference Time | DONE | CPU measured for {cpu_ok} projection rows with warmup 10; GPU measured for {gpu_ok}, unavailable for {gpu_unavail}; execution comparison includes Mode1/Mode2 |
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
"""
    (SUMMARY / "stage_status.md").write_text(text, encoding="utf-8")


def write_final_report(functional_rows: list[dict[str, object]], tier2_rows: list[dict[str, object]]) -> None:
    matmul_rows = read_csv(MATMUL / "matmul_timing_summary.csv")
    hf_rows = read_csv(LLM / "hf_weight_functional_summary.csv")
    hf_agg_rows = read_csv(LLM / "hf_weight_projection_aggregate_summary.csv")
    cpu_rows = read_csv(REFERENCE / "hf_weight_all_projection_cpu_gpu_summary.csv")
    execution_rows = read_csv(REFERENCE / "hf_weight_execution_time_comparison.csv")

    matmul_pass = sum(1 for r in matmul_rows if r.get("result") == "PASS")
    hf_row_sharded = sum(1 for r in hf_rows if r.get("channel_layout") == "row_sharded")
    hf_broadcast = sum(1 for r in hf_rows if r.get("channel_layout") == "broadcast")
    tier2_row_sharded = sum(1 for r in tier2_rows if r.get("channel_layout") == "row_sharded")
    tier2_broadcast = sum(1 for r in tier2_rows if r.get("channel_layout") == "broadcast")
    cpu_ok = [r for r in cpu_rows if r.get("backend") == "numpy_cpu" and r.get("status") == "ok"]
    gpu_ok = [r for r in cpu_rows if is_gpu_backend(r.get("backend", "")) and r.get("status") == "ok"]
    gpu_backends = sorted({r.get("backend", "") for r in gpu_ok if r.get("backend")})
    mode_pass = sum(1 for r in tier2_rows if r.get("result") == "PASS")
    row_shard_note = (
        "Row-sharded LLM trace update: implemented, smoke validated, and full `active_rows_rowshard32` summaries are now official."
        if hf_row_sharded and not hf_broadcast
        else "Row-sharded LLM trace update: implemented and smoke validated; final 32-channel-distributed LLM tables require `active_rows_rowshard32` full rerun when current summary rows are still broadcast."
    )

    llm_lines = []
    for r in hf_agg_rows:
        llm_lines.append(
            f"| {r['model']} | {r['projection_group']} | `{r['conceptual_shape']}` | {r['execution_rows']} | {r['mode2_memory_system_cycles_sum']} | {r['within_1_ulp']} | {r['max_ulp']} |"
        )

    report = f"""# Summer 2026 Experiment Completion Report

## Executive Summary

- Functional baseline: {len(functional_rows)} representative file-AIMD regression shapes completed with CPU BF16 ULP reports.
- Matmul timing regression: {matmul_pass}/{len(matmul_rows)} official 36-case shapes PASS for Mode1/Mode2 `memory_system_cycles` identity.
- Simulation Time Tier1/perf_comparison has been removed from the official result set.
- LLM actual weights: {len(hf_rows)} active projection rows in summary with HF safetensors values: {hf_row_sharded} row-sharded, {hf_broadcast} legacy broadcast.
- {row_shard_note}
- Tier2 exact timing: {mode_pass}/{len(tier2_rows)} Mode1/Mode2 rows have matching `memory_system_cycles`: {tier2_row_sharded} row-sharded, {tier2_broadcast} legacy broadcast.
- CPU/GPU/reference execution comparison: {len(execution_rows)} projection rows combine CPU, GPU, Mode1, and Mode2 timing; GPU backend is {", ".join(gpu_backends) if gpu_backends else "no available CUDA backend"}.

## LLM Actual-Weight Results

| Model | Projection | Conceptual shape | Chunk rows | Mode2 cycle sum | <=1 ULP | Max ULP |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(llm_lines)}

## Caveats

- Historical Tier1 raw files may remain under `simulation_time/tier1_perf_comparison`, but Tier1 is no longer part of the official summary.
- Legacy `active_rows` broadcast directories may remain on disk. Official row-sharded summaries use `active_rows_rowshard32` or `active_rows_rowshard32_chunk*`.
- CPU/GPU timing was recorded in the `summer26` conda environment. GPU timing is kernel-time only and remains separate from AiM simulated time.
"""
    (SUMMARY / "final_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    functional_rows = consolidate_functional()
    consolidate_hf_weight_functional()
    tier2_rows = consolidate_tier2()
    consolidate_reference_time(tier2_rows)
    copy_summary_inputs()
    write_stage_status(functional_rows, tier2_rows)
    write_final_report(functional_rows, tier2_rows)
    print(f"wrote {SUMMARY / 'stage_status.md'}")
    print(f"wrote {SUMMARY / 'final_experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
