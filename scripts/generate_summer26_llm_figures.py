#!/usr/bin/env python3
"""Generate Summer 2026 LLM-shape functional and runtime figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import struct
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "tests/result/result_summer26"
SUMMARY_DIR = RESULT_ROOT / "summary"
FIGURE_DIR = RESULT_ROOT / "figures"
FIGURE1A_PREACT_DIR = RESULT_ROOT / "llm_shape_pre_activation"

MODEL_ORDER = [
    "EXAONE-4.0-1.2B",
    "Phi-4-mini-Instruct",
    "Llama3.2-1B-Instruct",
    "Llama3.1-8B-Instruct",
    "Qwen3-8B",
]
EXCLUDED_FIGURE_MODELS = {"EXAONE-4.0-1.2B"}
FIGURE1C_MODEL_ORDER = [
    model for model in MODEL_ORDER if model not in EXCLUDED_FIGURE_MODELS
]
FIGURE3_MODEL_ORDER = [
    model for model in MODEL_ORDER if model not in EXCLUDED_FIGURE_MODELS
]
MODEL_NAME_ALIASES = {
    "Phi-4-mini-instruct": "Phi-4-mini-Instruct",
}
PROJECTION_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
FIGURE1A_PROJECTION_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]
PROJECTION_FULL_NAME = {
    "self_attn.q_proj": "Attention Query Projection",
    "self_attn.k_proj": "Attention Key Projection",
    "self_attn.v_proj": "Attention Value Projection",
    "self_attn.o_proj": "Attention Output Projection",
    "mlp.gate_proj": "MLP Gate Projection",
    "mlp.up_proj": "MLP Up Projection",
    "mlp.down_proj": "MLP Down Projection",
}
PROJECTION_MODULE_NAME = {
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.o_proj": "self_attn.o_proj",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.up_proj",
    "mlp.down_proj": "mlp.down_proj",
}
FIGURE1_SAMPLE_COUNT = 10000
FIGURE1A_OUTPUT_STAGE = "pre_activation"
FIGURE_FONT_SIZE = 18
FOCUS_FIGURE_FONT_SIZE = 20
FIGURE1A_FONT_SIZE = 40
FIGURE1B_FONT_SIZE = 60
FIGURE1B_TABLE_FONT_SIZE = 17
FIGURE1B_DATA_LABEL_FONT_SIZE = 60
FIGURE1B_HEATMAP_TEXT_FONT_SIZE = 55
FIGURE1B_ALLOW_BF16_REDUCED_PRECISION_REDUCTION = False
FIGURE1C_FONT_SIZE = 30
FIGURE1C_DATA_LABEL_FONT_SIZE = 25
FIGURE3_FONT_SIZE = 40
DATA_LABEL_FONT_SIZE = 12
MS_DATA_LABEL_FONT_SIZE = 18
FIGURE3_DATA_LABEL_FONT_SIZE = 30
FIGURE3A_Y_MAX_S = 40.0
DEFAULT_AIM_FREQUENCY_HZ = 1_000_000_000.0
DEFAULT_NONRESIDENT_BANDWIDTH_GBPS = 8.0
DEFAULT_PIM_IO_BANDWIDTH_GBPS = 32.0
HIGH_RES_PNG_DPI = 600
FIGURE1A_CPU_COLOR = "#005AB5"
FIGURE1A_GPU_COLOR = "#D55E00"
MONO_LINE_DARK = "#111111"
MONO_GRID = "#9a9a9a"
MONO_BAR_STYLES = [
    {"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
    {"facecolor": "#c8c8c8", "edgecolor": "black", "linewidth": 1.2},
    {"facecolor": "#303030", "edgecolor": "black", "linewidth": 1.2},
]
MONO_BAR_STYLES_2 = [
    {"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
    {"facecolor": "#303030", "edgecolor": "black", "linewidth": 1.2},
]
FIGURE2A_BAR_STYLES = [
    {"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
    {"facecolor": "#c8c8c8", "edgecolor": "black", "linewidth": 1.2},
    {"facecolor": "#303030", "edgecolor": "black", "linewidth": 1.2},
]
FIGURE3_BAR_STYLES = [
    {"facecolor": "white", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#303030", "edgecolor": "black", "linewidth": 2.0},
]
FIGURE1C_BAR_STYLES = [
    {"facecolor": "white", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#e6e6e6", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#c8c8c8", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#aaaaaa", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#7c7c7c", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#505050", "edgecolor": "black", "linewidth": 2.0},
    {"facecolor": "#252525", "edgecolor": "black", "linewidth": 2.0},
]


def configure_plot_font(plt: Any, font_size: int = FIGURE_FONT_SIZE) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "hatch.linewidth": 1.0,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "model" in row:
            row["model"] = MODEL_NAME_ALIASES.get(row["model"], row["model"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def bf16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_bits_to_float(bits: int) -> float:
    return float(bf16_to_float32(np.array([bits & 0xFFFF], dtype=np.uint16))[0])


def format_bf16_bits_exact(bits: int) -> str:
    return repr(bf16_bits_to_float(bits))


SUPERSCRIPT_TRANSLATION = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_decimal_power(value: float, significant_digits: int = 4) -> str:
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0.0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        exponent = int(math.floor(math.log10(magnitude)))
        mantissa = value / (10 ** exponent)
        precision = max(significant_digits - 1, 0)
        mantissa_text = f"{mantissa:.{precision}f}".rstrip("0").rstrip(".")
        if "." not in mantissa_text:
            mantissa_text += ".0"
        exponent_text = str(exponent).translate(SUPERSCRIPT_TRANSLATION)
        return f"{mantissa_text}×10{exponent_text}"
    decimal_text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in decimal_text:
        decimal_text += ".0"
    return decimal_text


def format_bf16_bits_display(bits: int) -> str:
    bits &= 0xFFFF
    return f"0x{bits:04X} = {format_decimal_power(bf16_bits_to_float(bits))}"


def bf16_ordered_int(bits: int) -> int:
    bits &= 0xFFFF
    if bits & 0x7FFF == 0:
        return 0x8000
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def bf16_ulp_distance_bits(left: int, right: int) -> int:
    return abs(bf16_ordered_int(left) - bf16_ordered_int(right))


def bf16_ordered_array(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.uint32)
    ordered = np.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    ordered = np.where((bits & 0x7FFF) == 0, 0x8000, ordered)
    return ordered.astype(np.int32)


def bf16_ulp_distance_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(bf16_ordered_array(left) - bf16_ordered_array(right))


def bf16_within_one_ulp_range(bits: int) -> tuple[float, float]:
    candidates = [bits & 0xFFFF]
    if bits > 0:
        candidates.append((bits - 1) & 0xFFFF)
    if bits < 0xFFFF:
        candidates.append((bits + 1) & 0xFFFF)
    values = [bf16_bits_to_float(candidate) for candidate in candidates]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.nan, math.nan
    return min(finite_values), max(finite_values)


def format_float_compact(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3 or magnitude >= 1e3:
        return f"{value:.2e}"
    if magnitude < 1:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def parse_fraction(value: str) -> tuple[int, int]:
    num, den = value.split("/")
    return int(num), int(den)


def fraction_rate(value: str) -> float:
    num, den = parse_fraction(value)
    return (num / den) if den else math.nan


def summarize(samples: list[float]) -> dict[str, str]:
    return {
        "mean_s": f"{statistics.mean(samples):.9f}",
        "median_s": f"{statistics.median(samples):.9f}",
        "stddev_s": f"{statistics.stdev(samples) if len(samples) > 1 else 0.0:.9f}",
        "min_s": f"{min(samples):.9f}",
        "max_s": f"{max(samples):.9f}",
    }


def case_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["model_source"], row["projection_group"]


def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    model = str(row["model"])
    group = str(row["projection_group"])
    model_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    group_idx = PROJECTION_ORDER.index(group) if group in PROJECTION_ORDER else len(PROJECTION_ORDER)
    return model_idx, group_idx, model


def load_cpu_reference(run_dir: Path) -> np.ndarray:
    refs = []
    with (run_dir / "cpu_reference.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(int(row["cpu_hex"], 16))
    return np.array(refs, dtype=np.uint16)


def load_rtl_output(run_dir: Path) -> np.ndarray:
    values = []
    with (run_dir / "functional_report.csv").open(newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            values.append(int(row["rtl_hex"], 16))
    return np.array(values, dtype=np.uint16)


def output_location(row: int, num_channels: int = 32, num_banks: int = 16) -> tuple[int, int, int, int]:
    group = row // (num_channels * num_banks)
    rem = row % (num_channels * num_banks)
    channel = rem // num_banks
    bank = rem % num_banks
    physical_row = group * num_channels + channel
    return group, channel, bank, physical_row


def load_aimd_weight_activation(run_dir: Path, q: int, r: int, channel_layout: str = "") -> tuple[np.ndarray, np.ndarray]:
    aimd_files = sorted(run_dir.glob("p1_q*_r*.aimd"))
    if not aimd_files:
        raise FileNotFoundError(f"AIMD file not found in {run_dir}")
    path = aimd_files[0]
    with path.open("rb") as f:
        magic, num_banks, rows_per_bank, tiles_per_row = struct.unpack("<IIII", f.read(16))
    if magic != 0x41494D44:
        raise ValueError(f"Bad AIMD magic in {path}")
    row_elems = tiles_per_row * 16
    if q > row_elems:
        raise ValueError(f"Q={q} exceeds row capacity {row_elems} in {path}")

    if channel_layout == "row_sharded" or r > rows_per_bank:
        weight_map = np.memmap(
            path,
            mode="r",
            dtype=np.uint16,
            offset=16,
            shape=(num_banks, rows_per_bank, row_elems),
        )
        weight = np.zeros((r, q), dtype=np.uint16)
        for output_row in range(r):
            _, _, bank, physical_row = output_location(output_row, num_banks=num_banks)
            if physical_row >= rows_per_bank:
                raise ValueError(
                    f"output_row={output_row} maps to physical_row={physical_row}, "
                    f"but rows_per_bank={rows_per_bank} in {path}"
                )
            weight[output_row, :] = weight_map[bank, physical_row, :q]
    else:
        weight_map = np.memmap(path, mode="r", dtype=np.uint16, offset=16, shape=(rows_per_bank, row_elems))
        weight = np.array(weight_map[:r, :q], dtype=np.uint16, copy=True)
    del weight_map

    activation_offset = 16 + num_banks * rows_per_bank * row_elems * 2
    activation_map = np.memmap(path, mode="r", dtype=np.uint16, offset=activation_offset, shape=(row_elems,))
    activation = np.array(activation_map[:q], dtype=np.uint16, copy=True)
    del activation_map
    return weight, activation


def compare_bits(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    diff = bf16_ulp_distance_array(candidate, reference)
    return {
        "exact_num": int(np.sum(diff == 0)),
        "within_1_num": int(np.sum(diff <= 1)),
        "within_2_num": int(np.sum(diff <= 2)),
        "den": int(diff.size),
        "max_ulp": int(diff.max()) if diff.size else 0,
        "avg_ulp": float(diff.mean()) if diff.size else 0.0,
    }


def torch_bf16_functional(
    weight_bf16: np.ndarray,
    activation_bf16: np.ndarray,
    apply_relu: bool = True,
    allow_bf16_reduced_precision_reduction: bool | None = None,
) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is not available")

    old_reduced_precision = None
    if allow_bf16_reduced_precision_reduction is not None:
        old_reduced_precision = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16_reduced_precision_reduction
    try:
        w = bf16_to_float32(weight_bf16.reshape(-1)).reshape(weight_bf16.shape)
        a = bf16_to_float32(activation_bf16)
        with torch.no_grad():
            wt = torch.from_numpy(w).to(device="cuda", dtype=torch.bfloat16)
            at = torch.from_numpy(a).to(device="cuda", dtype=torch.bfloat16)
            out = torch.matmul(wt, at)
            if apply_relu:
                out = torch.relu(out)
            out = out.to(torch.bfloat16)
            torch.cuda.synchronize()
            out_cpu = out.cpu().view(torch.int16).numpy().astype(np.uint16, copy=False).copy()
            del wt, at, out
            torch.cuda.empty_cache()
        return out_cpu
    finally:
        if old_reduced_precision is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced_precision


def cupy_cuda_benchmark(weight_bf16: np.ndarray, activation_bf16: np.ndarray, warmup: int, repeats: int) -> tuple[str, dict[str, str]]:
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(f"cupy import failed: {exc}") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy CUDA device is not available")

    w = bf16_to_float32(weight_bf16.reshape(-1)).reshape(weight_bf16.shape)
    a = bf16_to_float32(activation_bf16)
    wg = cp.asarray(w, dtype=cp.float32)
    ag = cp.asarray(a, dtype=cp.float32)
    for _ in range(warmup):
        _ = cp.maximum(wg @ ag, 0)
    cp.cuda.Stream.null.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        _ = cp.maximum(wg @ ag, 0)
        end_event.record()
        end_event.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
    del wg, ag
    cp.get_default_memory_pool().free_all_blocks()
    return "cupy_cuda_float32_kernel", summarize(samples)


def generate_supplemental_rows(functional_rows: list[dict[str, str]], force: bool, warmup: int,
                               repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch_path = FIGURE_DIR / "supplemental_torch_functional_summary.csv"
    cupy_path = FIGURE_DIR / "supplemental_cupy_cuda_summary.csv"
    if torch_path.exists() and cupy_path.exists() and not force:
        cached_torch = read_csv(torch_path)
        cached_cupy = read_csv(cupy_path)
        if cached_torch and "aimrtl_vs_torch_within_1_ulp" in cached_torch[0]:
            return cached_torch, cached_cupy
    else:
        cached_cupy = []
    cached_cupy_by_run = {row["run_id"]: row for row in cached_cupy}

    torch_rows: list[dict[str, Any]] = []
    cupy_rows: list[dict[str, Any]] = []
    total = len(functional_rows)
    for idx, row in enumerate(functional_rows, start=1):
        run_dir = ROOT / row["result_dir"]
        q = int(row["q"])
        r = int(row["r"])
        print(f"[{idx:02d}/{total}] {row['model']} {row['projection_group']} {q}x{r}")
        weight_bf16, activation_bf16 = load_aimd_weight_activation(
            run_dir,
            q,
            r,
            row.get("channel_layout", ""),
        )
        cpu_ref = load_cpu_reference(run_dir)
        rtl_bits = load_rtl_output(run_dir)

        torch_row: dict[str, Any] = {
            "run_id": row["run_id"],
            "model": row["model"],
            "model_source": row["model_source"],
            "projection_group": row["projection_group"],
            "tensor_name": row["tensor_name"],
            "shape": row["shape"],
            "backend": "torch_cuda_bf16_functional",
            "status": "ok",
        }
        try:
            torch_bits = torch_bf16_functional(weight_bf16, activation_bf16)
            torch_cpu_stats = compare_bits(torch_bits, cpu_ref)
            aimrtl_torch_stats = compare_bits(rtl_bits, torch_bits)
            torch_row.update(
                {
                    "exact": f"{torch_cpu_stats['exact_num']}/{torch_cpu_stats['den']}",
                    "within_1_ulp": f"{torch_cpu_stats['within_1_num']}/{torch_cpu_stats['den']}",
                    "within_2_ulp": f"{torch_cpu_stats['within_2_num']}/{torch_cpu_stats['den']}",
                    "max_ulp": torch_cpu_stats["max_ulp"],
                    "avg_ulp": f"{torch_cpu_stats['avg_ulp']:.6f}",
                    "aimrtl_vs_torch_exact": f"{aimrtl_torch_stats['exact_num']}/{aimrtl_torch_stats['den']}",
                    "aimrtl_vs_torch_within_1_ulp": (
                        f"{aimrtl_torch_stats['within_1_num']}/{aimrtl_torch_stats['den']}"
                    ),
                    "aimrtl_vs_torch_within_2_ulp": (
                        f"{aimrtl_torch_stats['within_2_num']}/{aimrtl_torch_stats['den']}"
                    ),
                    "aimrtl_vs_torch_max_ulp": aimrtl_torch_stats["max_ulp"],
                    "aimrtl_vs_torch_avg_ulp": f"{aimrtl_torch_stats['avg_ulp']:.6f}",
                }
            )
        except Exception as exc:
            torch_row.update(
                {
                    "status": f"unavailable_{exc.__class__.__name__}",
                    "exact": "",
                    "within_1_ulp": "",
                    "within_2_ulp": "",
                    "max_ulp": "",
                    "avg_ulp": "",
                    "aimrtl_vs_torch_exact": "",
                    "aimrtl_vs_torch_within_1_ulp": "",
                    "aimrtl_vs_torch_within_2_ulp": "",
                    "aimrtl_vs_torch_max_ulp": "",
                    "aimrtl_vs_torch_avg_ulp": "",
                }
            )
        torch_rows.append(torch_row)

        if not force and row["run_id"] in cached_cupy_by_run:
            cupy_rows.append(cached_cupy_by_run[row["run_id"]])
            continue
        cupy_row: dict[str, Any] = {
            "run_id": row["run_id"],
            "model": row["model"],
            "model_source": row["model_source"],
            "projection_group": row["projection_group"],
            "tensor_name": row["tensor_name"],
            "shape": row["shape"],
            "backend": "cupy_cuda_float32_kernel",
            "status": "ok",
            "warmup": warmup,
            "repeats": repeats,
        }
        try:
            backend, stats = cupy_cuda_benchmark(weight_bf16, activation_bf16, warmup, repeats)
            cupy_row["backend"] = backend
            cupy_row.update(stats)
        except Exception as exc:
            cupy_row.update(
                {
                    "status": f"unavailable_{exc.__class__.__name__}",
                    "mean_s": "",
                    "median_s": "",
                    "stddev_s": "",
                    "min_s": "",
                    "max_s": "",
                }
            )
        cupy_rows.append(cupy_row)

    write_csv(
        torch_path,
        torch_rows,
        [
            "run_id",
            "model",
            "model_source",
            "projection_group",
            "tensor_name",
            "shape",
            "backend",
            "status",
            "exact",
            "within_1_ulp",
            "within_2_ulp",
            "max_ulp",
            "avg_ulp",
            "aimrtl_vs_torch_exact",
            "aimrtl_vs_torch_within_1_ulp",
            "aimrtl_vs_torch_within_2_ulp",
            "aimrtl_vs_torch_max_ulp",
            "aimrtl_vs_torch_avg_ulp",
        ],
    )
    write_csv(
        cupy_path,
        cupy_rows,
        [
            "run_id",
            "model",
            "model_source",
            "projection_group",
            "tensor_name",
            "shape",
            "backend",
            "status",
            "warmup",
            "repeats",
            "mean_s",
            "median_s",
            "stddev_s",
            "min_s",
            "max_s",
        ],
    )
    return torch_rows, cupy_rows


def aggregate_torch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = case_key(row)
        item = grouped.setdefault(
            key,
            {
                "model": row["model"],
                "model_source": row["model_source"],
                "projection_group": row["projection_group"],
                "execution_rows": 0,
                "status": "ok",
                "exact_num": 0,
                "within_1_num": 0,
                "within_2_num": 0,
                "den": 0,
                "max_ulp": 0,
                "avg_weighted_sum": 0.0,
                "aimrtl_vs_torch_exact_num": 0,
                "aimrtl_vs_torch_within_1_num": 0,
                "aimrtl_vs_torch_within_2_num": 0,
                "aimrtl_vs_torch_den": 0,
                "aimrtl_vs_torch_max_ulp": 0,
                "aimrtl_vs_torch_avg_weighted_sum": 0.0,
            },
        )
        item["execution_rows"] += 1
        if (
            row.get("status") != "ok"
            or not row.get("within_1_ulp")
            or not row.get("aimrtl_vs_torch_within_1_ulp")
        ):
            item["status"] = "missing"
            continue
        exact_num, den = parse_fraction(str(row["exact"]))
        within_1_num, _ = parse_fraction(str(row["within_1_ulp"]))
        within_2_num, _ = parse_fraction(str(row["within_2_ulp"]))
        aimtorch_exact_num, aimtorch_den = parse_fraction(str(row["aimrtl_vs_torch_exact"]))
        aimtorch_within_1_num, _ = parse_fraction(str(row["aimrtl_vs_torch_within_1_ulp"]))
        aimtorch_within_2_num, _ = parse_fraction(str(row["aimrtl_vs_torch_within_2_ulp"]))
        item["exact_num"] += exact_num
        item["within_1_num"] += within_1_num
        item["within_2_num"] += within_2_num
        item["den"] += den
        item["max_ulp"] = max(int(item["max_ulp"]), int(row["max_ulp"]))
        item["avg_weighted_sum"] += float(row["avg_ulp"]) * den
        item["aimrtl_vs_torch_exact_num"] += aimtorch_exact_num
        item["aimrtl_vs_torch_within_1_num"] += aimtorch_within_1_num
        item["aimrtl_vs_torch_within_2_num"] += aimtorch_within_2_num
        item["aimrtl_vs_torch_den"] += aimtorch_den
        item["aimrtl_vs_torch_max_ulp"] = max(
            int(item["aimrtl_vs_torch_max_ulp"]),
            int(row["aimrtl_vs_torch_max_ulp"]),
        )
        item["aimrtl_vs_torch_avg_weighted_sum"] += float(row["aimrtl_vs_torch_avg_ulp"]) * aimtorch_den

    output = []
    for item in grouped.values():
        den = int(item["den"])
        avg = (float(item["avg_weighted_sum"]) / den) if den else math.nan
        aimtorch_den = int(item["aimrtl_vs_torch_den"])
        aimtorch_avg = (
            float(item["aimrtl_vs_torch_avg_weighted_sum"]) / aimtorch_den
            if aimtorch_den
            else math.nan
        )
        output.append(
            {
                "model": item["model"],
                "model_source": item["model_source"],
                "projection_group": item["projection_group"],
                "execution_rows": item["execution_rows"],
                "status": item["status"],
                "exact": f"{item['exact_num']}/{den}" if den else "",
                "within_1_ulp": f"{item['within_1_num']}/{den}" if den else "",
                "within_2_ulp": f"{item['within_2_num']}/{den}" if den else "",
                "max_ulp": item["max_ulp"] if den else "",
                "avg_ulp_weighted": f"{avg:.6f}" if den else "",
                "aimrtl_vs_torch_exact": (
                    f"{item['aimrtl_vs_torch_exact_num']}/{aimtorch_den}" if aimtorch_den else ""
                ),
                "aimrtl_vs_torch_within_1_ulp": (
                    f"{item['aimrtl_vs_torch_within_1_num']}/{aimtorch_den}" if aimtorch_den else ""
                ),
                "aimrtl_vs_torch_within_2_ulp": (
                    f"{item['aimrtl_vs_torch_within_2_num']}/{aimtorch_den}" if aimtorch_den else ""
                ),
                "aimrtl_vs_torch_max_ulp": item["aimrtl_vs_torch_max_ulp"] if aimtorch_den else "",
                "aimrtl_vs_torch_avg_ulp_weighted": f"{aimtorch_avg:.6f}" if aimtorch_den else "",
            }
        )
    return sorted(output, key=sort_key)


def aggregate_cupy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = case_key(row)
        item = grouped.setdefault(
            key,
            {
                "model": row["model"],
                "model_source": row["model_source"],
                "projection_group": row["projection_group"],
                "execution_rows": 0,
                "status": "ok",
                "mean_s_sum": 0.0,
                "median_s_sum": 0.0,
                "min_s_sum": 0.0,
                "max_s_sum": 0.0,
            },
        )
        item["execution_rows"] += 1
        if row.get("status") != "ok" or not row.get("mean_s"):
            item["status"] = "missing"
            continue
        item["mean_s_sum"] += float(row["mean_s"])
        item["median_s_sum"] += float(row["median_s"])
        item["min_s_sum"] += float(row["min_s"])
        item["max_s_sum"] += float(row["max_s"])

    output = []
    for item in grouped.values():
        output.append(
            {
                "model": item["model"],
                "model_source": item["model_source"],
                "projection_group": item["projection_group"],
                "execution_rows": item["execution_rows"],
                "status": item["status"],
                "cupy_cuda_mean_s_sum": f"{item['mean_s_sum']:.9f}",
                "cupy_cuda_median_s_sum": f"{item['median_s_sum']:.9f}",
                "cupy_cuda_min_s_sum": f"{item['min_s_sum']:.9f}",
                "cupy_cuda_max_s_sum": f"{item['max_s_sum']:.9f}",
            }
        )
    return sorted(output, key=sort_key)


def expected_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["model"], row["projection_group"]) for row in rows}


def validate_coverage(functional_agg: list[dict[str, str]], execution_agg: list[dict[str, str]],
                      torch_agg: list[dict[str, Any]], cupy_agg: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    messages = []
    ok = True
    expected = {(model, group) for model in MODEL_ORDER for group in PROJECTION_ORDER}
    datasets = [
        ("functional aggregate", expected_keys(functional_agg)),
        ("execution aggregate", expected_keys(execution_agg)),
        ("torch BF16 functional aggregate", expected_keys(torch_agg)),
        ("supplemental CuPy CUDA aggregate", expected_keys(cupy_agg)),
    ]
    for name, keys in datasets:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing:
            ok = False
            messages.append(f"- {name}: missing {len(missing)} cases: {missing}")
        else:
            messages.append(f"- {name}: complete {len(expected)}/{len(expected)} cases")
        if extra:
            messages.append(f"- {name}: extra cases ignored by plotting: {extra}")

    figure_execution_agg = [
        row
        for row in execution_agg
        if row.get("model") in MODEL_ORDER and row.get("projection_group") in PROJECTION_ORDER
    ]
    mode_pass = sum(1 for row in figure_execution_agg if row.get("mode_cycle_result") == "PASS")
    if mode_pass != len(figure_execution_agg):
        ok = False
        messages.append(
            f"- Mode1/Mode2 cycle match: {mode_pass}/{len(figure_execution_agg)} PASS"
        )
    else:
        messages.append(f"- Mode1/Mode2 cycle match: {mode_pass}/{len(figure_execution_agg)} PASS")

    figure_cupy_agg = [
        row
        for row in cupy_agg
        if row.get("model") in MODEL_ORDER and row.get("projection_group") in PROJECTION_ORDER
    ]
    figure_torch_agg = [
        row
        for row in torch_agg
        if row.get("model") in MODEL_ORDER and row.get("projection_group") in PROJECTION_ORDER
    ]
    cupy_ok = sum(1 for row in figure_cupy_agg if row.get("status") == "ok")
    torch_ok = sum(1 for row in figure_torch_agg if row.get("status") == "ok")
    if cupy_ok != len(figure_cupy_agg):
        ok = False
    if torch_ok != len(figure_torch_agg):
        ok = False
    messages.append(f"- torch BF16 functional aggregate status: {torch_ok}/{len(figure_torch_agg)} ok")
    messages.append(f"- supplemental CuPy CUDA aggregate status: {cupy_ok}/{len(figure_cupy_agg)} ok")
    return ok, messages


def make_label(row: dict[str, Any]) -> str:
    model = str(row["model"])
    return f"{model}\n{row['projection_group']}"


def make_full_projection_label(row: dict[str, Any]) -> str:
    model = str(row["model"])
    group = str(row["projection_group"])
    return f"{model}\n{PROJECTION_FULL_NAME.get(group, group)}"


def make_module_projection_label(row: dict[str, Any]) -> str:
    model = str(row["model"])
    group = str(row["projection_group"])
    return f"{model}\n{PROJECTION_MODULE_NAME.get(group, group)}"


def figure1a_run_dir(row: dict[str, str]) -> Path:
    if FIGURE1A_OUTPUT_STAGE == "post_activation":
        return ROOT / row["result_dir"]
    if FIGURE1A_OUTPUT_STAGE == "pre_activation":
        return FIGURE1A_PREACT_DIR / f"{Path(row['result_dir']).name}_pre_activation"
    raise ValueError(f"Unsupported Figure 1a output stage: {FIGURE1A_OUTPUT_STAGE}")


def row_start_from_tensor_name(tensor_name: str) -> int:
    match = re.search(r":rows\[(\d+):\d+\]", tensor_name)
    return int(match.group(1)) if match else 0


def parse_shape(shape: str) -> tuple[int, int]:
    q, r = shape.split("x")
    return int(q), int(r)


def transfer_time_s(num_bytes: int, bandwidth_gbps: float) -> float:
    if bandwidth_gbps <= 0:
        raise ValueError(f"bandwidth_gbps must be positive, got {bandwidth_gbps}")
    return num_bytes / (bandwidth_gbps * 1_000_000_000.0)


def compact_decimal(value: float) -> str:
    if value < 10:
        text = f"{value:.2f}"
    elif value < 100:
        text = f"{value:.1f}"
    else:
        text = f"{value:.0f}"
    return text.rstrip("0").rstrip(".")


def format_time_label(seconds: float) -> str:
    magnitude = abs(seconds)
    if magnitude < 1e-6:
        return f"{compact_decimal(seconds * 1e9)}ns"
    if magnitude < 1e-3:
        return f"{compact_decimal(seconds * 1e6)}us"
    if magnitude < 1:
        return f"{compact_decimal(seconds * 1e3)}ms"
    return f"{compact_decimal(seconds)}s"


def format_seconds_label(seconds: float) -> str:
    magnitude = abs(seconds)
    if magnitude < 1:
        return f"{seconds:.3f}s"
    if magnitude < 10:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def format_ms_label(milliseconds: float) -> str:
    return f"{compact_decimal(milliseconds)}ms"


def add_time_bar_labels(ax: Any, bars: Any, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            format_time_label(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=DATA_LABEL_FONT_SIZE,
            clip_on=False,
        )


def add_ms_bar_labels(
    ax: Any,
    bars: Any,
    values_ms: list[float],
    font_size: int = MS_DATA_LABEL_FONT_SIZE,
) -> None:
    for bar, value in zip(bars, values_ms):
        ax.annotate(
            format_ms_label(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=font_size,
            clip_on=False,
        )


def set_log_ylim_for_labels(ax: Any, values: list[float], upper_factor: float = 3.2) -> None:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return
    ax.set_ylim(min(positive_values) * 0.55, max(positive_values) * upper_factor)


def set_linear_ylim_for_labels(ax: Any, values: list[float], upper_factor: float = 1.28) -> None:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        ax.set_ylim(0, 1)
        return
    ax.set_ylim(0, max(positive_values) * upper_factor)


def format_axis_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return compact_decimal(value)


def plot_functional(functional_agg: list[dict[str, str]], torch_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_plot_font(plt)

    official_by_key = {case_key(row): row for row in functional_agg}
    torch_by_key = {case_key(row): row for row in torch_agg}
    plot_rows = []
    for key in sorted(official_by_key, key=lambda k: sort_key({"model": k[0], "projection_group": k[2]})):
        if key[0] not in MODEL_ORDER or key[2] not in PROJECTION_ORDER:
            continue
        official = official_by_key[key]
        torch_row = torch_by_key[key]
        plot_rows.append(
            {
                "model": key[0],
                "model_source": key[1],
                "projection_group": key[2],
                "conceptual_shape": official.get("conceptual_shape", ""),
                "aimrtl_mode2_vs_cpu_bf16_within_1_ulp_pct": (
                    100.0 * fraction_rate(official["within_1_ulp"])
                ),
                "aimrtl_mode2_vs_gpu_torch_bf16_within_1_ulp_pct": (
                    100.0 * fraction_rate(torch_row["aimrtl_vs_torch_within_1_ulp"])
                ),
                "aimrtl_mode2_vs_cpu_bf16_max_ulp": official.get("max_ulp", ""),
                "aimrtl_mode2_vs_gpu_torch_bf16_max_ulp": torch_row.get("aimrtl_vs_torch_max_ulp", ""),
            }
        )

    labels = [make_full_projection_label(row) for row in plot_rows]
    x = np.arange(len(plot_rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(24, len(labels) * 1.15), 9.0))
    ax.bar(
        x - width / 2,
        [row["aimrtl_mode2_vs_cpu_bf16_within_1_ulp_pct"] for row in plot_rows],
        width,
        label="AiMRTL Mode2 vs CPU BF16",
    )
    ax.bar(
        x + width / 2,
        [row["aimrtl_mode2_vs_gpu_torch_bf16_within_1_ulp_pct"] for row in plot_rows],
        width,
        label="AiMRTL Mode2 vs GPU torch BF16",
    )
    ax.set_ylabel("<= 1 ULP match rate (%)", fontsize=FIGURE_FONT_SIZE)
    ax.set_title("Figure 1. Actual LLM Shape Functional Validation", fontsize=FIGURE_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=FIGURE_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncols=2, fontsize=FIGURE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_llm_shape_functional_validation.png", dpi=220)
    fig.savefig(FIGURE_DIR / "figure1_llm_shape_functional_validation.svg")
    plt.close(fig)
    return plot_rows


def plot_llama_projection_element_ulp(functional_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font_size = FIGURE1A_FONT_SIZE
    configure_plot_font(plt, font_size)

    plot_rows: list[dict[str, Any]] = []
    panel_rows: dict[str, list[dict[str, Any]]] = {}
    for panel_idx, group in enumerate(FIGURE1A_PROJECTION_ORDER):
        target_rows = [
            row
            for row in functional_rows
            if row["model"] == "Llama3.1-8B-Instruct" and row["projection_group"] == group
        ]
        target_rows = sorted(target_rows, key=lambda row: row_start_from_tensor_name(row["tensor_name"]))
        if not target_rows:
            raise RuntimeError(f"No Llama3.1-8B-Instruct {group} rows found for Figure 1a")

        panel = chr(ord("a") + panel_idx)
        panel_rows[group] = []
        for row in target_rows:
            run_dir = figure1a_run_dir(row)
            if not run_dir.exists():
                raise RuntimeError(
                    f"Figure 1a {FIGURE1A_OUTPUT_STAGE} run is missing: {run_dir}. "
                    "Regenerate the Llama3.1 Figure 1a runs with "
                    "`scripts/run_hf_weight_projection.py --output-stage pre_activation`."
                )
            q = int(row["q"])
            r = int(row["r"])
            row_start = row_start_from_tensor_name(row["tensor_name"])
            weight_bf16, activation_bf16 = load_aimd_weight_activation(
                run_dir,
                q,
                r,
                row.get("channel_layout", ""),
            )
            rtl_bits = load_rtl_output(run_dir)
            cpu_bits = load_cpu_reference(run_dir)
            torch_bits = torch_bf16_functional(
                weight_bf16,
                activation_bf16,
                apply_relu=(FIGURE1A_OUTPUT_STAGE == "post_activation"),
            )
            if not (len(rtl_bits) == len(cpu_bits) == len(torch_bits) == r):
                raise RuntimeError(
                    f"Length mismatch for {row['run_id']}: "
                    f"rtl={len(rtl_bits)} cpu={len(cpu_bits)} torch={len(torch_bits)} r={r}"
                )
            for local_idx in range(r):
                element = row_start + local_idx
                rtl = int(rtl_bits[local_idx])
                cpu = int(cpu_bits[local_idx])
                gpu = int(torch_bits[local_idx])
                item = {
                    "panel": panel,
                    "projection_group": group,
                    "element": element,
                    "local_element": local_idx,
                    "chunk_start": row_start,
                    "run_id": row["run_id"],
                    "rtl_hex": f"0x{rtl:04X}",
                    "cpu_bf16_hex": f"0x{cpu:04X}",
                    "gpu_torch_bf16_hex": f"0x{gpu:04X}",
                    "rtl_bf16_display": format_bf16_bits_display(rtl),
                    "cpu_bf16_display": format_bf16_bits_display(cpu),
                    "gpu_torch_bf16_display": format_bf16_bits_display(gpu),
                    "rtl_bf16_value_exact": format_bf16_bits_exact(rtl),
                    "cpu_bf16_value_exact": format_bf16_bits_exact(cpu),
                    "gpu_torch_bf16_value_exact": format_bf16_bits_exact(gpu),
                    "aimrtl_mode2_vs_cpu_bf16_ulp": bf16_ulp_distance_bits(rtl, cpu),
                    "aimrtl_mode2_vs_gpu_torch_bf16_ulp": bf16_ulp_distance_bits(rtl, gpu),
                }
                panel_rows[group].append(item)
                plot_rows.append(item)

    fig, axes = plt.subplots(2, 2, figsize=(48, 32), sharey=False)
    axes_flat = axes.flatten()
    for panel_idx, group in enumerate(FIGURE1A_PROJECTION_ORDER):
        ax = axes_flat[panel_idx]
        rows = panel_rows[group]
        x = np.array([int(row["element"]) for row in rows])
        y_cpu = np.array([int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) for row in rows])
        y_gpu = np.array([int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) for row in rows])
        gpu_boundary = np.array(
            [
                int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) > 1
                and is_gpu_relu_boundary_flip(row)
                for row in rows
            ],
            dtype=bool,
        )
        y_gpu_plot = y_gpu.astype(float)
        if FIGURE1A_OUTPUT_STAGE == "post_activation":
            y_gpu_plot[gpu_boundary] = np.nan
        else:
            gpu_boundary[:] = False
        gpu_non_boundary = y_gpu[~gpu_boundary]
        cpu_gt1 = int(np.sum(y_cpu > 1))
        cpu_gt1_pct = 100.0 * cpu_gt1 / len(rows)
        gpu_gt1_excluding_boundary = int(np.sum((y_gpu > 1) & ~gpu_boundary))
        gpu_gt1_excluding_boundary_pct = 100.0 * gpu_gt1_excluding_boundary / len(rows)
        panel_max_ulp = int(max(y_cpu.max(initial=0), gpu_non_boundary.max(initial=0)))
        panel = chr(ord("a") + panel_idx)
        ax.plot(
            x,
            y_cpu,
            color=FIGURE1A_CPU_COLOR,
            linewidth=1.25,
            label="AiMRTL Mode2 vs CPU BF16",
        )
        ax.plot(
            x,
            y_gpu_plot,
            color=FIGURE1A_GPU_COLOR,
            linewidth=1.25,
            label="AiMRTL Mode2 vs GPU torch BF16",
        )
        ax.axhline(1, color="#b8b8b8", linewidth=1.0)
        model_name = "Llama3.1-8B-Instruct"
        projection_name = PROJECTION_MODULE_NAME.get(group, group)
        gpu_gt1_label = (
            "GPU >1 ULP excl. boundary"
            if FIGURE1A_OUTPUT_STAGE == "post_activation"
            else "GPU >1 ULP"
        )
        stage_label = (
            "pre-activation GEMV"
            if FIGURE1A_OUTPUT_STAGE == "pre_activation"
            else "GEMV + ReLU"
        )
        ax.set_title(
            f"({panel}) {model_name}\n"
            f"{projection_name} {stage_label} (n={len(rows):,}, max ULP={panel_max_ulp:,})\n"
            f"CPU >1 ULP: {cpu_gt1_pct:.3f}% | "
            f"{gpu_gt1_label}: {gpu_gt1_excluding_boundary_pct:.3f}%",
            fontsize=font_size,
            linespacing=1.0,
        )
        ax.set_xlim(-100, max(int(x.max(initial=0)) + 100, 100))
        ax.set_ylim(0, max(panel_max_ulp + 1, 3))
        ax.tick_params(axis="both", labelsize=font_size)
        ax.grid(axis="both", color=MONO_GRID, alpha=0.22, linewidth=0.8)

    for ax in axes[1, :]:
        ax.set_xlabel("Output element index", fontsize=font_size)
    for ax in axes[:, 0]:
        ax.set_ylabel("ULP difference", fontsize=font_size)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[1].legend(
        handles,
        labels,
        loc="upper right",
        ncols=1,
        fontsize=font_size,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.28, wspace=0.26)
    fig.savefig(FIGURE_DIR / "figure1a_llama31_projection_element_ulp.png", dpi=220)
    fig.savefig(FIGURE_DIR / "figure1a_llama31_projection_element_ulp.svg")
    plt.close(fig)
    return plot_rows


def is_gpu_relu_boundary_flip(row: dict[str, Any]) -> bool:
    """GPU ReLU produced zero while CPU/RTL remained positive after ReLU."""
    rtl = int(str(row["rtl_hex"]), 16)
    cpu = int(str(row["cpu_bf16_hex"]), 16)
    gpu = int(str(row["gpu_torch_bf16_hex"]), 16)
    return gpu == 0 and rtl != 0 and cpu != 0


def summarize_figure1a_ulp(plot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for group in FIGURE1A_PROJECTION_ORDER:
        rows = [row for row in plot_rows if row["projection_group"] == group]
        if not rows:
            continue
        n = len(rows)
        cpu_gt1 = sum(1 for row in rows if int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) > 1)
        gpu_boundary_ignored = (
            sum(
                1
                for row in rows
                if int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) > 1 and is_gpu_relu_boundary_flip(row)
            )
            if FIGURE1A_OUTPUT_STAGE == "post_activation"
            else 0
        )
        gpu_gt1_excluding_boundary = sum(
            1
            for row in rows
            if int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) > 1
            and not (FIGURE1A_OUTPUT_STAGE == "post_activation" and is_gpu_relu_boundary_flip(row))
        )
        gpu_rows_excluding_boundary = [
            row
            for row in rows
            if not (FIGURE1A_OUTPUT_STAGE == "post_activation" and is_gpu_relu_boundary_flip(row))
        ]
        gpu_max_excluding_boundary = max(
            (int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) for row in gpu_rows_excluding_boundary),
            default=0,
        )
        summary_rows.append(
            {
                "projection_group": group,
                "total_elements": n,
                "cpu_gt1_ulp_elements": cpu_gt1,
                "cpu_gt1_ulp_pct": f"{100.0 * cpu_gt1 / n:.6f}",
                "gpu_relu_boundary_flip_ignored": gpu_boundary_ignored,
                "gpu_gt1_ulp_elements": gpu_gt1_excluding_boundary,
                "gpu_gt1_ulp_pct": f"{100.0 * gpu_gt1_excluding_boundary / n:.6f}",
                "gpu_gt1_ulp_elements_excluding_relu_boundary_flip": gpu_gt1_excluding_boundary,
                "gpu_gt1_ulp_pct_excluding_relu_boundary_flip": (
                    f"{100.0 * gpu_gt1_excluding_boundary / n:.6f}"
                ),
                "cpu_max_ulp": max(int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) for row in rows),
                "gpu_max_ulp": gpu_max_excluding_boundary,
                "gpu_max_ulp_excluding_relu_boundary_flip": gpu_max_excluding_boundary,
            }
        )
    return summary_rows


def write_figure1a_summary_markdown(summary_rows: list[dict[str, Any]]) -> None:
    if FIGURE1A_OUTPUT_STAGE == "pre_activation":
        intro = (
            "Figure 1a compares pre-activation GEMV outputs. The AiMRTL path reads "
            "`RDMAC16`, while CPU and GPU references omit ReLU before BF16 comparison."
        )
        gpu_gt1_header = "GPU >1 ULP"
        gpu_gt1_pct_header = "GPU >1 ULP (%)"
        gpu_max_header = "GPU max ULP"
    else:
        intro = (
            "GPU ReLU-boundary flips are excluded from the GPU `>1 ULP` mismatch percentage. "
            "A boundary flip means GPU torch BF16 produced `0x0000` after ReLU while both "
            "AiMRTL and CPU BF16 produced nonzero positive values."
        )
        gpu_gt1_header = "GPU >1 ULP excl. boundary"
        gpu_gt1_pct_header = "GPU >1 ULP excl. boundary (%)"
        gpu_max_header = "GPU max ULP excl. boundary"
    lines = [
        "# Figure 1a Llama3.1 Projection ULP Summary",
        "",
        intro,
        "",
    ]
    if FIGURE1A_OUTPUT_STAGE == "pre_activation":
        lines.extend(
            [
                f"| Projection | Elements | CPU >1 ULP | CPU >1 ULP (%) | {gpu_gt1_header} | "
                f"{gpu_gt1_pct_header} | CPU max ULP | {gpu_max_header} |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                f"| Projection | Elements | CPU >1 ULP | CPU >1 ULP (%) | {gpu_gt1_header} | "
                f"{gpu_gt1_pct_header} | GPU boundary flips ignored | CPU max ULP | {gpu_max_header} |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for row in summary_rows:
        display_row = {
            **row,
            "projection_group": PROJECTION_MODULE_NAME.get(
                str(row["projection_group"]),
                str(row["projection_group"]),
            ),
        }
        if FIGURE1A_OUTPUT_STAGE == "pre_activation":
            lines.append(
                "| {projection_group} | {total_elements:,} | {cpu_gt1_ulp_elements} | "
                "{cpu_gt1_ulp_pct} | {gpu_gt1_ulp_elements} | {gpu_gt1_ulp_pct} | "
                "{cpu_max_ulp} | {gpu_max_ulp} |".format(**display_row)
            )
        else:
            lines.append(
                "| {projection_group} | {total_elements:,} | {cpu_gt1_ulp_elements} | "
                "{cpu_gt1_ulp_pct} | {gpu_gt1_ulp_elements_excluding_relu_boundary_flip} | "
                "{gpu_gt1_ulp_pct_excluding_relu_boundary_flip} | "
                "{gpu_relu_boundary_flip_ignored} | {cpu_max_ulp} | "
                "{gpu_max_ulp_excluding_relu_boundary_flip} |".format(**display_row)
            )
    down_row = next((row for row in summary_rows if row["projection_group"] == "mlp.down_proj"), None)
    if down_row:
        if FIGURE1A_OUTPUT_STAGE == "pre_activation":
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "`mlp.down_proj` uses the longest reduction length in the Llama3.1 set (`Q=14336`). "
                    "Since this figure compares pre-activation GEMV, large GPU outliers are attributed "
                    "to CUDA BF16 matmul reduction/rounding-path divergence rather than ReLU boundary flips.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Interpretation",
                    "",
                    "`mlp.down_proj` uses the longest reduction length in the Llama3.1 set (`Q=14336`). "
                    "The GPU `>1 ULP` count after boundary filtering is therefore interpreted as "
                    "CUDA BF16 matmul reduction/rounding-path divergence rather than an AiMRTL "
                    "functional failure: the same `mlp.down_proj` panel has only "
                    f"{down_row['cpu_gt1_ulp_elements']}/{down_row['total_elements']:,} "
                    "`AiMRTL Mode2 vs CPU BF16` elements above 1 ULP.",
                ]
            )
    lines.append("")
    (FIGURE_DIR / "figure1a_llama31_projection_ulp_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def collect_pre_activation_element_rows(functional_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    element_rows: list[dict[str, Any]] = []
    target_rows = [
        row
        for row in functional_rows
        if row["model"] in MODEL_ORDER and row["projection_group"] in PROJECTION_ORDER
    ]
    target_rows = sorted(
        target_rows,
        key=lambda row: (
            MODEL_ORDER.index(row["model"]),
            PROJECTION_ORDER.index(row["projection_group"]),
            row_start_from_tensor_name(row["tensor_name"]),
        ),
    )
    for row in target_rows:
        run_dir = figure1a_run_dir(row)
        if not run_dir.exists():
            raise RuntimeError(
                f"Figure 1b {FIGURE1A_OUTPUT_STAGE} run is missing: {run_dir}. "
                "Regenerate all LLM-shape runs with "
                "`scripts/run_hf_weight_projection.py --output-stage pre_activation`."
            )
        q = int(row["q"])
        r = int(row["r"])
        row_start = row_start_from_tensor_name(row["tensor_name"])
        weight_bf16, activation_bf16 = load_aimd_weight_activation(
            run_dir,
            q,
            r,
            row.get("channel_layout", ""),
        )
        rtl_bits = load_rtl_output(run_dir)
        cpu_bits = load_cpu_reference(run_dir)
        torch_bits = torch_bf16_functional(
            weight_bf16,
            activation_bf16,
            apply_relu=(FIGURE1A_OUTPUT_STAGE == "post_activation"),
            allow_bf16_reduced_precision_reduction=FIGURE1B_ALLOW_BF16_REDUCED_PRECISION_REDUCTION,
        )
        if not (len(rtl_bits) == len(cpu_bits) == len(torch_bits) == r):
            raise RuntimeError(
                f"Length mismatch for {row['run_id']}: "
                f"rtl={len(rtl_bits)} cpu={len(cpu_bits)} torch={len(torch_bits)} r={r}"
            )
        for local_idx in range(r):
            element = row_start + local_idx
            rtl = int(rtl_bits[local_idx])
            cpu = int(cpu_bits[local_idx])
            gpu = int(torch_bits[local_idx])
            element_rows.append(
                {
                    "model": row["model"],
                    "model_source": row["model_source"],
                    "projection_group": row["projection_group"],
                    "projection_module": PROJECTION_MODULE_NAME.get(row["projection_group"], row["projection_group"]),
                    "output_stage": FIGURE1A_OUTPUT_STAGE,
                    "element": element,
                    "local_element": local_idx,
                    "chunk_start": row_start,
                    "run_id": row["run_id"],
                    "rtl_hex": f"0x{rtl:04X}",
                    "cpu_bf16_hex": f"0x{cpu:04X}",
                    "gpu_torch_bf16_hex": f"0x{gpu:04X}",
                    "rtl_bf16_value_exact": format_bf16_bits_exact(rtl),
                    "cpu_bf16_value_exact": format_bf16_bits_exact(cpu),
                    "gpu_torch_bf16_value_exact": format_bf16_bits_exact(gpu),
                    "gpu_torch_bf16_reduction_policy": (
                        "bf16_reduced_precision_reduction_on"
                        if FIGURE1B_ALLOW_BF16_REDUCED_PRECISION_REDUCTION
                        else "bf16_reduced_precision_reduction_off"
                    ),
                    "aimrtl_mode2_vs_cpu_bf16_ulp": bf16_ulp_distance_bits(rtl, cpu),
                    "aimrtl_mode2_vs_gpu_torch_bf16_ulp": bf16_ulp_distance_bits(rtl, gpu),
                }
            )
    return element_rows


def summarize_outlier_magnitude_rows(element_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for group in PROJECTION_ORDER:
            rows = [
                row
                for row in element_rows
                if row["model"] == model and row["projection_group"] == group
            ]
            if not rows:
                continue
            worst = max(
                rows,
                key=lambda row: (
                    int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]),
                    abs(
                        bf16_bits_to_float(int(str(row["rtl_hex"]), 16))
                        - bf16_bits_to_float(int(str(row["gpu_torch_bf16_hex"]), 16))
                    ),
                ),
            )
            n = len(rows)
            cpu_gt1 = sum(1 for row in rows if int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) > 1)
            gpu_gt1 = sum(1 for row in rows if int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) > 1)
            rtl_bits = int(str(worst["rtl_hex"]), 16)
            cpu_bits = int(str(worst["cpu_bf16_hex"]), 16)
            gpu_bits = int(str(worst["gpu_torch_bf16_hex"]), 16)
            rtl_value = bf16_bits_to_float(rtl_bits)
            cpu_value = bf16_bits_to_float(cpu_bits)
            gpu_value = bf16_bits_to_float(gpu_bits)
            abs_error = abs(rtl_value - gpu_value)
            rel_error = abs_error / max(abs(rtl_value), 1e-12)
            within1_min, within1_max = bf16_within_one_ulp_range(rtl_bits)
            summary_rows.append(
                {
                    "model": model,
                    "model_source": rows[0]["model_source"],
                    "projection_group": group,
                    "projection_module": PROJECTION_MODULE_NAME.get(group, group),
                    "output_stage": FIGURE1A_OUTPUT_STAGE,
                    "gpu_torch_bf16_reduction_policy": rows[0].get(
                        "gpu_torch_bf16_reduction_policy",
                        "default",
                    ),
                    "total_elements": n,
                    "cpu_gt1_ulp_elements": cpu_gt1,
                    "cpu_gt1_ulp_pct": 100.0 * cpu_gt1 / n,
                    "gpu_gt1_ulp_elements": gpu_gt1,
                    "gpu_gt1_ulp_pct": 100.0 * gpu_gt1 / n,
                    "cpu_max_ulp": max(int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) for row in rows),
                    "gpu_max_ulp": max(int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) for row in rows),
                    "worst_gpu_output_element": int(worst["element"]),
                    "rtl_hex": f"0x{rtl_bits:04X}",
                    "cpu_bf16_hex": f"0x{cpu_bits:04X}",
                    "gpu_torch_bf16_hex": f"0x{gpu_bits:04X}",
                    "rtl_bf16_display": format_bf16_bits_display(rtl_bits),
                    "cpu_bf16_display": format_bf16_bits_display(cpu_bits),
                    "gpu_torch_bf16_display": format_bf16_bits_display(gpu_bits),
                    "rtl_bf16_value_exact": format_bf16_bits_exact(rtl_bits),
                    "cpu_bf16_value_exact": format_bf16_bits_exact(cpu_bits),
                    "gpu_torch_bf16_value_exact": format_bf16_bits_exact(gpu_bits),
                    "rtl_value": rtl_value,
                    "cpu_bf16_value": cpu_value,
                    "gpu_torch_bf16_value": gpu_value,
                    "gpu_abs_error_vs_rtl": abs_error,
                    "gpu_rel_error_vs_rtl": rel_error,
                    "rtl_within_1ulp_min_value": within1_min,
                    "rtl_within_1ulp_max_value": within1_max,
                    "gpu_inside_rtl_1ulp_value_range": within1_min <= gpu_value <= within1_max,
                }
            )
    return summary_rows


def plot_all_model_projection_outlier_magnitude(element_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_size = FIGURE1B_FONT_SIZE
    configure_plot_font(plt, font_size)
    plot_rows = summarize_outlier_magnitude_rows(element_rows)

    heatmap_delta = np.zeros((len(MODEL_ORDER), len(PROJECTION_ORDER)), dtype=float)
    gt1_pct = np.zeros_like(heatmap_delta)
    max_ulp = np.zeros_like(heatmap_delta, dtype=int)
    aim_values = [["" for _ in PROJECTION_ORDER] for _ in MODEL_ORDER]
    gpu_values = [["" for _ in PROJECTION_ORDER] for _ in MODEL_ORDER]
    row_by_key = {(row["model"], row["projection_group"]): row for row in plot_rows}
    for model_idx, model in enumerate(MODEL_ORDER):
        for group_idx, group in enumerate(PROJECTION_ORDER):
            row = row_by_key[(model, group)]
            heatmap_delta[model_idx, group_idx] = float(row["gpu_abs_error_vs_rtl"])
            gt1_pct[model_idx, group_idx] = float(row["gpu_gt1_ulp_pct"])
            max_ulp[model_idx, group_idx] = int(row["gpu_max_ulp"])
            aim_values[model_idx][group_idx] = str(row["rtl_bf16_display"])
            gpu_values[model_idx][group_idx] = str(row["gpu_torch_bf16_display"])

    delta_max = max(float(heatmap_delta.max()), 1e-12)
    fig, ax = plt.subplots(figsize=(68, 24))
    image = ax.imshow(heatmap_delta, cmap="Greys", aspect="auto", vmin=0, vmax=delta_max)
    ax.set_xticks(np.arange(len(PROJECTION_ORDER)))
    ax.set_xticklabels([PROJECTION_MODULE_NAME[group] for group in PROJECTION_ORDER], fontsize=font_size)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=font_size)
    ax.tick_params(axis="both", length=0)
    for model_idx in range(len(MODEL_ORDER)):
        for group_idx in range(len(PROJECTION_ORDER)):
            delta = heatmap_delta[model_idx, group_idx]
            text_color = "white" if delta > delta_max * 0.55 else "black"
            cell_lines = []
            if gt1_pct[model_idx, group_idx] > 0:
                cell_lines.append(f">1 ULP {gt1_pct[model_idx, group_idx]:.3f}%")
            cell_lines.extend(
                [
                    f"max ULP {max_ulp[model_idx, group_idx]:,}",
                    f"AiM {aim_values[model_idx][group_idx]}",
                    f"GPU {gpu_values[model_idx][group_idx]}",
                ]
            )
            ax.text(
                group_idx,
                model_idx,
                "\n".join(cell_lines),
                ha="center",
                va="center",
                color=text_color,
                fontsize=FIGURE1B_HEATMAP_TEXT_FONT_SIZE,
            )
    fig.subplots_adjust(left=0.13, right=0.84, top=0.97, bottom=0.18)
    cbar_ax = fig.add_axes([0.88, 0.18, 0.012, 0.79])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label(
        "|AiMRTL - GPU BF16| at max GPU ULP element\n"
        "(BF16 reduced-precision reduction OFF)",
        fontsize=FIGURE1B_DATA_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(labelsize=FIGURE1B_DATA_LABEL_FONT_SIZE)
    fig.savefig(
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def write_figure1b_summary_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Figure 1b All-model Pre-activation GEMV Outlier Magnitude",
        "",
        "Figure 1b uses pre-activation GEMV element data for all plotted models. It reports the "
        "`AiMRTL Mode2 vs GPU torch BF16` `>1 ULP` ratio and the worst GPU BF16 element with its "
        "actual AiMRTL/GPU output values.",
        "",
        "The GPU torch BF16 path disables `allow_bf16_reduced_precision_reduction` for this "
        "figure, so the comparison uses BF16 input/output with the more accurate CUDA reduction "
        "path instead of the reduced-precision split-K reduction path.",
        "",
        "The PNG/SVG figure contains only the heatmap panel. In the support table below, rows "
        "with `GPU >1 ULP (%) = 0` are omitted.",
        "",
        "| Model | Projection | GPU >1 ULP (%) | Worst element | GPU max ULP | AiMRTL value | GPU value | Absolute error | AiMRTL +/-1 ULP range |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    nonzero_rows = [row for row in rows if float(row["gpu_gt1_ulp_pct"]) > 0]
    for row in nonzero_rows:
        range_text = (
            f"{format_decimal_power(float(row['rtl_within_1ulp_min_value']))} to "
            f"{format_decimal_power(float(row['rtl_within_1ulp_max_value']))}"
        )
        lines.append(
            "| {model} | {projection_module} | {gpu_gt1_ulp_pct:.6f} | {worst_gpu_output_element} | "
            "{gpu_max_ulp:,} | {rtl_value_s} | {gpu_value_s} | {abs_error_s} | {range_text} |".format(
                **row,
                rtl_value_s=row["rtl_bf16_display"],
                gpu_value_s=row["gpu_torch_bf16_display"],
                abs_error_s=format_decimal_power(float(row["gpu_abs_error_vs_rtl"])),
                range_text=range_text,
            )
        )
    worst_row = max(nonzero_rows, key=lambda row: int(row["gpu_max_ulp"]), default=None)
    if worst_row:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"{len(nonzero_rows)}/{len(rows)} model/projection cases have nonzero GPU `>1 ULP` "
                "elements in the pre-activation GEMV comparison. The largest GPU ULP case is "
                f"`{worst_row['model']} {worst_row['projection_module']}`; its worst element is close "
                "to zero, so a small absolute difference "
                f"({format_decimal_power(float(worst_row['gpu_abs_error_vs_rtl']))}) produces a large "
                f"bit-distance ULP value ({int(worst_row['gpu_max_ulp']):,}).",
            ]
        )
    lines.append("")
    (FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def plot_figure1c_ulp_abs_error(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_size = FIGURE1C_FONT_SIZE
    configure_plot_font(plt, font_size)
    plot_rows = sorted(
        [row for row in rows if row["model"] in FIGURE1C_MODEL_ORDER],
        key=sort_key,
    )
    positive_errors = [
        float(row["gpu_abs_error_vs_rtl"])
        for row in plot_rows
        if float(row["gpu_abs_error_vs_rtl"]) > 0
    ]
    log_floor = min(positive_errors) / 5.0 if positive_errors else 1e-12
    max_error = max(positive_errors) if positive_errors else 1.0
    row_by_key = {(row["model"], row["projection_group"]): row for row in plot_rows}

    x = np.arange(len(FIGURE1C_MODEL_ORDER))
    width = 0.11
    fig, ax = plt.subplots(figsize=(40, 18))
    for group_idx, group in enumerate(PROJECTION_ORDER):
        offsets = x + (group_idx - (len(PROJECTION_ORDER) - 1) / 2) * width
        values: list[float] = []
        visible_values: list[float] = []
        max_ulps: list[int] = []
        for model in FIGURE1C_MODEL_ORDER:
            row = row_by_key[(model, group)]
            error = float(row["gpu_abs_error_vs_rtl"])
            values.append(error)
            visible_values.append(error if error > 0 else log_floor)
            max_ulps.append(int(row["gpu_max_ulp"]))
        bars = ax.bar(
            offsets,
            visible_values,
            width,
            label=PROJECTION_MODULE_NAME[group],
            **FIGURE1C_BAR_STYLES[group_idx],
        )
        for bar, error, ulp in zip(bars, values, max_ulps):
            if error == 0:
                label = "Max ULP 0"
            else:
                label = f"{format_decimal_power(error, 3)}\nMax ULP {ulp}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.18,
                label,
                ha="center",
                va="bottom",
                fontsize=FIGURE1C_DATA_LABEL_FONT_SIZE,
                rotation=90,
            )

    ax.set_yscale("log")
    ax.set_ylim(log_floor * 0.7, max_error * 80.0)
    ax.set_ylabel(
        "|AiMRTL - GPU BF16| absolute error (log scale)",
        fontsize=font_size,
        labelpad=42,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FIGURE1C_MODEL_ORDER, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", which="both", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        fontsize=font_size,
        framealpha=0.9,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.78, bottom=0.16)
    fig.savefig(
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error.png",
        dpi=HIGH_RES_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    fig.savefig(
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error.svg",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)

    for row in plot_rows:
        row["figure1c_abs_error_for_log_plot"] = max(float(row["gpu_abs_error_vs_rtl"]), log_floor)
        row["figure1c_zero_error_log_floor"] = log_floor if float(row["gpu_abs_error_vs_rtl"]) == 0 else 0.0
    return plot_rows


def bf16_roundtrip_ok(bits: int) -> bool:
    bits &= 0xFFFF
    value = bf16_to_float32(np.array([bits], dtype=np.uint16))
    if math.isnan(float(value[0])):
        return True
    return int(float32_to_bf16(value)[0]) == bits


def validate_figure1b_bf16_values(element_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for group in PROJECTION_ORDER:
            rows = [
                row
                for row in element_rows
                if row["model"] == model and row["projection_group"] == group
            ]
            if not rows:
                continue
            total = len(rows)
            parse_failures = 0
            roundtrip_failures = {"rtl": 0, "cpu": 0, "gpu": 0}
            ulp_mismatches = {"cpu": 0, "gpu": 0}
            finite_failures = {"rtl": 0, "cpu": 0, "gpu": 0}
            for row in rows:
                try:
                    rtl_bits = int(str(row["rtl_hex"]), 16)
                    cpu_bits = int(str(row["cpu_bf16_hex"]), 16)
                    gpu_bits = int(str(row["gpu_torch_bf16_hex"]), 16)
                except ValueError:
                    parse_failures += 1
                    continue
                for name, bits in (("rtl", rtl_bits), ("cpu", cpu_bits), ("gpu", gpu_bits)):
                    if not bf16_roundtrip_ok(bits):
                        roundtrip_failures[name] += 1
                    if not math.isfinite(bf16_bits_to_float(bits)):
                        finite_failures[name] += 1
                if int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) != bf16_ulp_distance_bits(rtl_bits, cpu_bits):
                    ulp_mismatches["cpu"] += 1
                if int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) != bf16_ulp_distance_bits(rtl_bits, gpu_bits):
                    ulp_mismatches["gpu"] += 1
            summary_rows.append(
                {
                    "model": model,
                    "projection_group": group,
                    "projection_module": PROJECTION_MODULE_NAME.get(group, group),
                    "output_stage": FIGURE1A_OUTPUT_STAGE,
                    "gpu_torch_bf16_reduction_policy": rows[0].get(
                        "gpu_torch_bf16_reduction_policy",
                        "default",
                    ),
                    "total_elements": total,
                    "hex_parse_failures": parse_failures,
                    "rtl_bf16_roundtrip_failures": roundtrip_failures["rtl"],
                    "cpu_bf16_roundtrip_failures": roundtrip_failures["cpu"],
                    "gpu_bf16_roundtrip_failures": roundtrip_failures["gpu"],
                    "rtl_nonfinite_values": finite_failures["rtl"],
                    "cpu_nonfinite_values": finite_failures["cpu"],
                    "gpu_nonfinite_values": finite_failures["gpu"],
                    "cpu_ulp_recompute_mismatches": ulp_mismatches["cpu"],
                    "gpu_ulp_recompute_mismatches": ulp_mismatches["gpu"],
                    "cpu_gt1_ulp_elements": sum(
                        1 for row in rows if int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) > 1
                    ),
                    "gpu_gt1_ulp_elements": sum(
                        1 for row in rows if int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) > 1
                    ),
                    "cpu_max_ulp": max(int(row["aimrtl_mode2_vs_cpu_bf16_ulp"]) for row in rows),
                    "gpu_max_ulp": max(int(row["aimrtl_mode2_vs_gpu_torch_bf16_ulp"]) for row in rows),
                }
            )
    return summary_rows


def plot_runtime(
    execution_agg: list[dict[str, str]],
    aim_frequency_hz: float,
    nonresident_bandwidth_gbps: float,
    pim_io_bandwidth_gbps: float,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font_size = FOCUS_FIGURE_FONT_SIZE
    configure_plot_font(plt, font_size)

    plot_rows = []
    for row in sorted(execution_agg, key=sort_key):
        if row["model"] not in MODEL_ORDER or row["projection_group"] not in PROJECTION_ORDER:
            continue
        mode1_cycles = float(row["mode1_memory_system_cycles_sum"])
        mode2_cycles = float(row["mode2_memory_system_cycles_sum"])
        q, r = parse_shape(row.get("conceptual_shape", ""))
        input_bytes = q * 2
        weight_bytes = q * r * 2
        output_bytes = r * 2
        total_min_bytes = input_bytes + weight_bytes + output_bytes
        nonresident_data_movement_s = transfer_time_s(total_min_bytes, nonresident_bandwidth_gbps)
        pim_io_bytes = input_bytes + output_bytes
        pim_io_time_s = transfer_time_s(pim_io_bytes, pim_io_bandwidth_gbps)
        mode1_simulated_time_s = mode1_cycles / aim_frequency_hz
        mode2_simulated_time_s = mode2_cycles / aim_frequency_hz
        plot_rows.append(
            {
                "model": row["model"],
                "model_source": row["model_source"],
                "projection_group": row["projection_group"],
                "conceptual_shape": row.get("conceptual_shape", ""),
                "q": q,
                "r": r,
                "input_bytes": input_bytes,
                "weight_bytes": weight_bytes,
                "output_bytes": output_bytes,
                "total_min_bytes": total_min_bytes,
                "pim_io_bytes": pim_io_bytes,
                "nonresident_bandwidth_gbps": nonresident_bandwidth_gbps,
                "pim_io_bandwidth_gbps": pim_io_bandwidth_gbps,
                "nonresident_data_movement_s": nonresident_data_movement_s,
                "pim_io_time_s": pim_io_time_s,
                "cpu_warmup10_s": float(row["cpu_mean_s_sum"]),
                "gpu_torch_bf16_s": float(row["gpu_mean_s_sum"]),
                "cpu_memory_bound_e2e_s": float(row["cpu_mean_s_sum"]) + nonresident_data_movement_s,
                "gpu_torch_memory_bound_e2e_s": float(row["gpu_mean_s_sum"]) + nonresident_data_movement_s,
                "aim_frequency_hz": aim_frequency_hz,
                "mode1_simulated_time_s": mode1_simulated_time_s,
                "mode2_simulated_time_s": mode2_simulated_time_s,
                "pim_mode1_memory_bound_e2e_s": mode1_simulated_time_s + pim_io_time_s,
                "pim_mode2_memory_bound_e2e_s": mode2_simulated_time_s + pim_io_time_s,
                "mode1_memory_system_cycles": int(mode1_cycles),
                "mode2_memory_system_cycles": int(mode2_cycles),
                "mode1_simulator_wall_s": float(row["mode1_wall_time_s_sum"]),
                "mode2_simulator_wall_s": float(row["mode2_wall_time_s_sum"]),
                "mode_cycle_result": row["mode_cycle_result"],
            }
        )

    labels = [make_module_projection_label(row) for row in plot_rows]
    freq_ghz = aim_frequency_hz / 1_000_000_000.0
    series = [
        (f"CPU nonresident E2E @{nonresident_bandwidth_gbps:g}GB/s", "cpu_memory_bound_e2e_s"),
        (f"GPU torch BF16 E2E @{nonresident_bandwidth_gbps:g}GB/s", "gpu_torch_memory_bound_e2e_s"),
        (f"PIM resident E2E @{freq_ghz:g}GHz", "pim_mode1_memory_bound_e2e_s"),
    ]
    x = np.arange(len(plot_rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(24, len(labels) * 1.15), 9.5))
    all_values: list[float] = []
    for idx, (label, field) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        values = [float(row[field]) for row in plot_rows]
        bars = ax.bar(x + offset, values, width, label=label, **MONO_BAR_STYLES[idx])
        add_time_bar_labels(ax, bars, values)
        all_values.extend(values)
    ax.set_yscale("log")
    set_log_ylim_for_labels(ax, all_values)
    ax.set_ylabel("End-to-end time (s, log scale)", fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", which="both", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        fontsize=font_size,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def plot_mode_walltime(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    font_size = FIGURE3_FONT_SIZE
    configure_plot_font(plt, font_size)

    plot_rows = sorted(
        [row for row in runtime_rows if row["model"] in FIGURE3_MODEL_ORDER],
        key=sort_key,
    )
    for row in plot_rows:
        mode1 = float(row["mode1_simulator_wall_s"])
        mode2 = float(row["mode2_simulator_wall_s"])
        row["mode1_simulator_wall_ms"] = mode1 * 1_000.0
        row["mode2_simulator_wall_ms"] = mode2 * 1_000.0
        row["mode2_walltime_slowdown_vs_mode1"] = mode2 / mode1 if mode1 else math.nan

    labels = [make_module_projection_label(row) for row in plot_rows]
    series = [
        ("Mode1 simulator wall time", "mode1_simulator_wall_ms"),
        ("Mode2 simulator wall time", "mode2_simulator_wall_ms"),
    ]
    x = np.arange(len(plot_rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(52, len(labels) * 2.7), 20.0))
    all_values: list[float] = []
    for idx, (label, field) in enumerate(series):
        offset = (idx - 0.5) * width
        values = [float(row[field]) for row in plot_rows]
        bars = ax.bar(x + offset, values, width, label=label, **FIGURE3_BAR_STYLES[idx])
        add_ms_bar_labels(ax, bars, values, font_size=FIGURE3_DATA_LABEL_FONT_SIZE)
        all_values.extend(values)
    set_linear_ylim_for_labels(ax, all_values, upper_factor=1.34)
    ax.set_ylabel("Simulator host wall time (ms)", fontsize=font_size)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: format_axis_number(value)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", rotation_mode="anchor", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        fontsize=font_size,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.44, top=0.78)
    fig.savefig(
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def aggregate_mode_walltime_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for model in FIGURE3_MODEL_ORDER:
        by_model[model] = {
            "model": model,
            "model_source": "",
            "projection_count": 0,
            "mode1_simulator_wall_s": 0.0,
            "mode2_simulator_wall_s": 0.0,
            "mode1_memory_system_cycles": 0,
            "mode2_memory_system_cycles": 0,
            "mode_cycle_result": "PASS",
        }

    for row in rows:
        model = str(row["model"])
        if model not in by_model:
            continue
        out = by_model[model]
        out["model_source"] = row.get("model_source", out["model_source"])
        out["projection_count"] += 1
        out["mode1_simulator_wall_s"] += float(row["mode1_simulator_wall_s"])
        out["mode2_simulator_wall_s"] += float(row["mode2_simulator_wall_s"])
        out["mode1_memory_system_cycles"] += int(float(row["mode1_memory_system_cycles"]))
        out["mode2_memory_system_cycles"] += int(float(row["mode2_memory_system_cycles"]))
        if row.get("mode_cycle_result") != "PASS":
            out["mode_cycle_result"] = "FAIL"

    aggregate_rows: list[dict[str, Any]] = []
    for model in FIGURE3_MODEL_ORDER:
        row = by_model[model]
        mode1_s = float(row["mode1_simulator_wall_s"])
        mode2_s = float(row["mode2_simulator_wall_s"])
        overhead_s = mode2_s - mode1_s
        row["mode1_simulator_wall_ms"] = mode1_s * 1_000.0
        row["mode2_simulator_wall_ms"] = mode2_s * 1_000.0
        row["mode2_rtl_cosim_overhead_s"] = overhead_s
        row["mode2_rtl_cosim_overhead_ms"] = overhead_s * 1_000.0
        row["mode2_walltime_slowdown_vs_mode1"] = mode2_s / mode1_s if mode1_s else math.nan
        row["mode1_share_of_mode2_pct"] = 100.0 * mode1_s / mode2_s if mode2_s else math.nan
        row["mode2_rtl_cosim_overhead_share_pct"] = 100.0 * overhead_s / mode2_s if mode2_s else math.nan
        aggregate_rows.append(row)
    return aggregate_rows


def plot_mode_walltime_model_absolute(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    font_size = FIGURE3_FONT_SIZE
    configure_plot_font(plt, font_size)

    plot_rows = aggregate_mode_walltime_by_model(rows)
    by_model_projection: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        model = str(row["model"])
        group = str(row["projection_group"])
        if model not in FIGURE3_MODEL_ORDER or group not in PROJECTION_ORDER:
            continue
        by_model_projection[(model, group)] = row

    x = np.arange(len(plot_rows))
    width = 0.28
    mode_specs = [
        ("AiM simulator", "mode1_simulator_wall_s", -0.18),
        ("AiM-CoSim", "mode2_simulator_wall_s", 0.18),
    ]
    fig, ax = plt.subplots(figsize=(34, 18))
    segment_label_font = max(22, FIGURE3_DATA_LABEL_FONT_SIZE - 6)
    label_box = {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.82,
    }

    for mode_idx, (mode_label, wall_field, offset) in enumerate(mode_specs):
        bottoms = np.zeros(len(plot_rows), dtype=float)
        positions = x + offset
        total_values = [float(row[wall_field]) for row in plot_rows]
        for group_idx, group in enumerate(PROJECTION_ORDER):
            values = []
            for row in plot_rows:
                source_row = by_model_projection.get((row["model"], group))
                values.append(float(source_row[wall_field]) if source_row else 0.0)
            bars = ax.bar(
                positions,
                values,
                width,
                bottom=bottoms,
                label=PROJECTION_MODULE_NAME[group] if mode_idx == 0 else "_nolegend_",
                **FIGURE1C_BAR_STYLES[group_idx],
            )
            for bar, value, bottom, total_s in zip(bars, values, bottoms, total_values):
                if value <= 0 or total_s <= 0:
                    continue
                visible_bottom = min(float(bottom), FIGURE3A_Y_MAX_S)
                visible_top = min(float(bottom) + float(value), FIGURE3A_Y_MAX_S)
                visible_height = visible_top - visible_bottom
                share_pct = 100.0 * float(value) / total_s
                show_share_label = (
                    mode_label == "AiM-CoSim"
                    and group not in {"self_attn.k_proj", "self_attn.v_proj"}
                )
                if show_share_label and visible_height >= 0.17 and share_pct >= 1.0:
                    gray_face = FIGURE1C_BAR_STYLES[group_idx]["facecolor"]
                    text_color = "white" if gray_face in {"#505050", "#252525"} else "black"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        visible_bottom + visible_height / 2.0,
                        f"{share_pct:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=segment_label_font,
                        color=text_color,
                    )
            bottoms += np.array(values, dtype=float)

        for xpos, total_s in zip(positions, total_values):
            if total_s >= FIGURE3A_Y_MAX_S:
                label_y = FIGURE3A_Y_MAX_S - 0.85
                va = "top"
            else:
                label_y = min(total_s + 0.65, FIGURE3A_Y_MAX_S - 0.85)
                va = "bottom"
            ax.text(
                xpos,
                label_y,
                format_seconds_label(total_s),
                ha="center",
                va=va,
                fontsize=FIGURE3_DATA_LABEL_FONT_SIZE,
                color="black",
                bbox=label_box,
                clip_on=False,
            )

    ax.set_ylim(0, FIGURE3A_Y_MAX_S)
    tick_step = 10.0
    ax.set_yticks(np.arange(0, FIGURE3A_Y_MAX_S + 1.0, tick_step))
    ax.set_ylabel("Simulator host wall time (s)", fontsize=font_size)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: format_axis_number(value)))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row["model"] for row in plot_rows],
        rotation=0,
        ha="center",
        fontsize=max(26, font_size - 10),
    )
    ax.tick_params(axis="x", pad=36, length=0)
    mode_label_font = max(26, font_size - 14)
    for mode_label, _wall_field, offset in mode_specs:
        for xpos in x + offset:
            ax.text(
                xpos,
                -0.014,
                mode_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=mode_label_font,
                clip_on=False,
            )
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        fontsize=FIGURE3_DATA_LABEL_FONT_SIZE,
        framealpha=0.9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.28)
    fig.savefig(
        FIGURE_DIR / "figure3a_model_walltime_comparison.png",
        dpi=HIGH_RES_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure3a_model_walltime_comparison.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def plot_mode_walltime_model_normalized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    font_size = FIGURE3_FONT_SIZE
    configure_plot_font(plt, font_size)

    by_model_projection: dict[tuple[str, str], dict[str, Any]] = {}
    by_model_total: dict[str, dict[str, Any]] = {
        row["model"]: row for row in aggregate_mode_walltime_by_model(rows)
    }
    for row in rows:
        model = str(row["model"])
        group = str(row["projection_group"])
        if model not in FIGURE3_MODEL_ORDER or group not in PROJECTION_ORDER:
            continue
        by_model_projection[(model, group)] = row

    plot_rows: list[dict[str, Any]] = []
    for model in FIGURE3_MODEL_ORDER:
        total = by_model_total[model]
        for mode, wall_field, total_field in [
            ("Mode1", "mode1_simulator_wall_s", "mode1_simulator_wall_s"),
            ("Mode2", "mode2_simulator_wall_s", "mode2_simulator_wall_s"),
        ]:
            total_mode_s = float(total[total_field])
            bottom_pct = 0.0
            for group in PROJECTION_ORDER:
                row = by_model_projection[(model, group)]
                wall_s = float(row[wall_field])
                share_pct = 100.0 * wall_s / total_mode_s if total_mode_s else math.nan
                plot_rows.append(
                    {
                        "model": model,
                        "model_source": row.get("model_source", ""),
                        "mode": mode,
                        "projection_group": group,
                        "projection_module": PROJECTION_MODULE_NAME[group],
                        "conceptual_shape": row.get("conceptual_shape", ""),
                        "mode_simulator_wall_s": wall_s,
                        "mode_simulator_wall_ms": wall_s * 1_000.0,
                        "projection_share_pct": share_pct,
                        "stack_bottom_pct": bottom_pct,
                        "stack_top_pct": bottom_pct + share_pct,
                        "model_mode_simulator_wall_s": total_mode_s,
                        "model_mode_simulator_wall_ms": total_mode_s * 1_000.0,
                        "model_mode1_simulator_wall_s": float(total["mode1_simulator_wall_s"]),
                        "model_mode2_simulator_wall_s": float(total["mode2_simulator_wall_s"]),
                        "model_mode2_walltime_slowdown_vs_mode1": float(
                            total["mode2_walltime_slowdown_vs_mode1"]
                        ),
                        "mode1_memory_system_cycles": row.get("mode1_memory_system_cycles", ""),
                        "mode2_memory_system_cycles": row.get("mode2_memory_system_cycles", ""),
                        "mode_cycle_result": row.get("mode_cycle_result", ""),
                    }
                )
                bottom_pct += share_pct

    model_x = np.arange(len(FIGURE3_MODEL_ORDER))
    width = 0.28
    mode_offsets = {"Mode1": -0.18, "Mode2": 0.18}

    fig, ax = plt.subplots(figsize=(34, 18))
    segment_label_font = max(20, FIGURE3_DATA_LABEL_FONT_SIZE - 6)
    for mode in ["Mode1", "Mode2"]:
        bottoms = np.zeros(len(FIGURE3_MODEL_ORDER), dtype=float)
        positions = model_x + mode_offsets[mode]
        for group_idx, group in enumerate(PROJECTION_ORDER):
            values = []
            for model in FIGURE3_MODEL_ORDER:
                row = next(
                    item
                    for item in plot_rows
                    if item["model"] == model
                    and item["mode"] == mode
                    and item["projection_group"] == group
                )
                values.append(float(row["projection_share_pct"]))
            bars = ax.bar(
                positions,
                values,
                width,
                bottom=bottoms,
                label=PROJECTION_MODULE_NAME[group] if mode == "Mode1" else "_nolegend_",
                **FIGURE1C_BAR_STYLES[group_idx],
            )
            for bar, value, bottom, model in zip(bars, values, bottoms, FIGURE3_MODEL_ORDER):
                force_attention_label = (
                    model in {"Llama3.2-1B-Instruct", "Llama3.1-8B-Instruct"}
                    and group in {"self_attn.q_proj", "self_attn.o_proj"}
                )
                if value >= 9.0 or force_attention_label:
                    gray_face = FIGURE1C_BAR_STYLES[group_idx]["facecolor"]
                    text_color = "white" if gray_face in {"#505050", "#252525"} else "black"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2.0,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=segment_label_font,
                        color=text_color,
                    )
            bottoms += np.array(values, dtype=float)

    for xpos, model in zip(model_x, FIGURE3_MODEL_ORDER):
        total = by_model_total[model]
        for mode, field in [("M1", "mode1_simulator_wall_s"), ("M2", "mode2_simulator_wall_s")]:
            mode_key = "Mode1" if mode == "M1" else "Mode2"
            ax.text(
                xpos + mode_offsets[mode_key],
                103.0,
                f"{mode}\n{format_seconds_label(float(total[field]))}",
                ha="center",
                va="bottom",
                fontsize=segment_label_font,
                clip_on=False,
            )
        ax.text(
            xpos,
            119.0,
            f"{float(total['mode2_walltime_slowdown_vs_mode1']):.1f}x",
            ha="center",
            va="bottom",
            fontsize=segment_label_font,
            clip_on=False,
        )

    ax.set_ylim(0, 132)
    ax.set_ylabel("Wall-time share by projection (%)", fontsize=font_size)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}"))
    ax.set_xticks(model_x)
    ax.set_xticklabels(FIGURE3_MODEL_ORDER, rotation=0, ha="center", fontsize=max(28, font_size - 6))
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        fontsize=FIGURE3_DATA_LABEL_FONT_SIZE,
        framealpha=0.9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.24)
    fig.savefig(
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown.png",
        dpi=HIGH_RES_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def plot_resident_inference(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    font_size = FOCUS_FIGURE_FONT_SIZE
    configure_plot_font(plt, font_size)

    plot_rows = sorted(runtime_rows, key=sort_key)
    for row in plot_rows:
        row["cpu_warmup10_ms"] = float(row["cpu_warmup10_s"]) * 1_000.0
        row["gpu_torch_bf16_ms"] = float(row["gpu_torch_bf16_s"]) * 1_000.0
        row["pim_mode1_memory_bound_e2e_ms"] = float(row["pim_mode1_memory_bound_e2e_s"]) * 1_000.0
    labels = [make_module_projection_label(row) for row in plot_rows]
    freq_ghz = float(plot_rows[0]["aim_frequency_hz"]) / 1_000_000_000.0 if plot_rows else 1.0
    series = [
        ("CPU NumPy BF16", "cpu_warmup10_ms"),
        ("GPU CUDA BF16", "gpu_torch_bf16_ms"),
        (f"PIM @{freq_ghz:g}GHz", "pim_mode1_memory_bound_e2e_ms"),
    ]
    x = np.arange(len(plot_rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(24, len(labels) * 1.15), 9.5))
    all_values: list[float] = []
    for idx, (label, field) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        values = [float(row[field]) for row in plot_rows]
        bars = ax.bar(x + offset, values, width, label=label, **FIGURE2A_BAR_STYLES[idx])
        add_ms_bar_labels(ax, bars, values)
        all_values.extend(values)
    set_linear_ylim_for_labels(ax, all_values, upper_factor=1.34)
    ax.set_ylabel("Resident inference time (ms)", fontsize=font_size)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: format_axis_number(value)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", color=MONO_GRID, alpha=0.22, linewidth=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        fontsize=font_size,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison.svg",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    return plot_rows


def write_report(ok: bool, messages: list[str], outputs: list[Path]) -> None:
    lines = [
        "# Summer 2026 LLM Figure Coverage Report",
        "",
        f"Coverage result: {'PASS' if ok else 'FAIL'}",
        "",
        "Interpretation note: Figure 2 is the memory-bound end-to-end comparison. CPU/GPU add nonresident data movement for `total_min_bytes = (Q + Q*R + R) * 2`; PIM uses resident weights and adds only input/output movement to `memory_system_cycles / frequency`. Figure 2a is the resident inference comparison, where CPU/GPU use measured execution time without extra nonresident transfer. Figure 3 separately plots Mode1/Mode2 simulator host wall time as simulator overhead.",
        "",
        "## Checks",
        "",
        *messages,
        "",
        "## Generated Outputs",
        "",
    ]
    lines.extend(f"- `{path.relative_to(RESULT_ROOT)}`" for path in outputs)
    lines.append("")
    (FIGURE_DIR / "figure_coverage_report.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Summer 2026 LLM-shape figures")
    parser.add_argument("--force", action="store_true", help="Regenerate supplemental torch/CuPy measurements")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--aim-frequency-hz", type=float, default=DEFAULT_AIM_FREQUENCY_HZ)
    parser.add_argument("--nonresident-bandwidth-gbps", type=float, default=DEFAULT_NONRESIDENT_BANDWIDTH_GBPS)
    parser.add_argument("--pim-io-bandwidth-gbps", type=float, default=DEFAULT_PIM_IO_BANDWIDTH_GBPS)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    functional_rows = read_csv(SUMMARY_DIR / "hf_weight_functional_summary.csv")
    functional_agg = read_csv(SUMMARY_DIR / "hf_weight_projection_aggregate_summary.csv")
    execution_agg = read_csv(SUMMARY_DIR / "hf_weight_execution_time_aggregate.csv")
    if any(row.get("activation_source") == "prompt_hash_bf16" for row in functional_rows):
        functional_rows = [row for row in functional_rows if row.get("activation_source") == "prompt_hash_bf16"]

    torch_rows, cupy_rows = generate_supplemental_rows(functional_rows, args.force, args.warmup, args.repeats)
    torch_agg = aggregate_torch_rows(torch_rows)
    cupy_agg = aggregate_cupy_rows(cupy_rows)

    write_csv(
        FIGURE_DIR / "supplemental_torch_functional_aggregate.csv",
        torch_agg,
        [
            "model",
            "model_source",
            "projection_group",
            "execution_rows",
            "status",
            "exact",
            "within_1_ulp",
            "within_2_ulp",
            "max_ulp",
            "avg_ulp_weighted",
            "aimrtl_vs_torch_exact",
            "aimrtl_vs_torch_within_1_ulp",
            "aimrtl_vs_torch_within_2_ulp",
            "aimrtl_vs_torch_max_ulp",
            "aimrtl_vs_torch_avg_ulp_weighted",
        ],
    )
    write_csv(
        FIGURE_DIR / "supplemental_cupy_cuda_aggregate.csv",
        cupy_agg,
        [
            "model",
            "model_source",
            "projection_group",
            "execution_rows",
            "status",
            "cupy_cuda_mean_s_sum",
            "cupy_cuda_median_s_sum",
            "cupy_cuda_min_s_sum",
            "cupy_cuda_max_s_sum",
        ],
    )

    ok, messages = validate_coverage(functional_agg, execution_agg, torch_agg, cupy_agg)
    if not ok:
        write_report(ok, messages, [])
        print("\n".join(messages))
        return 1

    functional_plot_rows = plot_functional(functional_agg, torch_agg)
    figure1a_rows = plot_llama_projection_element_ulp(functional_rows)
    figure1a_summary_rows = summarize_figure1a_ulp(figure1a_rows)
    figure1b_element_rows = collect_pre_activation_element_rows(functional_rows)
    figure1b_rows = plot_all_model_projection_outlier_magnitude(figure1b_element_rows)
    figure1c_rows = plot_figure1c_ulp_abs_error(figure1b_rows)
    figure1b_bf16_check_rows = validate_figure1b_bf16_values(figure1b_element_rows)
    runtime_plot_rows = plot_runtime(
        execution_agg,
        args.aim_frequency_hz,
        args.nonresident_bandwidth_gbps,
        args.pim_io_bandwidth_gbps,
    )
    walltime_plot_rows = plot_mode_walltime(runtime_plot_rows)
    walltime_model_absolute_rows = plot_mode_walltime_model_absolute(walltime_plot_rows)
    walltime_model_normalized_rows = plot_mode_walltime_model_normalized(walltime_plot_rows)
    resident_plot_rows = plot_resident_inference(runtime_plot_rows)
    write_csv(
        FIGURE_DIR / "figure1_llm_shape_functional_validation_data.csv",
        functional_plot_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "conceptual_shape",
            "aimrtl_mode2_vs_cpu_bf16_within_1_ulp_pct",
            "aimrtl_mode2_vs_gpu_torch_bf16_within_1_ulp_pct",
            "aimrtl_mode2_vs_cpu_bf16_max_ulp",
            "aimrtl_mode2_vs_gpu_torch_bf16_max_ulp",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure1a_llama31_projection_element_ulp_data.csv",
        figure1a_rows,
        [
            "panel",
            "projection_group",
            "element",
            "local_element",
            "chunk_start",
            "run_id",
            "rtl_hex",
            "cpu_bf16_hex",
            "gpu_torch_bf16_hex",
            "rtl_bf16_display",
            "cpu_bf16_display",
            "gpu_torch_bf16_display",
            "rtl_bf16_value_exact",
            "cpu_bf16_value_exact",
            "gpu_torch_bf16_value_exact",
            "aimrtl_mode2_vs_cpu_bf16_ulp",
            "aimrtl_mode2_vs_gpu_torch_bf16_ulp",
        ],
    )
    if FIGURE1A_OUTPUT_STAGE == "pre_activation":
        figure1a_summary_fields = [
            "projection_group",
            "total_elements",
            "cpu_gt1_ulp_elements",
            "cpu_gt1_ulp_pct",
            "gpu_gt1_ulp_elements",
            "gpu_gt1_ulp_pct",
            "cpu_max_ulp",
            "gpu_max_ulp",
        ]
    else:
        figure1a_summary_fields = [
            "projection_group",
            "total_elements",
            "cpu_gt1_ulp_elements",
            "cpu_gt1_ulp_pct",
            "gpu_relu_boundary_flip_ignored",
            "gpu_gt1_ulp_elements_excluding_relu_boundary_flip",
            "gpu_gt1_ulp_pct_excluding_relu_boundary_flip",
            "cpu_max_ulp",
            "gpu_max_ulp_excluding_relu_boundary_flip",
        ]
    write_csv(
        FIGURE_DIR / "figure1a_llama31_projection_ulp_summary.csv",
        figure1a_summary_rows,
        figure1a_summary_fields,
    )
    write_figure1a_summary_markdown(figure1a_summary_rows)
    write_csv(
        FIGURE_DIR / "figure1b_all_models_pre_activation_element_ulp_data.csv",
        figure1b_element_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "projection_module",
            "output_stage",
            "element",
            "local_element",
            "chunk_start",
            "run_id",
            "rtl_hex",
            "cpu_bf16_hex",
            "gpu_torch_bf16_hex",
            "rtl_bf16_display",
            "cpu_bf16_display",
            "gpu_torch_bf16_display",
            "rtl_bf16_value_exact",
            "cpu_bf16_value_exact",
            "gpu_torch_bf16_value_exact",
            "gpu_torch_bf16_reduction_policy",
            "aimrtl_mode2_vs_cpu_bf16_ulp",
            "aimrtl_mode2_vs_gpu_torch_bf16_ulp",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude_data.csv",
        figure1b_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "projection_module",
            "output_stage",
            "gpu_torch_bf16_reduction_policy",
            "total_elements",
            "cpu_gt1_ulp_elements",
            "cpu_gt1_ulp_pct",
            "gpu_gt1_ulp_elements",
            "gpu_gt1_ulp_pct",
            "cpu_max_ulp",
            "gpu_max_ulp",
            "worst_gpu_output_element",
            "rtl_hex",
            "cpu_bf16_hex",
            "gpu_torch_bf16_hex",
            "rtl_bf16_display",
            "cpu_bf16_display",
            "gpu_torch_bf16_display",
            "rtl_bf16_value_exact",
            "cpu_bf16_value_exact",
            "gpu_torch_bf16_value_exact",
            "rtl_value",
            "cpu_bf16_value",
            "gpu_torch_bf16_value",
            "gpu_abs_error_vs_rtl",
            "gpu_rel_error_vs_rtl",
            "rtl_within_1ulp_min_value",
            "rtl_within_1ulp_max_value",
            "gpu_inside_rtl_1ulp_value_range",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure1b_all_models_pre_activation_bf16_value_check.csv",
        figure1b_bf16_check_rows,
        [
            "model",
            "projection_group",
            "projection_module",
            "output_stage",
            "gpu_torch_bf16_reduction_policy",
            "total_elements",
            "hex_parse_failures",
            "rtl_bf16_roundtrip_failures",
            "cpu_bf16_roundtrip_failures",
            "gpu_bf16_roundtrip_failures",
            "rtl_nonfinite_values",
            "cpu_nonfinite_values",
            "gpu_nonfinite_values",
            "cpu_ulp_recompute_mismatches",
            "gpu_ulp_recompute_mismatches",
            "cpu_gt1_ulp_elements",
            "gpu_gt1_ulp_elements",
            "cpu_max_ulp",
            "gpu_max_ulp",
        ],
    )
    write_figure1b_summary_markdown(figure1b_rows)
    write_csv(
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error_data.csv",
        figure1c_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "projection_module",
            "output_stage",
            "gpu_torch_bf16_reduction_policy",
            "total_elements",
            "gpu_gt1_ulp_elements",
            "gpu_gt1_ulp_pct",
            "gpu_max_ulp",
            "worst_gpu_output_element",
            "rtl_hex",
            "gpu_torch_bf16_hex",
            "rtl_bf16_display",
            "gpu_torch_bf16_display",
            "gpu_abs_error_vs_rtl",
            "figure1c_abs_error_for_log_plot",
            "figure1c_zero_error_log_floor",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison_data.csv",
        runtime_plot_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "conceptual_shape",
            "q",
            "r",
            "input_bytes",
            "weight_bytes",
            "output_bytes",
            "total_min_bytes",
            "pim_io_bytes",
            "nonresident_bandwidth_gbps",
            "pim_io_bandwidth_gbps",
            "nonresident_data_movement_s",
            "pim_io_time_s",
            "cpu_warmup10_s",
            "gpu_torch_bf16_s",
            "cpu_memory_bound_e2e_s",
            "gpu_torch_memory_bound_e2e_s",
            "aim_frequency_hz",
            "mode1_simulated_time_s",
            "mode2_simulated_time_s",
            "pim_mode1_memory_bound_e2e_s",
            "pim_mode2_memory_bound_e2e_s",
            "mode1_memory_system_cycles",
            "mode2_memory_system_cycles",
            "mode1_simulator_wall_s",
            "mode2_simulator_wall_s",
            "mode_cycle_result",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison_data.csv",
        walltime_plot_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "conceptual_shape",
            "mode1_simulator_wall_s",
            "mode2_simulator_wall_s",
            "mode1_simulator_wall_ms",
            "mode2_simulator_wall_ms",
            "mode2_walltime_slowdown_vs_mode1",
            "mode1_memory_system_cycles",
            "mode2_memory_system_cycles",
            "mode_cycle_result",
        ],
    )
    figure3_model_fields = [
        "model",
        "model_source",
        "projection_count",
        "mode1_simulator_wall_s",
        "mode2_simulator_wall_s",
        "mode2_rtl_cosim_overhead_s",
        "mode1_simulator_wall_ms",
        "mode2_simulator_wall_ms",
        "mode2_rtl_cosim_overhead_ms",
        "mode2_walltime_slowdown_vs_mode1",
        "mode1_share_of_mode2_pct",
        "mode2_rtl_cosim_overhead_share_pct",
        "mode1_memory_system_cycles",
        "mode2_memory_system_cycles",
        "mode_cycle_result",
    ]
    write_csv(
        FIGURE_DIR / "figure3a_model_walltime_comparison_data.csv",
        walltime_model_absolute_rows,
        figure3_model_fields,
    )
    write_csv(
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown_data.csv",
        walltime_model_normalized_rows,
        [
            "model",
            "model_source",
            "mode",
            "projection_group",
            "projection_module",
            "conceptual_shape",
            "mode_simulator_wall_s",
            "mode_simulator_wall_ms",
            "projection_share_pct",
            "stack_bottom_pct",
            "stack_top_pct",
            "model_mode_simulator_wall_s",
            "model_mode_simulator_wall_ms",
            "model_mode1_simulator_wall_s",
            "model_mode2_simulator_wall_s",
            "model_mode2_walltime_slowdown_vs_mode1",
            "mode1_memory_system_cycles",
            "mode2_memory_system_cycles",
            "mode_cycle_result",
        ],
    )
    write_csv(
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison_data.csv",
        resident_plot_rows,
        [
            "model",
            "model_source",
            "projection_group",
            "conceptual_shape",
            "q",
            "r",
            "cpu_warmup10_s",
            "gpu_torch_bf16_s",
            "cpu_warmup10_ms",
            "gpu_torch_bf16_ms",
            "pim_io_time_s",
            "aim_frequency_hz",
            "mode1_simulated_time_s",
            "mode2_simulated_time_s",
            "pim_mode1_memory_bound_e2e_s",
            "pim_mode2_memory_bound_e2e_s",
            "pim_mode1_memory_bound_e2e_ms",
            "mode1_memory_system_cycles",
            "mode2_memory_system_cycles",
            "mode_cycle_result",
        ],
    )
    outputs = [
        FIGURE_DIR / "figure1_llm_shape_functional_validation.png",
        FIGURE_DIR / "figure1_llm_shape_functional_validation.svg",
        FIGURE_DIR / "figure1_llm_shape_functional_validation_data.csv",
        FIGURE_DIR / "figure1a_llama31_projection_element_ulp.png",
        FIGURE_DIR / "figure1a_llama31_projection_element_ulp.svg",
        FIGURE_DIR / "figure1a_llama31_projection_element_ulp_data.csv",
        FIGURE_DIR / "figure1a_llama31_projection_ulp_summary.csv",
        FIGURE_DIR / "figure1a_llama31_projection_ulp_summary.md",
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude.png",
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude.svg",
        FIGURE_DIR / "figure1b_all_models_pre_activation_element_ulp_data.csv",
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude_data.csv",
        FIGURE_DIR / "figure1b_all_models_pre_activation_bf16_value_check.csv",
        FIGURE_DIR / "figure1b_all_models_pre_activation_outlier_magnitude_summary.md",
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error.png",
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error.svg",
        FIGURE_DIR / "figure1c_all_models_ulp_abs_error_data.csv",
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison.png",
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison.svg",
        FIGURE_DIR / "figure2_llm_shape_memory_bound_e2e_comparison_data.csv",
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison.png",
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison.svg",
        FIGURE_DIR / "figure2a_llm_shape_resident_inference_comparison_data.csv",
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison.png",
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison.svg",
        FIGURE_DIR / "figure3_mode1_mode2_walltime_comparison_data.csv",
        FIGURE_DIR / "figure3a_model_walltime_comparison.png",
        FIGURE_DIR / "figure3a_model_walltime_comparison.svg",
        FIGURE_DIR / "figure3a_model_walltime_comparison_data.csv",
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown.png",
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown.svg",
        FIGURE_DIR / "figure3b_model_walltime_normalized_breakdown_data.csv",
        FIGURE_DIR / "supplemental_torch_functional_summary.csv",
        FIGURE_DIR / "supplemental_torch_functional_aggregate.csv",
        FIGURE_DIR / "supplemental_cupy_cuda_summary.csv",
        FIGURE_DIR / "supplemental_cupy_cuda_aggregate.csv",
    ]
    write_report(ok, messages, outputs)
    print("\n".join(messages))
    print(f"wrote figures under {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
