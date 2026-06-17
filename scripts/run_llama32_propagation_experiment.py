#!/usr/bin/env python3
"""Run true propagation experiments by injecting AiMRTL projection outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_hf_weight_projection as hfproj  # noqa: E402


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MODEL_DIR = ROOT / "weight" / DEFAULT_MODEL_ID.replace("/", "__")
MODEL_ID = DEFAULT_MODEL_ID
MODEL_DIR = DEFAULT_MODEL_DIR
RESULT_ROOT = ROOT / "tests" / "result" / "llama32_error_accumulation"
SUMMARY_DIR = RESULT_ROOT / "summary"
FIGURE_DIR = RESULT_ROOT / "figures"
PROP_ROOT = RESULT_ROOT / "runs" / "propagation"
PIM_CACHE_ROOT = PROP_ROOT / "pim_cache"
DEFAULT_RAMULATOR = ROOT / "build_mag32" / "extern" / "aim_simulator" / "ramulator2"
MODEL_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

PROMPTS = [
    "The capital of France is Paris",
    "In machine learning, gradient descent is",
    "The theory of relativity states that",
    "Python is a programming language that",
    "The human brain consists of approximately",
]

PROJECTION_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

ELEMENT_FIELDS = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "tensor_name",
    "projection",
    "token_index",
    "element_index",
    "gpu_bf16_hex",
    "pim_propagated_bf16_hex",
    "gpu_value",
    "pim_propagated_value",
    "delta_gpu_minus_pim",
    "abs_delta",
    "relative_delta",
]

SUMMARY_FIELDS = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "tensor_name",
    "projection",
    "count",
    "delta_mean",
    "delta_std",
    "delta_min",
    "delta_max",
    "abs_delta_p50",
    "abs_delta_p90",
    "abs_delta_p99",
    "abs_delta_max",
    "nonzero_delta_ratio",
    "positive_delta_ratio",
    "negative_delta_ratio",
    "zero_delta_ratio",
]

LOGIT_FIELDS = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "top1_match",
    "gpu_top1_id",
    "pim_top1_id",
    "gpu_top1_token",
    "pim_top1_token",
    "top5_overlap",
    "max_abs_logit_delta",
    "abs_logit_delta_p99",
    "next_token_probability_delta",
    "kl_gpu_to_pim",
    "js_divergence",
]


def clean_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def hex16(value: int) -> str:
    return f"0x{int(value) & 0xFFFF:04X}"


def torch_bf16_bits(tensor: Any) -> np.ndarray:
    import torch

    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    bits = tensor.detach().contiguous().cpu().view(torch.int16).numpy()
    return bits.astype(np.uint16, copy=True)


def bf16_bits_to_torch(bits: Iterable[int], shape: tuple[int, ...], device: Any) -> Any:
    import torch

    arr = np.array(list(bits), dtype=np.uint16).reshape(shape)
    values = hfproj.bf16_to_float32(arr.reshape(-1)).reshape(shape)
    return torch.from_numpy(values).to(device=device, dtype=torch.bfloat16)


def bf16_values(bits: np.ndarray) -> np.ndarray:
    return hfproj.bf16_to_float32(bits.reshape(-1)).reshape(bits.shape)


def ramulator_env(ramulator: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_dir = str(ramulator.resolve().parent)
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = lib_dir if not current else f"{lib_dir}:{current}"
    return env


def parse_cycles(output: str) -> str:
    for line in output.splitlines():
        if "memory_system_cycles:" in line:
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "NA"


def row_sharded_groups_by_channel(r: int) -> dict[int, list[int]]:
    groups_by_channel: dict[int, set[int]] = {ch: set() for ch in range(hfproj.NUM_CHANNELS)}
    for row in range(r):
        group, channel, _bank, _physical_row = hfproj.output_location(row)
        groups_by_channel[channel].add(group)
    return {ch: sorted(groups) for ch, groups in groups_by_channel.items()}


def write_batch_token_aimd(
    path: Path,
    weight_rq_bf16: np.ndarray,
    activations_bf16: np.ndarray,
    q: int,
    r: int,
    vector_row_base: int,
    vector_bank: int = 0,
) -> tuple[int, int]:
    if q % hfproj.ELEMS_PER_TILE != 0:
        raise ValueError(f"Q={q} must be multiple of {hfproj.ELEMS_PER_TILE}")
    if activations_bf16.ndim != 2 or activations_bf16.shape[1] != q:
        raise ValueError(f"Expected activations with shape [tokens, {q}], got {activations_bf16.shape}")
    if vector_row_base % hfproj.NUM_CHANNELS != 0:
        raise ValueError("-- vector_row_base must be aligned to NUM_CHANNELS")
    if vector_bank < 0 or vector_bank >= hfproj.NUM_BANKS:
        raise ValueError(f"vector_bank must be in [0, {hfproj.NUM_BANKS})")

    tiles_q = q // hfproj.ELEMS_PER_TILE
    tiles_per_row = max(hfproj.MIN_TILES_PER_ROW, tiles_q)
    active_rows = max((hfproj.output_location(row)[3] for row in range(r)), default=-1) + 1
    token_count = int(activations_bf16.shape[0])
    rows_per_bank = vector_row_base + token_count * hfproj.NUM_CHANNELS
    if rows_per_bank > hfproj.MAX_ROWS:
        raise ValueError(
            f"batch-token AIMD requires {rows_per_bank} rows per bank, "
            f"exceeding MAX_ROWS={hfproj.MAX_ROWS}"
        )

    zero_tile = np.zeros(hfproj.ELEMS_PER_TILE, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct_pack_header(hfproj.AIMD_MAGIC, hfproj.NUM_BANKS, rows_per_bank, tiles_per_row))
        for bank in range(hfproj.NUM_BANKS):
            for row in range(rows_per_bank):
                if row < active_rows:
                    group = row // hfproj.NUM_CHANNELS
                    channel = row % hfproj.NUM_CHANNELS
                    output_row = group * (hfproj.NUM_CHANNELS * hfproj.NUM_BANKS) + channel * hfproj.NUM_BANKS + bank
                    if output_row < r:
                        row_data = weight_rq_bf16[output_row]
                        for tile in range(tiles_per_row):
                            if tile < tiles_q:
                                start = tile * hfproj.ELEMS_PER_TILE
                                end = start + hfproj.ELEMS_PER_TILE
                                f.write(row_data[start:end].tobytes())
                            else:
                                f.write(zero_tile.tobytes())
                        continue
                if vector_row_base <= row < rows_per_bank and bank == vector_bank:
                    token_index = (row - vector_row_base) // hfproj.NUM_CHANNELS
                    row_data = activations_bf16[token_index]
                    for tile in range(tiles_per_row):
                        if tile < tiles_q:
                            start = tile * hfproj.ELEMS_PER_TILE
                            end = start + hfproj.ELEMS_PER_TILE
                            f.write(row_data[start:end].tobytes())
                        else:
                            f.write(zero_tile.tobytes())
                    continue
                f.write(bytes(tiles_per_row * hfproj.ELEMS_PER_TILE * 2))

        first_activation = activations_bf16[0] if token_count else np.zeros(q, dtype=np.uint16)
        for tile in range(tiles_per_row):
            if tile < tiles_q:
                start = tile * hfproj.ELEMS_PER_TILE
                end = start + hfproj.ELEMS_PER_TILE
                f.write(first_activation[start:end].tobytes())
            else:
                f.write(zero_tile.tobytes())

        for gpr in range(hfproj.NUM_GPRS):
            if gpr < tiles_q:
                start = gpr * hfproj.ELEMS_PER_TILE
                end = start + hfproj.ELEMS_PER_TILE
                f.write(first_activation[start:end].tobytes())
            else:
                f.write(zero_tile.tobytes())

        f.write(bytes(hfproj.NUM_BANKS * 2))

    return tiles_per_row, rows_per_bank


def struct_pack_header(magic: int, num_banks: int, rows_per_bank: int, tiles_per_row: int) -> bytes:
    import struct

    return struct.pack("<IIII", magic, num_banks, rows_per_bank, tiles_per_row)


def write_batch_token_trace(
    path: Path,
    q: int,
    r: int,
    token_count: int,
    vector_row_base: int,
    vector_bank: int = 0,
    output_stage: str = "pre_activation",
) -> None:
    if output_stage not in {"post_activation", "pre_activation"}:
        raise ValueError(f"Unsupported output stage: {output_stage}")
    opsize = q // hfproj.ELEMS_PER_TILE
    physical_rows_by_group: dict[int, list[int]] = {}
    for row in range(r):
        group, _channel, _bank, physical_row = hfproj.output_location(row)
        physical_rows_by_group.setdefault(group, [])
        if physical_row not in physical_rows_by_group[group]:
            physical_rows_by_group[group].append(physical_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# AiM HF-weight batch-token GEMV trace: tokens={token_count} Q={q} R={r}\n")
        f.write(
            f"# opsize={opsize}, channel_layout=row_sharded, output_stage={output_stage}, "
            f"vector_bank={vector_bank}, vector_row_base={vector_row_base}\n"
        )
        for token_index in range(token_count):
            f.write(f"# batch_token={token_index}\n")
            for group in sorted(physical_rows_by_group):
                physical_rows = sorted(physical_rows_by_group[group])
                f.write(f"# row_shard_group={group}, physical_rows={physical_rows[0]}..{physical_rows[-1]}\n")
                for physical_row in physical_rows:
                    channel = physical_row % hfproj.NUM_CHANNELS
                    channel_mask = hfproj.channel_mask_for(channel)
                    f.write(f"AiM WR_BIAS {opsize} {channel_mask}\n")
                for physical_row in physical_rows:
                    channel = physical_row % hfproj.NUM_CHANNELS
                    channel_mask = hfproj.channel_mask_for(channel)
                    vector_row = vector_row_base + token_index * hfproj.NUM_CHANNELS + channel
                    f.write(f"AiM COPY_BKGB {opsize} {channel_mask} {vector_bank} {vector_row}\n")
                for physical_row in physical_rows:
                    channel_mask = hfproj.channel_mask_for(physical_row % hfproj.NUM_CHANNELS)
                    f.write(f"AiM MAC_ABK {opsize} {channel_mask} {physical_row}\n")
                if output_stage == "post_activation":
                    for physical_row in physical_rows:
                        channel_mask = hfproj.channel_mask_for(physical_row % hfproj.NUM_CHANNELS)
                        f.write(f"AiM AF {channel_mask}\n")
                    for physical_row in physical_rows:
                        channel_mask = hfproj.channel_mask_for(physical_row % hfproj.NUM_CHANNELS)
                        f.write(f"AiM RD_AF {opsize} {channel_mask}\n")
                else:
                    for physical_row in physical_rows:
                        channel_mask = hfproj.channel_mask_for(physical_row % hfproj.NUM_CHANNELS)
                        f.write(f"AiM RD_MAC {opsize} {channel_mask}\n")
        f.write("AiM ISR_EOC\n")


def parse_batch_token_results(
    result_base: Path,
    r: int,
    token_count: int,
    output_stage: str = "pre_activation",
) -> list[list[int]]:
    target_cmd = hfproj.CMD_RDAF16 if output_stage == "post_activation" else hfproj.CMD_RDMAC16
    by_channel: dict[int, list[list[int | None]]] = {}
    for ch in range(hfproj.NUM_CHANNELS):
        csv_path = Path(f"{result_base}_ch{ch}.csv")
        by_channel[ch] = hfproj.parse_rtl_events(csv_path, target_cmd) if csv_path.exists() else []

    groups_by_channel = row_sharded_groups_by_channel(r)
    group_pos = {ch: {group: idx for idx, group in enumerate(groups)} for ch, groups in groups_by_channel.items()}
    merged_by_token: list[list[int]] = []
    for token_index in range(token_count):
        merged: list[int] = []
        for row in range(r):
            group, channel, bank, _physical_row = hfproj.output_location(row)
            local_groups = groups_by_channel[channel]
            local_index = token_index * len(local_groups) + group_pos[channel][group]
            events = by_channel[channel]
            if local_index >= len(events):
                raise RuntimeError(
                    f"Missing batch-token RTL result for token {token_index}, output row {row} "
                    f"(channel {channel}, local result index {local_index})"
                )
            value = events[local_index][bank]
            if value is None:
                raise RuntimeError(
                    f"Missing batch-token RTL bank result for token {token_index}, output row {row} "
                    f"(channel {channel}, bank {bank}, local result index {local_index})"
                )
            merged.append(value)
        merged_by_token.append(merged)
    return merged_by_token


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_model_dir(model_id: str) -> Path:
    return ROOT / "weight" / model_id.replace("/", "__")


def resolve_model_dir(model_id: str, model_dir: Path | None = None) -> Path:
    return Path(model_dir).expanduser() if model_dir else default_model_dir(model_id)


def model_files_ready(model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
    if not model_dir.exists():
        return False
    if not all((model_dir / name).exists() for name in MODEL_REQUIRED_FILES):
        return False
    return any(model_dir.glob("*.safetensors"))


def ensure_model_available(
    offline: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    model_dir: Path | None = None,
) -> Path:
    model_dir = resolve_model_dir(model_id, model_dir)
    if model_files_ready(model_dir):
        return model_dir
    if offline:
        raise FileNotFoundError(
            f"Missing local model files under {model_dir}. "
            "Re-run without --offline or set --model-dir to a populated local model path."
        )

    downloader = ROOT / "scripts" / "download_hf_weights.py"
    if not downloader.exists():
        raise FileNotFoundError(f"Missing HF downloader script: {downloader}")

    env = os.environ.copy()
    if not env.get("HF_TOKEN") and env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HF_TOKEN"] = env["HUGGINGFACE_HUB_TOKEN"]
    if not env.get("HF_TOKEN"):
        raise RuntimeError(
            "HF model weight/tokenizer files are missing and HF_TOKEN is not set. "
            "Set HF_TOKEN to a Hugging Face token with access to "
            f"{model_id}, then re-run the command."
        )

    cmd = [
        sys.executable,
        str(downloader),
        model_id,
        "--output-dir",
        str(model_dir),
        "--include-tokenizer",
    ]
    print(f"Model files are missing; downloading {model_id} into {model_dir}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"HF model download failed with exit code {proc.returncode}")
    if not model_files_ready(model_dir):
        raise RuntimeError(f"HF download completed but required files are still missing under {model_dir}")
    return model_dir


class SummaryBuckets:
    def __init__(self, key_names: list[str]) -> None:
        self.key_names = key_names
        self.values: dict[tuple[Any, ...], list[float]] = {}

    def add(self, key: dict[str, Any], delta: float) -> None:
        bucket_key = tuple(key.get(name, "") for name in self.key_names)
        self.values.setdefault(bucket_key, []).append(delta)

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bucket_key, deltas in sorted(self.values.items()):
            key = {name: bucket_key[idx] for idx, name in enumerate(self.key_names)}
            rows.append(summary_row(key, deltas))
        return rows


def summary_row(key: dict[str, Any], deltas: list[float]) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=np.float64)
    abs_arr = np.abs(arr)
    count = int(arr.size)
    return {
        **key,
        "count": count,
        "delta_mean": float(np.mean(arr)) if count else 0.0,
        "delta_std": float(np.std(arr)) if count else 0.0,
        "delta_min": float(np.min(arr)) if count else 0.0,
        "delta_max": float(np.max(arr)) if count else 0.0,
        "abs_delta_p50": float(np.percentile(abs_arr, 50)) if count else 0.0,
        "abs_delta_p90": float(np.percentile(abs_arr, 90)) if count else 0.0,
        "abs_delta_p99": float(np.percentile(abs_arr, 99)) if count else 0.0,
        "abs_delta_max": float(np.max(abs_arr)) if count else 0.0,
        "nonzero_delta_ratio": float(np.sum(arr != 0.0)) / count if count else 0.0,
        "positive_delta_ratio": float(np.sum(arr > 0.0)) / count if count else 0.0,
        "negative_delta_ratio": float(np.sum(arr < 0.0)) / count if count else 0.0,
        "zero_delta_ratio": float(np.sum(arr == 0.0)) / count if count else 0.0,
    }


class PIMRunner:
    def __init__(
        self,
        ramulator: Path,
        cache_root: Path,
        resume: bool = True,
        keep_artifacts: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
        batch_tokens: bool = False,
        parallel_workers: int = 1,
        batch_token_chunk_size: int = 0,
    ) -> None:
        self.ramulator = ramulator
        self.cache_root = cache_root
        self.resume = resume
        self.keep_artifacts = keep_artifacts
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.parallel_workers = max(1, int(parallel_workers))
        self.batch_token_chunk_size = max(0, int(batch_token_chunk_size))
        self.active_layers: set[int] = set()
        self.policy = "disabled"
        self.target_layer = -1
        self.prompt_index = 0
        self.decode_step = 0
        self.runs = 0
        self.batch_runs = 0
        self.gemv_calls = 0
        self.cache_hits = 0
        self.total_wall_time_s = 0.0
        self._stats_lock = threading.Lock()

    def set_context(
        self,
        policy: str,
        target_layer: int,
        active_layers: set[int],
        prompt_index: int,
        decode_step: int,
    ) -> None:
        self.policy = policy
        self.target_layer = target_layer
        self.active_layers = active_layers
        self.prompt_index = prompt_index
        self.decode_step = decode_step

    def should_replace(self, layer_idx: int) -> bool:
        return layer_idx in self.active_layers

    def _case_dir_for(
        self,
        layer_idx: int,
        projection: str,
        q: int,
        r: int,
        activation_bf16: np.ndarray,
    ) -> tuple[Path, str]:
        digest = hashlib.sha256()
        digest.update(f"model={self.model_id};layer={layer_idx};projection={projection};q={q};r={r};".encode("utf-8"))
        digest.update(activation_bf16.tobytes())
        case_hash = digest.hexdigest()[:20]
        case_dir = (
            self.cache_root
            / f"layer_{layer_idx:02d}"
            / clean_name(projection)
            / f"q{q}_r{r}_{case_hash}"
        )
        return case_dir, case_hash

    def _read_cached_bits(self, output_path: Path, r: int) -> list[int] | None:
        if not (self.resume and output_path.exists()):
            return None
        with output_path.open(newline="", encoding="utf-8") as f:
            bits = [int(row["pim_bf16_hex"], 16) for row in csv.DictReader(f)]
        if len(bits) != r:
            return None
        with self._stats_lock:
            self.cache_hits += 1
        return bits

    def _write_output_cache(self, output_path: Path, bits: list[int]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["output_element", "pim_bf16_hex", "pim_value"])
            writer.writeheader()
            values = bf16_values(np.asarray(bits, dtype=np.uint16))
            for idx, bit in enumerate(bits):
                writer.writerow(
                    {
                        "output_element": idx,
                        "pim_bf16_hex": hex16(bit),
                        "pim_value": f"{float(values[idx]):.10g}",
                    }
                )

    def run_gemv(
        self,
        layer_idx: int,
        projection: str,
        weight_rq_bf16: np.ndarray,
        activation_bf16: np.ndarray,
        token_index: int,
    ) -> list[int]:
        q = int(weight_rq_bf16.shape[1])
        r = int(weight_rq_bf16.shape[0])
        if int(activation_bf16.shape[0]) != q:
            raise RuntimeError(f"Activation shape mismatch for {projection}: {activation_bf16.shape[0]} != {q}")
        with self._stats_lock:
            self.gemv_calls += 1
        case_dir, case_hash = self._case_dir_for(layer_idx, projection, q, r, activation_bf16)
        output_path = case_dir / "pim_output_bf16.csv"
        metadata_path = case_dir / "metadata.json"
        cached_bits = self._read_cached_bits(output_path, r)
        if cached_bits is not None:
            return cached_bits

        case_dir.mkdir(parents=True, exist_ok=True)
        shape = f"p1_q{q}_r{r}"
        aimd_path = case_dir / f"{shape}.aimd"
        trace_path = case_dir / f"{shape}.trace"
        config_path = case_dir / "aim_rtl_file.yaml"
        result_base = case_dir / "rtl_results"
        mode2_out = case_dir / "mode2.txt"

        tiles_per_row, rows_per_bank = hfproj.write_aimd(
            aimd_path,
            weight_rq_bf16,
            activation_bf16,
            q,
            r,
            dense_rows=False,
            channel_layout="row_sharded",
        )
        hfproj.write_trace(trace_path, q, r, channel_layout="row_sharded", output_stage="pre_activation")
        hfproj.write_config(config_path, aimd_path, result_base, tiles_per_row, seed=42, channel_layout="row_sharded")

        start = time.perf_counter()
        proc = subprocess.run(
            [str(self.ramulator), "-f", str(config_path), "-t", str(trace_path)],
            cwd=ROOT,
            env=ramulator_env(self.ramulator),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall_time_s = time.perf_counter() - start
        with self._stats_lock:
            self.total_wall_time_s += wall_time_s
            self.runs += 1
        mode2_out.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Mode2 failed for {case_dir}\n{proc.stdout[-4000:]}")

        bits = hfproj.parse_rtl_results(result_base, r, "row_sharded", "pre_activation")
        if len(bits) != r:
            raise RuntimeError(f"Mode2 output mismatch for {case_dir}: {len(bits)} != {r}")

        self._write_output_cache(output_path, [int(v) for v in bits])

        metadata = {
            "model_id": self.model_id,
            "policy": self.policy,
            "target_layer": self.target_layer,
            "prompt_index": self.prompt_index,
            "decode_step": self.decode_step,
            "layer": layer_idx,
            "projection": projection,
            "token_index": token_index,
            "q": q,
            "r": r,
            "trace_opsize": q // hfproj.ELEMS_PER_TILE,
            "tiles_per_row": tiles_per_row,
            "aimd_rows_per_bank": rows_per_bank,
            "memory_system_cycles": parse_cycles(proc.stdout),
            "mode2_wall_time_s": wall_time_s,
            "case_hash": case_hash,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if not self.keep_artifacts:
            aimd_path.unlink(missing_ok=True)
            for csv_path in case_dir.glob("rtl_results_ch*.csv"):
                csv_path.unlink(missing_ok=True)

        return [int(v) for v in bits]

    def run_gemv_batch(
        self,
        layer_idx: int,
        projection: str,
        weight_rq_bf16: np.ndarray,
        activations_bf16: np.ndarray,
    ) -> list[list[int]]:
        q = int(weight_rq_bf16.shape[1])
        r = int(weight_rq_bf16.shape[0])
        if activations_bf16.ndim != 2 or int(activations_bf16.shape[1]) != q:
            raise RuntimeError(f"Activation batch shape mismatch for {projection}: {activations_bf16.shape} != (*, {q})")
        token_count = int(activations_bf16.shape[0])
        with self._stats_lock:
            self.gemv_calls += token_count

        outputs: list[list[int] | None] = [None] * token_count
        missing_indices: list[int] = []
        missing_activations: list[np.ndarray] = []
        for token_index in range(token_count):
            case_dir, _case_hash = self._case_dir_for(layer_idx, projection, q, r, activations_bf16[token_index])
            cached_bits = self._read_cached_bits(case_dir / "pim_output_bf16.csv", r)
            if cached_bits is None:
                missing_indices.append(token_index)
                missing_activations.append(activations_bf16[token_index])
            else:
                outputs[token_index] = cached_bits

        if missing_indices:
            chunks = self._batch_chunks(missing_indices, missing_activations)
            if self.parallel_workers > 1 and len(chunks) > 1:
                with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(chunks))) as executor:
                    futures = [
                        executor.submit(
                            self._run_gemv_batch_uncached,
                            layer_idx,
                            projection,
                            weight_rq_bf16,
                            np.stack(chunk_activations, axis=0),
                            chunk_indices,
                        )
                        for chunk_indices, chunk_activations in chunks
                    ]
                    for future in futures:
                        for token_index, bits in future.result().items():
                            outputs[token_index] = bits
            else:
                for chunk_indices, chunk_activations in chunks:
                    result = self._run_gemv_batch_uncached(
                        layer_idx,
                        projection,
                        weight_rq_bf16,
                        np.stack(chunk_activations, axis=0),
                        chunk_indices,
                    )
                    for token_index, bits in result.items():
                        outputs[token_index] = bits

        if any(bits is None for bits in outputs):
            missing = [idx for idx, bits in enumerate(outputs) if bits is None]
            raise RuntimeError(f"Missing batch-token outputs for {projection}: token indices {missing}")
        return [list(bits) for bits in outputs if bits is not None]

    def _batch_chunks(
        self,
        token_indices: list[int],
        activations: list[np.ndarray],
    ) -> list[tuple[list[int], list[np.ndarray]]]:
        if self.batch_token_chunk_size <= 0 or self.batch_token_chunk_size >= len(token_indices):
            return [(token_indices, activations)]
        chunks: list[tuple[list[int], list[np.ndarray]]] = []
        for start in range(0, len(token_indices), self.batch_token_chunk_size):
            end = start + self.batch_token_chunk_size
            chunks.append((token_indices[start:end], activations[start:end]))
        return chunks

    def _run_gemv_batch_uncached(
        self,
        layer_idx: int,
        projection: str,
        weight_rq_bf16: np.ndarray,
        activations_bf16: np.ndarray,
        token_indices: list[int],
    ) -> dict[int, list[int]]:
        q = int(weight_rq_bf16.shape[1])
        r = int(weight_rq_bf16.shape[0])
        token_count = int(activations_bf16.shape[0])
        digest = hashlib.sha256()
        digest.update(f"model={self.model_id};layer={layer_idx};projection={projection};q={q};r={r};".encode("utf-8"))
        digest.update(f"tokens={','.join(str(idx) for idx in token_indices)};".encode("utf-8"))
        digest.update(activations_bf16.tobytes())
        batch_hash = digest.hexdigest()[:20]
        batch_dir = (
            self.cache_root
            / f"layer_{layer_idx:02d}"
            / clean_name(projection)
            / f"batch_q{q}_r{r}_n{token_count}_{batch_hash}"
        )
        batch_dir.mkdir(parents=True, exist_ok=True)

        shape = f"p{token_count}_q{q}_r{r}"
        aimd_path = batch_dir / f"{shape}.aimd"
        trace_path = batch_dir / f"{shape}.trace"
        config_path = batch_dir / "aim_rtl_file.yaml"
        result_base = batch_dir / "rtl_results"
        mode2_out = batch_dir / "mode2.txt"

        active_rows = max((hfproj.output_location(row)[3] for row in range(r)), default=-1) + 1
        vector_row_base = int(math.ceil(active_rows / hfproj.NUM_CHANNELS) * hfproj.NUM_CHANNELS)
        vector_bank = 0
        tiles_per_row, rows_per_bank = write_batch_token_aimd(
            aimd_path,
            weight_rq_bf16,
            activations_bf16,
            q,
            r,
            vector_row_base,
            vector_bank,
        )
        write_batch_token_trace(
            trace_path,
            q,
            r,
            token_count,
            vector_row_base,
            vector_bank,
            output_stage="pre_activation",
        )
        hfproj.write_config(config_path, aimd_path, result_base, tiles_per_row, seed=42, channel_layout="row_sharded")

        start = time.perf_counter()
        proc = subprocess.run(
            [str(self.ramulator), "-f", str(config_path), "-t", str(trace_path)],
            cwd=ROOT,
            env=ramulator_env(self.ramulator),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall_time_s = time.perf_counter() - start
        with self._stats_lock:
            self.total_wall_time_s += wall_time_s
            self.runs += 1
            self.batch_runs += 1
        mode2_out.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Batch-token Mode2 failed for {batch_dir}\n{proc.stdout[-4000:]}")

        bits_by_position = parse_batch_token_results(result_base, r, token_count, "pre_activation")
        if len(bits_by_position) != token_count:
            raise RuntimeError(f"Batch-token output mismatch for {batch_dir}: {len(bits_by_position)} != {token_count}")

        outputs: dict[int, list[int]] = {}
        cycles = parse_cycles(proc.stdout)
        for batch_position, token_index in enumerate(token_indices):
            bits = [int(v) for v in bits_by_position[batch_position]]
            if len(bits) != r:
                raise RuntimeError(f"Batch-token output width mismatch for {batch_dir}: {len(bits)} != {r}")
            case_dir, case_hash = self._case_dir_for(layer_idx, projection, q, r, activations_bf16[batch_position])
            case_dir.mkdir(parents=True, exist_ok=True)
            self._write_output_cache(case_dir / "pim_output_bf16.csv", bits)
            metadata = {
                "model_id": self.model_id,
                "policy": self.policy,
                "target_layer": self.target_layer,
                "prompt_index": self.prompt_index,
                "decode_step": self.decode_step,
                "layer": layer_idx,
                "projection": projection,
                "token_index": token_index,
                "q": q,
                "r": r,
                "trace_opsize": q // hfproj.ELEMS_PER_TILE,
                "tiles_per_row": tiles_per_row,
                "aimd_rows_per_bank": rows_per_bank,
                "memory_system_cycles": cycles,
                "mode2_wall_time_s": wall_time_s,
                "case_hash": case_hash,
                "batch_token_mode": True,
                "batch_token_count": token_count,
                "batch_token_position": batch_position,
                "batch_case_hash": batch_hash,
                "vector_load": "COPY_BKGB",
                "vector_bank": vector_bank,
                "vector_row_base": vector_row_base,
            }
            (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            outputs[token_index] = bits

        batch_metadata = {
            "model_id": self.model_id,
            "policy": self.policy,
            "target_layer": self.target_layer,
            "prompt_index": self.prompt_index,
            "decode_step": self.decode_step,
            "layer": layer_idx,
            "projection": projection,
            "token_indices": token_indices,
            "q": q,
            "r": r,
            "token_count": token_count,
            "trace_opsize": q // hfproj.ELEMS_PER_TILE,
            "tiles_per_row": tiles_per_row,
            "aimd_rows_per_bank": rows_per_bank,
            "memory_system_cycles": cycles,
            "mode2_wall_time_s": wall_time_s,
            "batch_case_hash": batch_hash,
            "vector_load": "COPY_BKGB",
            "vector_bank": vector_bank,
            "vector_row_base": vector_row_base,
        }
        (batch_dir / "metadata.json").write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")

        if not self.keep_artifacts:
            aimd_path.unlink(missing_ok=True)
            for csv_path in batch_dir.glob("rtl_results_ch*.csv"):
                csv_path.unlink(missing_ok=True)

        return outputs


class PIMLinearWrapper:
    def __init__(self, original: Any, layer_idx: int, projection: str, runner: PIMRunner) -> None:
        import torch

        self.module = torch.nn.Module()
        self.module.original = original
        self.layer_idx = layer_idx
        self.projection = projection
        self.runner = runner
        self.in_features = int(original.in_features)
        self.out_features = int(original.out_features)
        self.weight_bf16 = torch_bf16_bits(original.weight.detach()).reshape(self.out_features, self.in_features)

    def as_module(self) -> Any:
        import torch

        outer = self

        class WrappedLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.original = outer.module.original

            def forward(self, input: Any) -> Any:
                if not outer.runner.should_replace(outer.layer_idx):
                    return self.original(input)
                if input.shape[-1] != outer.in_features:
                    raise RuntimeError(
                        f"{outer.projection} expected input dim {outer.in_features}, got {input.shape[-1]}"
                    )
                flat = input.reshape(-1, outer.in_features)
                activations = [torch_bf16_bits(flat[token_index]) for token_index in range(flat.shape[0])]
                if outer.runner.batch_tokens and len(activations) > 1:
                    batch_bits = outer.runner.run_gemv_batch(
                        outer.layer_idx,
                        outer.projection,
                        outer.weight_bf16,
                        np.stack(activations, axis=0),
                    )
                    outputs = [
                        bf16_bits_to_torch(pim_bits, (outer.out_features,), input.device)
                        for pim_bits in batch_bits
                    ]
                elif outer.runner.parallel_workers > 1 and len(activations) > 1:
                    def run_one(item: tuple[int, np.ndarray]) -> list[int]:
                        token_index, activation_bits = item
                        return outer.runner.run_gemv(
                            outer.layer_idx,
                            outer.projection,
                            outer.weight_bf16,
                            activation_bits,
                            token_index,
                        )

                    with ThreadPoolExecutor(max_workers=min(outer.runner.parallel_workers, len(activations))) as executor:
                        batch_bits = list(executor.map(run_one, enumerate(activations)))
                    outputs = [
                        bf16_bits_to_torch(pim_bits, (outer.out_features,), input.device)
                        for pim_bits in batch_bits
                    ]
                else:
                    outputs = []
                    for token_index, activation_bits in enumerate(activations):
                        pim_bits = outer.runner.run_gemv(
                            outer.layer_idx,
                            outer.projection,
                            outer.weight_bf16,
                            activation_bits,
                            token_index,
                        )
                        outputs.append(bf16_bits_to_torch(pim_bits, (outer.out_features,), input.device))
                stacked = torch.stack(outputs, dim=0)
                return stacked.reshape(*input.shape[:-1], outer.out_features)

        return WrappedLinear()


def decoder_layers(model: Any) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Expected a decoder-only HF model with model.layers")
    return layers


def decoder_layer_count(model: Any) -> int:
    return len(decoder_layers(model))


def wrap_model_projections(model: Any, runner: PIMRunner) -> None:
    for layer_idx, layer in enumerate(decoder_layers(model)):
        layer.self_attn.q_proj = PIMLinearWrapper(layer.self_attn.q_proj, layer_idx, "self_attn.q_proj", runner).as_module()
        layer.self_attn.k_proj = PIMLinearWrapper(layer.self_attn.k_proj, layer_idx, "self_attn.k_proj", runner).as_module()
        layer.self_attn.v_proj = PIMLinearWrapper(layer.self_attn.v_proj, layer_idx, "self_attn.v_proj", runner).as_module()
        layer.self_attn.o_proj = PIMLinearWrapper(layer.self_attn.o_proj, layer_idx, "self_attn.o_proj", runner).as_module()
        layer.mlp.gate_proj = PIMLinearWrapper(layer.mlp.gate_proj, layer_idx, "mlp.gate_proj", runner).as_module()
        layer.mlp.up_proj = PIMLinearWrapper(layer.mlp.up_proj, layer_idx, "mlp.up_proj", runner).as_module()
        layer.mlp.down_proj = PIMLinearWrapper(layer.mlp.down_proj, layer_idx, "mlp.down_proj", runner).as_module()


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def default_probe_layers(num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("Expected a model with at least one decoder layer")
    if num_layers == 16:
        return [0, 4, 8, 12, 15]
    return sorted({0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1})


def parse_layer_spec(raw: str | None, num_layers: int, default: str = "auto") -> list[int]:
    spec = (raw or default).strip().lower()
    if spec in {"all", "*"}:
        values = list(range(num_layers))
    elif spec in {"auto", "probe"}:
        values = default_probe_layers(num_layers)
    else:
        values = parse_int_list(raw or "")
    invalid = [layer for layer in values if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} decoder layers: {invalid}")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one item")
    return values


def active_layers_for(policy: str, target_layer: int, num_layers: int) -> set[int]:
    if policy == "single_layer":
        return {target_layer}
    if policy == "prefix":
        return set(range(target_layer + 1))
    if policy == "full":
        return set(range(num_layers))
    raise ValueError(f"Unsupported propagation policy: {policy}")


def tensor_bits_last_or_all(tensor: Any, all_tokens: bool) -> tuple[np.ndarray, tuple[int, int]]:
    selected = tensor[0] if all_tokens else tensor[0, -1:, :]
    bits = torch_bf16_bits(selected)
    return bits, (int(selected.shape[0]), int(selected.shape[1]))


def emit_tensor_delta(
    writer: csv.DictWriter,
    bucket: SummaryBuckets,
    prompt_index: int,
    decode_step: int,
    policy: str,
    target_layer: int,
    tensor_name: str,
    projection: str,
    gpu_tensor: Any,
    pim_tensor: Any,
    all_tokens: bool,
) -> None:
    gpu_bits, shape = tensor_bits_last_or_all(gpu_tensor, all_tokens)
    pim_bits, _shape = tensor_bits_last_or_all(pim_tensor, all_tokens)
    if gpu_bits.shape != pim_bits.shape:
        raise RuntimeError(f"Tensor shape mismatch for {tensor_name}: {gpu_bits.shape} != {pim_bits.shape}")
    gpu_values = bf16_values(gpu_bits)
    pim_values = bf16_values(pim_bits)
    eps = 1.0e-30
    num_tokens, width = shape
    for token_index in range(num_tokens):
        for element_index in range(width):
            gpu_value = float(gpu_values[token_index, element_index])
            pim_value = float(pim_values[token_index, element_index])
            delta = gpu_value - pim_value
            abs_delta = abs(delta)
            row = {
                "prompt_index": prompt_index,
                "decode_step": decode_step,
                "propagation_policy": policy,
                "target_layer": target_layer,
                "tensor_name": tensor_name,
                "projection": projection,
                "token_index": token_index,
                "element_index": element_index,
                "gpu_bf16_hex": hex16(int(gpu_bits[token_index, element_index])),
                "pim_propagated_bf16_hex": hex16(int(pim_bits[token_index, element_index])),
                "gpu_value": f"{gpu_value:.10g}",
                "pim_propagated_value": f"{pim_value:.10g}",
                "delta_gpu_minus_pim": f"{delta:.10g}",
                "abs_delta": f"{abs_delta:.10g}",
                "relative_delta": f"{abs_delta / max(abs(gpu_value), eps):.10g}",
            }
            writer.writerow(row)
            bucket.add(row, delta)


def softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def logits_summary(
    tokenizer: Any,
    prompt_index: int,
    decode_step: int,
    policy: str,
    target_layer: int,
    gpu_logits_tensor: Any,
    pim_logits_tensor: Any,
) -> dict[str, Any]:
    gpu_logits = gpu_logits_tensor[0, -1].detach().float().cpu().numpy()
    pim_logits = pim_logits_tensor[0, -1].detach().float().cpu().numpy()
    delta = gpu_logits - pim_logits
    abs_delta = np.abs(delta)
    gpu_top5 = np.argsort(gpu_logits)[-5:][::-1].astype(int).tolist()
    pim_top5 = np.argsort(pim_logits)[-5:][::-1].astype(int).tolist()
    gpu_probs = softmax_np(gpu_logits.astype(np.float64))
    pim_probs = softmax_np(pim_logits.astype(np.float64))
    eps = 1.0e-30
    kl = float(np.sum(gpu_probs * (np.log(gpu_probs + eps) - np.log(pim_probs + eps))))
    mean_prob = 0.5 * (gpu_probs + pim_probs)
    js = 0.5 * float(np.sum(gpu_probs * (np.log(gpu_probs + eps) - np.log(mean_prob + eps))))
    js += 0.5 * float(np.sum(pim_probs * (np.log(pim_probs + eps) - np.log(mean_prob + eps))))
    gpu_top1 = int(gpu_top5[0])
    pim_top1 = int(pim_top5[0])
    return {
        "prompt_index": prompt_index,
        "decode_step": decode_step,
        "propagation_policy": policy,
        "target_layer": target_layer,
        "top1_match": gpu_top1 == pim_top1,
        "gpu_top1_id": gpu_top1,
        "pim_top1_id": pim_top1,
        "gpu_top1_token": tokenizer.decode([gpu_top1]),
        "pim_top1_token": tokenizer.decode([pim_top1]),
        "top5_overlap": len(set(gpu_top5) & set(pim_top5)),
        "max_abs_logit_delta": float(np.max(abs_delta)),
        "abs_logit_delta_p99": float(np.percentile(abs_delta, 99)),
        "next_token_probability_delta": float(gpu_probs[gpu_top1] - pim_probs[gpu_top1]),
        "kl_gpu_to_pim": kl,
        "js_divergence": js,
    }


def plot_results(layer_rows: list[dict[str, Any]], logits_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = [row for row in layer_rows if row["tensor_name"] == "final_hidden"]
    policies = sorted({str(row["propagation_policy"]) for row in final_rows})

    plt.figure(figsize=(9, 5))
    for policy in policies:
        rows = sorted([row for row in final_rows if row["propagation_policy"] == policy], key=lambda row: int(row["target_layer"]))
        plt.plot(
            [int(row["target_layer"]) for row in rows],
            [float(row["abs_delta_max"]) for row in rows],
            marker="o",
            linewidth=2,
            label=policy,
        )
    plt.xlabel("Target layer")
    plt.ylabel("Final hidden max abs delta")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "llama32_propagation_abs_delta_by_layer.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    for policy in policies:
        rows = sorted([row for row in final_rows if row["propagation_policy"] == policy], key=lambda row: int(row["target_layer"]))
        plt.plot(
            [int(row["target_layer"]) for row in rows],
            [100.0 * float(row["nonzero_delta_ratio"]) for row in rows],
            marker="o",
            linewidth=2,
            label=policy,
        )
    plt.xlabel("Target layer")
    plt.ylabel("Final hidden non-zero delta (%)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "llama32_propagation_nonzero_ratio_by_layer.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    for policy in sorted({str(row["propagation_policy"]) for row in logits_rows}):
        rows = sorted([row for row in logits_rows if row["propagation_policy"] == policy], key=lambda row: int(row["target_layer"]))
        plt.plot(
            [int(row["target_layer"]) for row in rows],
            [float(row["max_abs_logit_delta"]) for row in rows],
            marker="o",
            linewidth=2,
            label=policy,
        )
    plt.xlabel("Target layer")
    plt.ylabel("Max abs logit delta")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "llama32_propagation_logit_delta.png", dpi=600, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="auto", help="Comma-separated layers, 'auto', or 'all'")
    parser.add_argument("--policies", default="prefix")
    parser.add_argument("--prompt-indices", default="0")
    parser.add_argument("--decode-steps", type=int, default=1, help="Metadata field for pilot; current implementation evaluates prompt logits.")
    parser.add_argument("--all-tokens", action="store_true", help="Compare every prompt token hidden-state instead of only last token")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face repo id for online download, metadata, and cache keys")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Use a local HF model directory. If omitted, use online mode with the default weight cache.",
    )
    parser.add_argument("--ramulator", type=Path, default=DEFAULT_RAMULATOR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require existing local HF model files")
    parser.add_argument("--cache-root", type=Path, default=PIM_CACHE_ROOT)
    parser.add_argument("--mode2-token-batch", action="store_true", help="Run all tokens of one projection in one Mode2 invocation")
    parser.add_argument("--pim-workers", type=int, default=1, help="Parallel ramulator workers for independent token/chunk jobs")
    parser.add_argument(
        "--batch-token-chunk-size",
        type=int,
        default=0,
        help="Split batch-token Mode2 into chunks; 0 means one batch per projection call",
    )
    args = parser.parse_args()
    model_dir = resolve_model_dir(args.model_id, args.model_dir)

    if not args.ramulator.exists():
        raise FileNotFoundError(f"ramulator2 not found: {args.ramulator}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_model_available(offline=args.offline, model_id=args.model_id, model_dir=model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Experiment 3 requires CUDA for the GPU baseline and forward graph")

    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PROP_ROOT.mkdir(parents=True, exist_ok=True)
    runner = PIMRunner(
        args.ramulator,
        args.cache_root,
        resume=args.resume,
        keep_artifacts=args.keep_run_artifacts,
        model_id=args.model_id,
        batch_tokens=args.mode2_token_batch,
        parallel_workers=args.pim_workers,
        batch_token_chunk_size=args.batch_token_chunk_size,
    )
    policies = parse_str_list(args.policies)
    prompt_indices = parse_int_list(args.prompt_indices)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model.to(device)
    model.eval()
    num_layers = decoder_layer_count(model)
    layers = parse_layer_spec(args.layers, num_layers, default="auto")
    wrap_model_projections(model, runner)

    element_path = SUMMARY_DIR / "propagation_delta_elementwise.csv"
    logits_rows: list[dict[str, Any]] = []
    bucket = SummaryBuckets(
        [
            "prompt_index",
            "decode_step",
            "propagation_policy",
            "target_layer",
            "tensor_name",
            "projection",
        ]
    )

    try:
        with element_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ELEMENT_FIELDS)
            writer.writeheader()

            with torch.inference_mode():
                for prompt_index in prompt_indices:
                    prompt = PROMPTS[prompt_index]
                    encoded = tokenizer(prompt, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    runner.set_context("disabled", -1, set(), prompt_index, 0)
                    baseline = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    for policy in policies:
                        for target_layer in layers:
                            active_layers = active_layers_for(policy, target_layer, num_layers)
                            runner.set_context(policy, target_layer, active_layers, prompt_index, 0)
                            pim_out = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True,
                            )

                            emit_tensor_delta(
                                writer,
                                bucket,
                                prompt_index,
                                0,
                                policy,
                                target_layer,
                                "target_layer_input",
                                "",
                                baseline.hidden_states[target_layer],
                                pim_out.hidden_states[target_layer],
                                args.all_tokens,
                            )
                            emit_tensor_delta(
                                writer,
                                bucket,
                                prompt_index,
                                0,
                                policy,
                                target_layer,
                                "target_layer_output",
                                "",
                                baseline.hidden_states[target_layer + 1],
                                pim_out.hidden_states[target_layer + 1],
                                args.all_tokens,
                            )
                            emit_tensor_delta(
                                writer,
                                bucket,
                                prompt_index,
                                0,
                                policy,
                                target_layer,
                                "final_hidden",
                                "",
                                baseline.hidden_states[-1],
                                pim_out.hidden_states[-1],
                                args.all_tokens,
                            )
                            logits_rows.append(
                                logits_summary(
                                    tokenizer,
                                    prompt_index,
                                    0,
                                    policy,
                                    target_layer,
                                    baseline.logits,
                                    pim_out.logits,
                                )
                            )
                            f.flush()
                            print(
                                f"[propagation] prompt={prompt_index} policy={policy} "
                                f"target_layer={target_layer} done "
                                f"(mode2_runs={runner.runs}, cache_hits={runner.cache_hits})",
                                flush=True,
                            )
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced

    layer_rows = bucket.rows()
    write_csv(SUMMARY_DIR / "propagation_delta_by_layer.csv", SUMMARY_FIELDS, layer_rows)
    write_csv(SUMMARY_DIR / "propagation_delta_by_tensor.csv", SUMMARY_FIELDS, layer_rows)
    write_csv(SUMMARY_DIR / "propagation_logits_summary.csv", LOGIT_FIELDS, logits_rows)
    plot_results(layer_rows, logits_rows)

    metadata = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "num_layers": num_layers,
        "prompt_indices": prompt_indices,
        "prompts": [PROMPTS[idx] for idx in prompt_indices],
        "layers": layers,
        "policies": policies,
        "all_tokens": args.all_tokens,
        "output_stage": "pre_activation",
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "allow_bf16_reduced_precision_reduction": False,
        "mode2_runs": runner.runs,
        "mode2_batch_runs": runner.batch_runs,
        "mode2_gemv_calls": runner.gemv_calls,
        "cache_hits": runner.cache_hits,
        "mode2_total_wall_time_s": runner.total_wall_time_s,
        "mode2_token_batch": runner.batch_tokens,
        "pim_workers": runner.parallel_workers,
        "batch_token_chunk_size": runner.batch_token_chunk_size,
        "ramulator": str(args.ramulator),
        "cache_root": str(args.cache_root),
        "result_files": {
            "elementwise": str(element_path),
            "by_layer": str(SUMMARY_DIR / "propagation_delta_by_layer.csv"),
            "by_tensor": str(SUMMARY_DIR / "propagation_delta_by_tensor.csv"),
            "logits": str(SUMMARY_DIR / "propagation_logits_summary.csv"),
        },
    }
    (RESULT_ROOT / "artifacts" / "propagation_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "artifacts" / "propagation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"result_root: {RESULT_ROOT}")
    print(f"mode2_runs: {runner.runs}")
    print(f"mode2_batch_runs: {runner.batch_runs}")
    print(f"mode2_gemv_calls: {runner.gemv_calls}")
    print(f"cache_hits: {runner.cache_hits}")
    print(f"mode2_total_wall_time_s: {runner.total_wall_time_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
