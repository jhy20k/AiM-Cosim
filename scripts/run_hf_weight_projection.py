#!/usr/bin/env python3
"""Run AiMRTL functional validation using HF or random projection values.

The script can read safetensors directly or generate deterministic random BF16
values. It then emits one AIMD file, an AiM ISR trace, a CPU BF16 reference,
runs Mode2, and compares the RTL result against the CPU reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any

import numpy as np


AIMD_MAGIC = 0x41494D44
NUM_BANKS = 16
MAX_ROWS = 16384
MIN_TILES_PER_ROW = 128
ELEMS_PER_TILE = 16
NUM_GPRS = 32
NUM_CHANNELS = 32
CMD_RDMAC16 = 19
CMD_RDAF16 = 20


def output_location(row: int) -> tuple[int, int, int, int]:
    """Map a conceptual output row to (group, channel, bank, physical_row)."""
    group = row // (NUM_CHANNELS * NUM_BANKS)
    rem = row % (NUM_CHANNELS * NUM_BANKS)
    channel = rem // NUM_BANKS
    bank = rem % NUM_BANKS
    physical_row = group * NUM_CHANNELS + channel
    return group, channel, bank, physical_row


def ramulator_env(ramulator: str) -> dict[str, str]:
    env = os.environ.copy()
    lib_dir = str(Path(ramulator).resolve().parent)
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = lib_dir if not current else f"{lib_dir}:{current}"
    return env


PROJECTION_CANDIDATES = {
    "self_attn.q_proj": [
        ("self_attn.q_proj.weight", None),
        ("self_attn.qkv_proj.weight", "qkv:q"),
    ],
    "self_attn.k_proj": [
        ("self_attn.k_proj.weight", None),
        ("self_attn.qkv_proj.weight", "qkv:k"),
    ],
    "self_attn.v_proj": [
        ("self_attn.v_proj.weight", None),
        ("self_attn.qkv_proj.weight", "qkv:v"),
    ],
    "self_attn.o_proj": [
        ("self_attn.o_proj.weight", None),
    ],
    "mlp.gate_proj": [
        ("mlp.gate_proj.weight", None),
        ("mlp.gate_up_proj.weight", "gate_up:gate"),
    ],
    "mlp.up_proj": [
        ("mlp.up_proj.weight", None),
        ("mlp.gate_up_proj.weight", "gate_up:up"),
    ],
    "mlp.down_proj": [
        ("mlp.down_proj.weight", None),
    ],
    "q/o": [
        ("self_attn.q_proj.weight", None),
        ("self_attn.qkv_proj.weight", "qkv:q"),
        ("self_attn.o_proj.weight", None),
    ],
    "k/v": [
        ("self_attn.k_proj.weight", None),
        ("self_attn.qkv_proj.weight", "qkv:k"),
        ("self_attn.v_proj.weight", None),
    ],
    "gate/up": [
        ("mlp.gate_proj.weight", None),
        ("mlp.gate_up_proj.weight", "gate_up:gate"),
        ("mlp.up_proj.weight", None),
    ],
    "down": [
        ("mlp.down_proj.weight", None),
    ],
}


def bf16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header_len, header


def tensor_index(model_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index_path = model_dir / "model.safetensors.index.json"
    file_map: dict[str, str] = {}
    if index_path.exists():
        with index_path.open() as f:
            data = json.load(f)
        file_map = data.get("weight_map", {})

    tensors: dict[str, tuple[Path, dict[str, Any]]] = {}
    if file_map:
        for name, filename in file_map.items():
            path = model_dir / filename
            if not path.exists():
                continue
            _, header = read_header(path)
            meta = header.get(name)
            if meta:
                tensors[name] = (path, meta)
    else:
        for path in sorted(model_dir.glob("*.safetensors")):
            _, header = read_header(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                tensors[name] = (path, meta)
    return tensors


def read_tensor(path: Path, name: str, meta: dict[str, Any]) -> np.ndarray:
    header_len, _ = read_header(path)
    start, end = meta["data_offsets"]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    with path.open("rb") as f:
        f.seek(8 + header_len + start)
        raw = f.read(end - start)

    if dtype == "BF16":
        return np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
    if dtype == "F32":
        return float32_to_bf16(np.frombuffer(raw, dtype=np.float32).reshape(shape))
    if dtype == "F16":
        return float32_to_bf16(np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape))
    raise ValueError(f"Unsupported dtype for {name}: {dtype}")


def choose_tensor(tensors: dict[str, tuple[Path, dict[str, Any]]], group: str, layer: int) -> tuple[str, str | None]:
    layer_prefix = f"model.layers.{layer}."
    candidates = PROJECTION_CANDIDATES[group]
    for suffix, split in candidates:
        exact = layer_prefix + suffix
        if exact in tensors:
            return exact, split
    for suffix, split in candidates:
        matches = [name for name in tensors if name.endswith(suffix)]
        if matches:
            return sorted(matches)[0], split
    available = "\n".join(sorted(tensors)[:80])
    raise KeyError(f"No tensor found for projection group {group}. First available tensors:\n{available}")


def split_tensor(weight: np.ndarray, split: str | None, q: int, r: int, intermediate: int | None) -> np.ndarray:
    if split is None:
        return weight
    if split == "qkv:q":
        return weight[:r, :]
    if split == "qkv:k":
        return weight[q : q + r, :]
    if split == "qkv:v":
        return weight[q + r : q + 2 * r, :]
    if split == "gate_up:gate":
        rows = intermediate if intermediate is not None else r
        return weight[:rows, :]
    if split == "gate_up:up":
        rows = intermediate if intermediate is not None else r
        return weight[rows : 2 * rows, :]
    raise ValueError(f"Unsupported split policy: {split}")


def load_projection(model_dir: Path, group: str, q: int, r: int, layer: int, tensor_name: str | None,
                    intermediate: int | None) -> tuple[str, np.ndarray]:
    tensors = tensor_index(model_dir)
    split = None
    if tensor_name is None:
        tensor_name, split = choose_tensor(tensors, group, layer)
    elif ":" in tensor_name:
        tensor_name, split = tensor_name.split(":", 1)

    path, meta = tensors[tensor_name]
    raw = read_tensor(path, tensor_name, meta)
    weight = split_tensor(raw, split, q, r, intermediate)

    if weight.shape == (r, q):
        return tensor_name + (f":{split}" if split else ""), weight
    if weight.shape == (q, r):
        return tensor_name + (f":{split}:transpose" if split else ":transpose"), weight.T.copy()
    raise ValueError(f"Tensor {tensor_name} has shape {weight.shape}, expected {(r, q)} or {(q, r)}")


def generate_activation(q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return float32_to_bf16(rng.standard_normal(q).astype(np.float32))


def load_prompt_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{path} must contain a string list field named 'prompts'")
    max_new_tokens = int(data.get("max_new_tokens", 0))
    return {"max_new_tokens": max_new_tokens, "prompts": prompts}


def prompt_digest(prompt_config: dict[str, Any], seed: int) -> str:
    payload = json.dumps(
        {
            "max_new_tokens": prompt_config.get("max_new_tokens", 0),
            "prompts": prompt_config.get("prompts", []),
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def generate_prompt_hash_activation(q: int, seed: int, prompt_config: dict[str, Any]) -> np.ndarray:
    digest = prompt_digest(prompt_config, seed)
    rng_seed = int(digest[:16], 16) ^ (seed & 0xFFFFFFFF)
    rng = np.random.default_rng(rng_seed)
    return float32_to_bf16(rng.standard_normal(q).astype(np.float32))


def activation_from_args(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.activation_source == "random_bf16":
        return generate_activation(args.q, args.seed), {
            "activation_source": "random_bf16",
            "prompt_file": "",
            "max_new_tokens": "",
            "prompt_count": "",
            "prompt_digest": "",
        }
    if not args.prompt_file:
        raise ValueError("--prompt-file is required when --activation-source=prompt_hash_bf16")
    prompt_path = Path(args.prompt_file)
    prompt_config = load_prompt_config(prompt_path)
    digest = prompt_digest(prompt_config, args.seed)
    return generate_prompt_hash_activation(args.q, args.seed, prompt_config), {
        "activation_source": "prompt_hash_bf16",
        "prompt_file": str(prompt_path),
        "max_new_tokens": prompt_config["max_new_tokens"],
        "prompt_count": len(prompt_config["prompts"]),
        "prompt_digest": digest,
    }


def bf16_ordered_int(bits: int) -> int:
    bits &= 0xFFFF
    if bits & 0x7FFF == 0:
        return 0x8000
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def bf16_ulp_distance(left: int, right: int) -> int:
    return abs(bf16_ordered_int(left) - bf16_ordered_int(right))


def generate_random_weight(q: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return float32_to_bf16(rng.standard_normal((r, q)).astype(np.float32))


def write_aimd(path: Path, weight_rq_bf16: np.ndarray, activation_bf16: np.ndarray, q: int, r: int,
               dense_rows: bool = False, channel_layout: str = "broadcast") -> tuple[int, int]:
    if channel_layout != "row_sharded" and r > MAX_ROWS:
        raise ValueError(f"R={r} exceeds MAX_ROWS={MAX_ROWS}; use R chunking or expand memory rows")
    if q % ELEMS_PER_TILE != 0:
        raise ValueError(f"Q={q} must be multiple of {ELEMS_PER_TILE}")

    tiles_q = q // ELEMS_PER_TILE
    tiles_per_row = max(MIN_TILES_PER_ROW, tiles_q)
    zero_tile = np.zeros(ELEMS_PER_TILE, dtype=np.uint16)
    if channel_layout == "row_sharded":
        active_rows = max((output_location(row)[3] for row in range(r)), default=-1) + 1
        if active_rows > MAX_ROWS:
            raise ValueError(
                f"row-sharded R={r} requires {active_rows} physical rows, "
                f"exceeding MAX_ROWS={MAX_ROWS}"
            )
        rows_per_bank = MAX_ROWS if dense_rows else active_rows
    else:
        rows_per_bank = MAX_ROWS if dense_rows else r
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(struct.pack("<IIII", AIMD_MAGIC, NUM_BANKS, rows_per_bank, tiles_per_row))
        for bank in range(NUM_BANKS):
            for row in range(rows_per_bank):
                if channel_layout == "row_sharded":
                    group = row // NUM_CHANNELS
                    channel = row % NUM_CHANNELS
                    output_row = group * (NUM_CHANNELS * NUM_BANKS) + channel * NUM_BANKS + bank
                else:
                    output_row = row
                if output_row < r:
                    row_data = weight_rq_bf16[output_row]
                    for t in range(tiles_per_row):
                        if t < tiles_q:
                            f.write(row_data[t * ELEMS_PER_TILE : (t + 1) * ELEMS_PER_TILE].tobytes())
                        else:
                            f.write(zero_tile.tobytes())
                else:
                    f.write(bytes(tiles_per_row * ELEMS_PER_TILE * 2))

        for t in range(tiles_per_row):
            if t < tiles_q:
                f.write(activation_bf16[t * ELEMS_PER_TILE : (t + 1) * ELEMS_PER_TILE].tobytes())
            else:
                f.write(zero_tile.tobytes())

        for g in range(NUM_GPRS):
            if g < tiles_q:
                f.write(activation_bf16[g * ELEMS_PER_TILE : (g + 1) * ELEMS_PER_TILE].tobytes())
            else:
                f.write(zero_tile.tobytes())

        f.write(bytes(NUM_BANKS * 2))

    return tiles_per_row, rows_per_bank


def row_shard_channel(row: int, num_channels: int = NUM_CHANNELS) -> int:
    if num_channels != NUM_CHANNELS:
        return row % num_channels
    return output_location(row)[1]


def channel_mask_for(channel: int) -> str:
    # AiMDRAMSystem::FindFirstChannelIndex maps mask bit i to channel 31 - i.
    # Use the inverse mapping so logical channel N is actually sent to CHN.
    return f"0x{1 << (NUM_CHANNELS - 1 - channel):08x}"


def write_trace(
    path: Path,
    q: int,
    r: int,
    channel_layout: str = "broadcast",
    output_stage: str = "post_activation",
) -> None:
    if output_stage not in {"post_activation", "pre_activation"}:
        raise ValueError(f"Unsupported output stage: {output_stage}")
    opsize = q // ELEMS_PER_TILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# AiM HF-weight GEMV trace: P=1 Q={q} R={r}\n")
        f.write(f"# opsize={opsize}, channel_layout={channel_layout}, output_stage={output_stage}\n")
        if channel_layout == "broadcast":
            for row in range(r):
                channel_mask = "0xffffffff"
                f.write(f"# output_row={row}, channel_mask={channel_mask}\n")
                f.write(f"AiM WR_BIAS {opsize} {channel_mask}\n")
                f.write(f"AiM WR_GB {opsize} 0 {channel_mask}\n")
                f.write(f"AiM MAC_ABK {opsize} {channel_mask} {row}\n")
                if output_stage == "post_activation":
                    f.write(f"AiM AF {channel_mask}\n")
                    f.write(f"AiM RD_AF {opsize} {channel_mask}\n")
                else:
                    f.write(f"AiM RD_MAC {opsize} {channel_mask}\n")
        elif channel_layout == "row_sharded":
            physical_rows_by_group: dict[int, list[int]] = {}
            for row in range(r):
                group, _channel, _bank, physical_row = output_location(row)
                physical_rows_by_group.setdefault(group, [])
                if physical_row not in physical_rows_by_group[group]:
                    physical_rows_by_group[group].append(physical_row)

            for group in sorted(physical_rows_by_group):
                physical_rows = sorted(physical_rows_by_group[group])
                f.write(f"# row_shard_group={group}, physical_rows={physical_rows[0]}..{physical_rows[-1]}\n")
                for physical_row in physical_rows:
                    channel = physical_row % NUM_CHANNELS
                    channel_mask = channel_mask_for(channel)
                    first_output = group * (NUM_CHANNELS * NUM_BANKS) + channel * NUM_BANKS
                    last_output = min(first_output + NUM_BANKS, r) - 1
                    f.write(
                        f"# physical_row={physical_row}, channel={channel}, "
                        f"output_rows={first_output}..{last_output}, "
                        f"channel_mask={channel_mask}\n"
                    )
                    f.write(f"AiM WR_BIAS {opsize} {channel_mask}\n")
                for physical_row in physical_rows:
                    channel_mask = channel_mask_for(physical_row % NUM_CHANNELS)
                    f.write(f"AiM WR_GB {opsize} 0 {channel_mask}\n")
                for physical_row in physical_rows:
                    channel_mask = channel_mask_for(physical_row % NUM_CHANNELS)
                    f.write(f"AiM MAC_ABK {opsize} {channel_mask} {physical_row}\n")
                if output_stage == "post_activation":
                    for physical_row in physical_rows:
                        channel_mask = channel_mask_for(physical_row % NUM_CHANNELS)
                        f.write(f"AiM AF {channel_mask}\n")
                    for physical_row in physical_rows:
                        channel_mask = channel_mask_for(physical_row % NUM_CHANNELS)
                        f.write(f"AiM RD_AF {opsize} {channel_mask}\n")
                else:
                    for physical_row in physical_rows:
                        channel_mask = channel_mask_for(physical_row % NUM_CHANNELS)
                        f.write(f"AiM RD_MAC {opsize} {channel_mask}\n")
        else:
            raise ValueError(f"Unsupported channel layout: {channel_layout}")
        f.write("AiM ISR_EOC\n")


def write_config(
    path: Path,
    aimd_path: Path,
    result_base: Path,
    tiles_per_row: int,
    seed: int,
    channel_layout: str,
) -> None:
    shard_load = "      load_row_shard_mod: 32\n" if channel_layout == "row_sharded" else ""
    path.write_text(
        f"""Frontend:
  impl: AiMTrace
  clock_ratio: 1
  Translation:
    impl: NoTranslation
    max_addr: 2147483648
MemorySystem:
  impl: AiMDRAM
  clock_ratio: 1
  DRAM:
    impl: GDDR6
    org:
      preset: GDDR6_AiM_org
    timing:
      preset: GDDR6_AiM_timing
  Controller:
    impl: AiMRTL
    clock_ratio: 1
    memory_manager:
      num_banks: 16
      max_rows: {MAX_ROWS}
      tiles_per_row: {tiles_per_row}
      init_mode: "file"
      init_seed: {seed}
      data_file: "{aimd_path}"
{shard_load.rstrip()}
    verilator:
      enabled: true
      result_log: "{result_base}.csv"
      vcd_trace: false
      vcd_file: "aim_rtl.vcd"
      logging:
        buffer_size: 65536
        result_level: full
        timing_level: off
        rdaf16_bank0_only: false
    Scheduler:
      impl: FRFCFS
    RefreshManager:
      impl: AllBank
    plugins:
  AddrMapper:
    impl: RoBaRaCoCh
""",
        encoding="utf-8",
    )


def parse_rtl_events(path: Path, target_cmd: int = CMD_RDAF16) -> list[list[int | None]]:
    events: list[list[int | None]] = []
    current: list[int | None] | None = None
    last_bank = -1
    with path.open() as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            if int(row[1]) != target_cmd:
                continue
            bank = int(row[2])
            if bank < 0 or bank >= NUM_BANKS:
                continue
            if current is None or bank <= last_bank:
                if current is not None:
                    events.append(current)
                current = [None] * NUM_BANKS
            first = row[5].strip().split()[0]
            current[bank] = int(first, 16) if first != "0" else 0
            last_bank = bank
    if current is not None:
        events.append(current)
    return events


def parse_rtl_csv(path: Path, target_cmd: int = CMD_RDAF16) -> list[int]:
    out: list[int] = []
    for event in parse_rtl_events(path, target_cmd):
        value = event[0]
        if value is not None:
            out.append(value)
    return out


def parse_rtl_results(
    result_base: Path,
    r: int,
    channel_layout: str,
    output_stage: str = "post_activation",
) -> list[int]:
    target_cmd = CMD_RDAF16 if output_stage == "post_activation" else CMD_RDMAC16
    if channel_layout == "broadcast":
        return parse_rtl_csv(Path(f"{result_base}_ch0.csv"), target_cmd)
    if channel_layout != "row_sharded":
        raise ValueError(f"Unsupported channel layout: {channel_layout}")

    by_channel: dict[int, list[list[int | None]]] = {}
    for ch in range(NUM_CHANNELS):
        csv_path = Path(f"{result_base}_ch{ch}.csv")
        by_channel[ch] = parse_rtl_events(csv_path, target_cmd) if csv_path.exists() else []

    merged: list[int] = []
    for row in range(r):
        group, ch, bank, _physical_row = output_location(row)
        events = by_channel[ch]
        if group >= len(events):
            raise RuntimeError(
                f"Missing row-sharded RTL result for output row {row} "
                f"(channel {ch}, local result index {group})"
            )
        value = events[group][bank]
        if value is None:
            raise RuntimeError(
                f"Missing row-sharded RTL bank result for output row {row} "
                f"(channel {ch}, bank {bank}, local result index {group})"
            )
        merged.append(value)
    return merged


def cpu_reference(
    weight_rq_bf16: np.ndarray,
    activation_bf16: np.ndarray,
    output_stage: str = "post_activation",
) -> list[int]:
    w = bf16_to_float32(weight_rq_bf16.reshape(-1)).reshape(weight_rq_bf16.shape)
    a = bf16_to_float32(activation_bf16)
    y = w @ a
    if output_stage == "post_activation":
        y = np.maximum(y, 0)
    return [int(v) for v in float32_to_bf16(y)]


def write_report(
    path: Path,
    metadata: dict[str, Any],
    rtl_hex: list[int],
    cpu_hex: list[int],
    channel_layout: str,
) -> dict[str, Any]:
    n = min(len(rtl_hex), len(cpu_hex))
    diffs = [bf16_ulp_distance(rtl_hex[i], cpu_hex[i]) for i in range(n)]
    stats = {
        "n": n,
        "exact": sum(1 for d in diffs if d == 0),
        "within_1": sum(1 for d in diffs if d <= 1),
        "within_2": sum(1 for d in diffs if d <= 2),
        "max_ulp": max(diffs) if diffs else 0,
        "avg_ulp": float(np.mean(diffs)) if diffs else 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# HF-weight AiMRTL functional report\n")
        for k, v in metadata.items():
            f.write(f"# {k}: {v}\n")
        f.write(f"# total_output_elements: {stats['n']}\n")
        f.write(f"# exact: {stats['exact']}/{stats['n']}\n")
        f.write(f"# within_1_ulp: {stats['within_1']}/{stats['n']}\n")
        f.write(f"# within_2_ulp: {stats['within_2']}/{stats['n']}\n")
        f.write(f"# max_ulp: {stats['max_ulp']}\n")
        f.write(f"# avg_ulp: {stats['avg_ulp']:.6f}\n")
        f.write("row,channel,bank,physical_row,rtl_hex,cpu_hex,ulp_diff\n")
        for i in range(n):
            if channel_layout == "row_sharded":
                _group, channel, bank, physical_row = output_location(i)
            else:
                channel, bank, physical_row = 0, 0, i
            ulp_diff = bf16_ulp_distance(rtl_hex[i], cpu_hex[i])
            f.write(
                f"{i},{channel},{bank},{physical_row},"
                f"{rtl_hex[i]:04x},{cpu_hex[i]:04x},{ulp_diff}\n"
            )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one HF projection weight through AiMRTL")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-dir", help="Downloaded model directory. Defaults to weight/<repo_id with / replaced>")
    parser.add_argument("--projection-group", required=True, choices=sorted(PROJECTION_CANDIDATES))
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--tensor-name", help="Override tensor name. Use name:split for fused tensors.")
    parser.add_argument("--row-start", type=int, default=0, help="Start row for R-dimension chunk execution")
    parser.add_argument("--row-count", type=int, help="Number of R rows for chunk execution")
    parser.add_argument(
        "--weight-source",
        choices=["hf_safetensors", "random_bf16"],
        default="hf_safetensors",
        help="Use actual HF safetensors weight or deterministic random BF16 values",
    )
    parser.add_argument("--weight-seed", type=int, default=314159, help="Seed used when --weight-source=random_bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-source",
        choices=["random_bf16", "prompt_hash_bf16"],
        default="random_bf16",
        help="Generate the GEMV input vector from the numeric seed or from a prompt-config digest",
    )
    parser.add_argument(
        "--prompt-file",
        help="JSON file with max_new_tokens and prompts, used by --activation-source=prompt_hash_bf16",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ramulator", default="build_mag32/extern/aim_simulator/ramulator2")
    parser.add_argument("--keep-csv", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dense-aimd", action="store_true", help="Emit legacy dense AIMD with MAX_ROWS rows per bank")
    parser.add_argument(
        "--output-stage",
        choices=["post_activation", "pre_activation"],
        default="post_activation",
        help="Read post-activation RD_AF output or pre-activation RD_MAC output",
    )
    parser.add_argument(
        "--channel-layout",
        choices=["broadcast", "row_sharded"],
        default="broadcast",
        help="Use legacy 32-channel broadcast or distribute output rows across channels",
    )
    parser.add_argument("--run-tag", help="Append a short tag to the run directory name")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path("weight") / safe_name(args.repo_id)
    if args.weight_source == "random_bf16":
        tensor_name = f"random_bf16_weight_seed{args.weight_seed}"
        weight = generate_random_weight(args.q, args.r, args.weight_seed)
        value_policy = "random_bf16_shape_preserving"
    else:
        tensor_name, weight = load_projection(
            model_dir,
            args.projection_group,
            args.q,
            args.r,
            args.layer,
            args.tensor_name,
            args.intermediate,
        )
        value_policy = "real_hf_safetensors"

    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.row_start or args.row_count is not None:
        row_end = args.r if args.row_count is None else args.row_start + args.row_count
        if row_end > args.r:
            raise ValueError(f"row chunk [{args.row_start}, {row_end}) exceeds R={args.r}")
        weight = weight[args.row_start:row_end].copy()
        tensor_name = f"{tensor_name}:rows[{args.row_start}:{row_end}]"

    effective_r = int(weight.shape[0])
    out_dir = Path(args.output_dir)
    shape = f"p1_q{args.q}_r{effective_r}"
    group_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.projection_group)
    weight_suffix = ""
    if args.weight_source == "random_bf16":
        weight_suffix = f"_random_bf16_wseed{args.weight_seed}"
    tag_suffix = ""
    if args.run_tag:
        clean_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.run_tag)
        tag_suffix = f"_{clean_tag}"
    if args.output_stage == "pre_activation":
        tag_suffix += "_pre_activation"
    run_id = f"{safe_name(args.repo_id)}_{group_id}_{shape}_layer{args.layer}_seed{args.seed}{weight_suffix}{tag_suffix}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    activation, activation_metadata = activation_from_args(args)

    aimd_path = run_dir / f"{shape}.aimd"
    trace_path = run_dir / f"{shape}.trace"
    config_path = run_dir / "aim_rtl_file.yaml"
    result_base = run_dir / "rtl_results"
    mode2_out = run_dir / "mode2.txt"

    tiles_per_row, rows_per_bank = write_aimd(
        aimd_path,
        weight,
        activation,
        args.q,
        effective_r,
        args.dense_aimd,
        args.channel_layout,
    )
    write_trace(trace_path, args.q, effective_r, args.channel_layout, args.output_stage)
    write_config(config_path, aimd_path, result_base, tiles_per_row, args.seed, args.channel_layout)

    cpu_hex = cpu_reference(weight, activation, args.output_stage)
    cpu_ref_path = run_dir / "cpu_reference.csv"
    with cpu_ref_path.open("w") as f:
        f.write("row,cpu_hex\n")
        for i, h in enumerate(cpu_hex):
            f.write(f"{i},{h:04x}\n")

    metadata = {
        "repo_id": args.repo_id,
        "model_dir": model_dir,
        "value_policy": value_policy,
        "weight_source": args.weight_source,
        "weight_seed": args.weight_seed if args.weight_source == "random_bf16" else "",
        "projection_group": args.projection_group,
        "tensor_name": tensor_name,
        "layer": args.layer,
        "q": args.q,
        "r": effective_r,
        "full_r": args.r,
        "row_start": args.row_start,
        "row_count": effective_r,
        "shape": shape,
        "seed": args.seed,
        **activation_metadata,
        "output_stage": args.output_stage,
        "result_command": "RDAF16" if args.output_stage == "post_activation" else "RDMAC16",
        "tiles_per_row": tiles_per_row,
        "aimd_rows_per_bank": rows_per_bank,
        "aimd_layout": (
            "dense_max_rows"
            if args.dense_aimd
            else ("active_rows_bank_sharded" if args.channel_layout == "row_sharded" else "active_rows")
        ),
        "channel_layout": args.channel_layout,
        "num_channels": NUM_CHANNELS,
        "num_banks": NUM_BANKS,
        "row_shard_policy": (
            "output_row_to_channel_and_bank"
            if args.channel_layout == "row_sharded"
            else "full_broadcast"
        ),
        "mode2_ramulator": args.ramulator,
        "mode2_speed_options": (
            "selective_row_shard_aimd_load,full_bank_rdaf16,protocol_hidden_scoreboard"
            if args.channel_layout == "row_sharded"
            else "default"
        ),
        "aimd_size_bytes": aimd_path.stat().st_size,
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump({k: str(v) for k, v in metadata.items()}, f, indent=2)

    if args.generate_only:
        print(f"generated: {run_dir}")
        return 0

    start = time.perf_counter()
    proc = subprocess.run(
        [args.ramulator, "-f", str(config_path), "-t", str(trace_path)],
        cwd=Path.cwd(),
        env=ramulator_env(args.ramulator),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_time_s = time.perf_counter() - start
    mode2_out.write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        return proc.returncode

    cycles = "NA"
    for line in proc.stdout.splitlines():
        if "memory_system_cycles:" in line:
            cycles = line.split()[1]
            break

    rtl_hex = parse_rtl_results(result_base, effective_r, args.channel_layout, args.output_stage)
    metadata["memory_system_cycles"] = cycles
    metadata["wall_time_s"] = f"{wall_time_s:.6f}"
    stats = write_report(run_dir / "functional_report.csv", metadata, rtl_hex, cpu_hex, args.channel_layout)

    if not args.keep_csv and args.channel_layout == "broadcast":
        for csv_file in run_dir.glob("rtl_results_ch*.csv"):
            if csv_file.name != "rtl_results_ch0.csv":
                csv_file.unlink()

    print(f"run_dir: {run_dir}")
    print(f"tensor: {tensor_name}")
    print(f"cycles: {cycles}")
    print(f"wall_time_s: {wall_time_s:.3f}")
    print(
        f"exact={stats['exact']}/{stats['n']} "
        f"within_1={stats['within_1']}/{stats['n']} "
        f"max_ulp={stats['max_ulp']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
