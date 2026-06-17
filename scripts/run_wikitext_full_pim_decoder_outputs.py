#!/usr/bin/env python3
"""Collect decoder-layer output deltas from one full PIM-injected forward."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_llama32_propagation_experiment as prop  # noqa: E402


SUMMARY_FIELDS = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "layer",
    "tensor_name",
    "count",
    "delta_mean",
    "abs_delta_mean",
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

ELEMENT_FIELDS = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "layer",
    "tensor_name",
    "token_index",
    "element_index",
    "gpu_bf16_hex",
    "pim_bf16_hex",
    "gpu_value",
    "pim_value",
    "delta_gpu_minus_pim",
    "abs_delta",
    "relative_delta",
]

CACHE_ROOT = (
    ROOT
    / "tests"
    / "result"
    / "llama32_error_accumulation"
    / "runs"
    / "propagation"
    / "pim_cache"
)


def hex16(value: int) -> str:
    return f"0x{int(value) & 0xFFFF:04X}"


def selected_bits(tensor: Any, all_tokens: bool) -> tuple[np.ndarray, tuple[int, int]]:
    selected = tensor[0] if all_tokens else tensor[0, -1:, :]
    bits = prop.torch_bf16_bits(selected)
    return bits, (int(selected.shape[0]), int(selected.shape[1]))


def capture_decoder_outputs(model: Any, output_store: dict[int, Any]) -> list[Any]:
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            output_store[layer_idx] = tensor.detach()

        return hook

    for layer_idx, layer in enumerate(prop.decoder_layers(model)):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    return handles


def summary_from_deltas(key: dict[str, Any], deltas: np.ndarray) -> dict[str, Any]:
    arr = deltas.astype(np.float64).reshape(-1)
    row = prop.summary_row(key, arr.tolist())
    row["abs_delta_mean"] = float(np.mean(np.abs(arr))) if arr.size else 0.0
    return row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_tag(model_id: str, seq_len: int) -> str:
    return f"{prop.clean_name(model_id)}_seq{seq_len}"


def tagged_output_path(path: Path, tag: str) -> Path:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    if tag in stem:
        return path
    return path.with_name(f"{stem}_{tag}{suffix}")


def cuda_ready() -> None:
    import torch

    last_error: Exception | None = None
    for _attempt in range(5):
        try:
            _ = torch.empty(1, device="cuda")
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError("Full PIM decoder-output run requires CUDA") from last_error


def main() -> int:
    script_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dataset-samples", type=int, default=64)
    parser.add_argument("--all-tokens", action="store_true")
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--elementwise-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--model-id", default=prop.DEFAULT_MODEL_ID, help="Hugging Face repo id for online download, metadata, and cache keys")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Use a local HF model directory. If omitted, use online mode with the default weight cache.",
    )
    parser.add_argument("--ramulator", type=Path, default=prop.DEFAULT_RAMULATOR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require existing local HF model and WikiText caches")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--mode2-token-batch", action="store_true", help="Run all tokens of one projection in one Mode2 invocation")
    parser.add_argument("--pim-workers", type=int, default=1, help="Parallel ramulator workers for independent token/chunk jobs")
    parser.add_argument(
        "--batch-token-chunk-size",
        type=int,
        default=0,
        help="Split batch-token Mode2 into chunks; 0 means one batch per projection call",
    )
    args = parser.parse_args()
    model_dir = prop.resolve_model_dir(args.model_id, args.model_dir)
    tag = output_tag(args.model_id, args.seq_len)
    args.summary_output = tagged_output_path(args.summary_output, tag)
    args.elementwise_output = tagged_output_path(args.elementwise_output, tag)
    args.metadata_output = tagged_output_path(args.metadata_output, tag)

    if not args.ramulator.exists():
        raise FileNotFoundError(f"ramulator2 not found: {args.ramulator}")

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_wikitext_prefix_final_hidden import load_wikitext_prompt

    cuda_ready()
    prop.ensure_model_available(offline=args.offline, model_id=args.model_id, model_dir=model_dir)
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt, actual_seq_len = load_wikitext_prompt(tokenizer, args.seq_len, args.dataset_samples)

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model.to("cuda")
    model.eval()
    num_layers = prop.decoder_layer_count(model)
    target_layer = num_layers - 1
    active_layers = set(range(num_layers))

    runner = prop.PIMRunner(
        args.ramulator,
        args.cache_root,
        resume=args.resume,
        keep_artifacts=args.keep_run_artifacts,
        model_id=args.model_id,
        batch_tokens=args.mode2_token_batch,
        parallel_workers=args.pim_workers,
        batch_token_chunk_size=args.batch_token_chunk_size,
    )
    prop.wrap_model_projections(model, runner)

    baseline_outputs: dict[int, Any] = {}
    pim_outputs: dict[int, Any] = {}

    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")

        with torch.inference_mode():
            runner.set_context("disabled", -1, set(), 0, 0)
            handles = capture_decoder_outputs(model, baseline_outputs)
            try:
                _baseline = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()

            runner.set_context("full", target_layer, active_layers, 0, 0)
            handles = capture_decoder_outputs(model, pim_outputs)
            try:
                _pim = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()

        summary_rows: list[dict[str, Any]] = []
        element_rows: list[dict[str, Any]] = []
        eps = 1.0e-30
        for layer_idx in range(num_layers):
            if layer_idx not in baseline_outputs or layer_idx not in pim_outputs:
                raise RuntimeError(f"Missing captured decoder output for layer {layer_idx}")
            gpu_bits, shape = selected_bits(baseline_outputs[layer_idx], args.all_tokens)
            pim_bits, _shape = selected_bits(pim_outputs[layer_idx], args.all_tokens)
            if gpu_bits.shape != pim_bits.shape:
                raise RuntimeError(f"Layer {layer_idx} output shape mismatch: {gpu_bits.shape} != {pim_bits.shape}")

            gpu_values = prop.bf16_values(gpu_bits)
            pim_values = prop.bf16_values(pim_bits)
            deltas = gpu_values.astype(np.float64) - pim_values.astype(np.float64)
            key = {
                "prompt_index": 0,
                "decode_step": 0,
                "propagation_policy": "full",
                "target_layer": target_layer,
                "layer": layer_idx,
                "tensor_name": "decoder_layer_output",
            }
            summary_rows.append(summary_from_deltas(key, deltas))

            num_tokens, width = shape
            for token_index in range(num_tokens):
                for element_index in range(width):
                    gpu_value = float(gpu_values[token_index, element_index])
                    pim_value = float(pim_values[token_index, element_index])
                    delta = gpu_value - pim_value
                    abs_delta = abs(delta)
                    element_rows.append(
                        {
                            "prompt_index": 0,
                            "decode_step": 0,
                            "propagation_policy": "full",
                            "target_layer": target_layer,
                            "layer": layer_idx,
                            "tensor_name": "decoder_layer_output",
                            "token_index": token_index,
                            "element_index": element_index,
                            "gpu_bf16_hex": hex16(int(gpu_bits[token_index, element_index])),
                            "pim_bf16_hex": hex16(int(pim_bits[token_index, element_index])),
                            "gpu_value": f"{gpu_value:.10g}",
                            "pim_value": f"{pim_value:.10g}",
                            "delta_gpu_minus_pim": f"{delta:.10g}",
                            "abs_delta": f"{abs_delta:.10g}",
                            "relative_delta": f"{abs_delta / max(abs(gpu_value), eps):.10g}",
                        }
                    )

        write_csv(args.summary_output, SUMMARY_FIELDS, summary_rows)
        write_csv(args.elementwise_output, ELEMENT_FIELDS, element_rows)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced

    metadata = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "num_layers": num_layers,
        "source": "wikitext-2-raw-v1/test",
        "dataset_samples": args.dataset_samples,
        "requested_seq_len": args.seq_len,
        "actual_seq_len": actual_seq_len,
        "output_tag": tag,
        "all_tokens": args.all_tokens,
        "prompt_preview": prompt[:500],
        "propagation_policy": "full",
        "target_layer": target_layer,
        "active_layers": list(range(num_layers)),
        "captured_tensor": "decoder_layer_output",
        "captured_layers": list(range(num_layers)),
        "final_rmsnorm_included": False,
        "lm_head_included": False,
        "summary_csv": str(args.summary_output),
        "elementwise_csv": str(args.elementwise_output),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "allow_bf16_reduced_precision_reduction": False,
        "ramulator": str(args.ramulator),
        "cache_root": str(args.cache_root),
        "mode2_runs": runner.runs,
        "mode2_batch_runs": runner.batch_runs,
        "mode2_gemv_calls": runner.gemv_calls,
        "cache_hits": runner.cache_hits,
        "mode2_total_wall_time_s": runner.total_wall_time_s,
        "mode2_token_batch": runner.batch_tokens,
        "pim_workers": runner.parallel_workers,
        "batch_token_chunk_size": runner.batch_token_chunk_size,
        "script_wall_time_s": time.perf_counter() - script_start,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"summary_csv: {args.summary_output}")
    print(f"elementwise_csv: {args.elementwise_output}")
    print(f"metadata: {args.metadata_output}")
    print(f"actual_seq_len: {actual_seq_len}")
    print(f"mode2_runs: {runner.runs}")
    print(f"mode2_batch_runs: {runner.batch_runs}")
    print(f"mode2_gemv_calls: {runner.gemv_calls}")
    print(f"cache_hits: {runner.cache_hits}")
    print(f"mode2_total_wall_time_s: {runner.total_wall_time_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
