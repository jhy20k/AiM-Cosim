#!/usr/bin/env python3
"""Collect prefix final-hidden delta summaries with WikiText prompts.

This reuses the propagation Mode2 runner, but writes only the final_hidden
by-layer summary so WikiText all-token runs do not materialize huge elementwise
CSV files.
"""

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


OUT_DIR = (
    ROOT
    / "tests"
    / "result"
    / "llama32_error_accumulation"
    / "summary"
    / "prefix_final_hidden"
)

SUMMARY_FIELDS_WITH_ABS_MEAN = [
    "prompt_index",
    "decode_step",
    "propagation_policy",
    "target_layer",
    "tensor_name",
    "projection",
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
CACHE_ROOT = (
    ROOT
    / "tests"
    / "result"
    / "llama32_error_accumulation"
    / "runs"
    / "propagation"
    / "pim_cache"
)
RUN_ROOT = (
    ROOT
    / "tests"
    / "result"
    / "llama32_error_accumulation"
    / "runs"
    / "wikitext_prefix_final_hidden"
)


def load_wikitext_prompt(tokenizer: Any, seq_len: int, dataset_samples: int) -> tuple[str, int]:
    from datasets import load_dataset

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        download_mode="reuse_dataset_if_exists",
    )
    texts: list[str] = []
    for item in dataset:
        text = str(item.get("text", "")).strip()
        if not text or text.startswith("="):
            continue
        texts.append(text)
        if len(texts) >= dataset_samples:
            break
    if not texts:
        raise RuntimeError("No non-empty WikiText samples were found")

    raw = "\n\n".join(texts)
    tokenized = tokenizer(raw, add_special_tokens=True, truncation=True, max_length=seq_len)
    token_ids = list(tokenized["input_ids"])
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

    # The propagation script tokenizes prompt text normally. Trim by tokens until
    # the retokenized length is bounded by seq_len while staying close to it.
    while True:
        actual_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(actual_ids) <= seq_len:
            return prompt, len(actual_ids)
        body_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not body_ids:
            raise RuntimeError("Could not trim WikiText prompt to requested length")
        prompt = tokenizer.decode(body_ids[:-1], skip_special_tokens=True)


def selected_final_hidden_bits(tensor: Any, all_tokens: bool) -> np.ndarray:
    selected = tensor[0] if all_tokens else tensor[0, -1:, :]
    return prop.torch_bf16_bits(selected)


def summarize_final_hidden(
    prompt_index: int,
    target_layer: int,
    gpu_tensor: Any,
    pim_tensor: Any,
    all_tokens: bool,
) -> dict[str, Any]:
    gpu_bits = selected_final_hidden_bits(gpu_tensor, all_tokens)
    pim_bits = selected_final_hidden_bits(pim_tensor, all_tokens)
    if gpu_bits.shape != pim_bits.shape:
        raise RuntimeError(f"final_hidden shape mismatch: {gpu_bits.shape} != {pim_bits.shape}")

    gpu_values = prop.bf16_values(gpu_bits).reshape(-1)
    pim_values = prop.bf16_values(pim_bits).reshape(-1)
    deltas = (gpu_values - pim_values).astype(np.float64)
    key = {
        "prompt_index": prompt_index,
        "decode_step": 0,
        "propagation_policy": "prefix",
        "target_layer": target_layer,
        "tensor_name": "final_hidden",
        "projection": "",
    }
    row = prop.summary_row(key, deltas.tolist())
    row["abs_delta_mean"] = float(np.mean(np.abs(deltas))) if deltas.size else 0.0
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS_WITH_ABS_MEAN, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--dataset-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--all-tokens", action="store_true")
    parser.add_argument("--layers", default="all", help="Comma-separated layers, 'auto', or 'all'")
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
    args = parser.parse_args()
    model_dir = prop.resolve_model_dir(args.model_id, args.model_dir)

    if not args.ramulator.exists():
        raise FileNotFoundError(f"ramulator2 not found: {args.ramulator}")

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prop.ensure_model_available(offline=args.offline, model_id=args.model_id, model_dir=model_dir)

    device = "cuda"
    cuda_error: Exception | None = None
    for _attempt in range(5):
        try:
            _ = torch.empty(1, device=device)
            cuda_error = None
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            cuda_error = exc
            time.sleep(2.0)
    if cuda_error is not None:
        raise RuntimeError("WikiText propagation run requires CUDA") from cuda_error

    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt, actual_seq_len = load_wikitext_prompt(tokenizer, args.seq_len, args.dataset_samples)

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model.to(device)
    model.eval()
    num_layers = prop.decoder_layer_count(model)

    run_cache = CACHE_ROOT
    run_cache.mkdir(parents=True, exist_ok=True)
    runner = prop.PIMRunner(
        args.ramulator,
        run_cache,
        resume=args.resume,
        keep_artifacts=args.keep_run_artifacts,
        model_id=args.model_id,
    )
    prop.wrap_model_projections(model, runner)
    layers = prop.parse_layer_spec(args.layers, num_layers, default="all")

    rows: list[dict[str, Any]] = []
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        with torch.inference_mode():
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            runner.set_context("disabled", -1, set(), 0, 0)
            baseline = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

            for target_layer in layers:
                active_layers = prop.active_layers_for("prefix", target_layer, num_layers)
                runner.set_context("prefix", target_layer, active_layers, 0, 0)
                pim_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                rows.append(
                    summarize_final_hidden(
                        0,
                        target_layer,
                        baseline.hidden_states[-1],
                        pim_out.hidden_states[-1],
                        args.all_tokens,
                    )
                )
                write_rows(args.output, rows)
                print(
                    f"[wikitext-final-hidden] layer={target_layer} done "
                    f"(rows={len(rows)}, mode2_runs={runner.runs}, cache_hits={runner.cache_hits})",
                    flush=True,
                )
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
        "all_tokens": args.all_tokens,
        "prompt_preview": prompt[:500],
        "layers": layers,
        "propagation_policy": "prefix",
        "tensor_name": "final_hidden",
        "output_csv": str(args.output),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "allow_bf16_reduced_precision_reduction": False,
        "ramulator": str(args.ramulator),
        "cache_root": str(run_cache),
        "mode2_runs": runner.runs,
        "cache_hits": runner.cache_hits,
        "mode2_total_wall_time_s": runner.total_wall_time_s,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"output_csv: {args.output}")
    print(f"metadata: {metadata_path}")
    print(f"actual_seq_len: {actual_seq_len}")
    print(f"mode2_runs: {runner.runs}")
    print(f"cache_hits: {runner.cache_hits}")
    print(f"mode2_total_wall_time_s: {runner.total_wall_time_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
