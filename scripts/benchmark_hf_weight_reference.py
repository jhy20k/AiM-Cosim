#!/usr/bin/env python3
"""Benchmark CPU/GPU reference GEMV for HF safetensors or random projection values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hf_weight_projection as hfproj  # noqa: E402


def summarize(samples: list[float]) -> dict[str, str]:
    return {
        "mean_s": f"{statistics.mean(samples):.9f}",
        "median_s": f"{statistics.median(samples):.9f}",
        "stddev_s": f"{statistics.stdev(samples) if len(samples) > 1 else 0.0:.9f}",
        "min_s": f"{min(samples):.9f}",
        "max_s": f"{max(samples):.9f}",
    }


def benchmark_numpy_cpu(w, a, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        _ = hfproj.float32_to_bf16((w @ a).clip(min=0))

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = hfproj.float32_to_bf16((w @ a).clip(min=0))
        samples.append(time.perf_counter() - start)
    return samples


def benchmark_torch_cuda(w, a, warmup: int, repeats: int) -> tuple[str, list[float]] | tuple[None, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, f"unavailable_torch_import_{exc.__class__.__name__}"

    if not torch.cuda.is_available():
        return None, "unavailable_torch_cuda_not_available"

    try:
        device = torch.device("cuda")
        wt = torch.from_numpy(w).to(device=device, dtype=torch.bfloat16)
        at = torch.from_numpy(a).to(device=device, dtype=torch.bfloat16)
        samples: list[float] = []
        with torch.no_grad():
            for _ in range(warmup):
                _ = torch.relu(torch.matmul(wt, at))
            torch.cuda.synchronize()

            for _ in range(repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = torch.relu(torch.matmul(wt, at))
                end_event.record()
                torch.cuda.synchronize()
                samples.append(start_event.elapsed_time(end_event) / 1000.0)
        return "torch_cuda_bf16_kernel", samples
    except Exception as exc:  # pragma: no cover - depends on GPU/runtime
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None, f"unavailable_torch_cuda_{exc.__class__.__name__}"


def benchmark_cupy_cuda(w, a, warmup: int, repeats: int) -> tuple[str, list[float]] | tuple[None, str]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, f"unavailable_cupy_import_{exc.__class__.__name__}"

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None, "unavailable_cupy_cuda_not_available"

        wg = cp.asarray(w, dtype=cp.float32)
        ag = cp.asarray(a, dtype=cp.float32)
        samples: list[float] = []
        for _ in range(warmup):
            _ = cp.maximum(wg @ ag, 0)
        cp.cuda.Stream.null.synchronize()

        for _ in range(repeats):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            _ = cp.maximum(wg @ ag, 0)
            end_event.record()
            end_event.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
        return "cupy_cuda_float32_kernel", samples
    except Exception as exc:  # pragma: no cover - depends on GPU/runtime
        return None, f"unavailable_cupy_cuda_{exc.__class__.__name__}"


def benchmark_gpu(w, a, warmup: int, repeats: int) -> tuple[str, str, list[float] | None]:
    backend, result = benchmark_torch_cuda(w, a, warmup, repeats)
    if backend is not None:
        return backend, "ok", result

    torch_status = result
    backend, result = benchmark_cupy_cuda(w, a, warmup, repeats)
    if backend is not None:
        return backend, "ok", result

    return "gpu", f"{torch_status};{result}", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU reference time for LLM-shape GEMV")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--projection-group", required=True, choices=sorted(hfproj.PROJECTION_CANDIDATES))
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--row-start", type=int, default=0, help="Start row for R-dimension chunk timing")
    parser.add_argument("--row-count", type=int, help="Number of R rows for chunk timing")
    parser.add_argument(
        "--weight-source",
        choices=["hf_safetensors", "random_bf16"],
        default="hf_safetensors",
        help="Use actual HF safetensors weight or deterministic random BF16 values",
    )
    parser.add_argument("--weight-seed", type=int, default=314159)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-source",
        choices=["random_bf16", "prompt_hash_bf16"],
        default="random_bf16",
    )
    parser.add_argument("--prompt-file")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path("weight") / hfproj.safe_name(args.repo_id)
    model_source = args.repo_id if args.repo_id.startswith("http") else f"https://huggingface.co/{args.repo_id}"
    value_policy = "random_bf16_shape_preserving" if args.weight_source == "random_bf16" else "real_hf_safetensors"
    if args.weight_source == "random_bf16":
        tensor_name = f"random_bf16_weight_seed{args.weight_seed}"
        weight = hfproj.generate_random_weight(args.q, args.r, args.weight_seed)
    else:
        tensor_name, weight = hfproj.load_projection(
            model_dir,
            args.projection_group,
            args.q,
            args.r,
            args.layer,
            None,
            args.intermediate,
        )
    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.row_start or args.row_count is not None:
        row_end = args.r if args.row_count is None else args.row_start + args.row_count
        if row_end > args.r:
            raise ValueError(f"row chunk [{args.row_start}, {row_end}) exceeds R={args.r}")
        weight = weight[args.row_start:row_end].copy()
        tensor_name = f"{tensor_name}:rows[{args.row_start}:{row_end}]"
    effective_r = int(weight.shape[0])
    activation, activation_metadata = hfproj.activation_from_args(args)
    w = hfproj.bf16_to_float32(weight.reshape(-1)).reshape(weight.shape)
    a = hfproj.bf16_to_float32(activation)

    cpu_samples = benchmark_numpy_cpu(w, a, args.warmup, args.repeats)
    cpu_stats = summarize(cpu_samples)
    gpu_backend, gpu_status, gpu_samples = benchmark_gpu(w, a, args.warmup, args.repeats)
    gpu_stats = summarize(gpu_samples) if gpu_samples else {
        "mean_s": "",
        "median_s": "",
        "stddev_s": "",
        "min_s": "",
        "max_s": "",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
                "activation_source",
                "prompt_file",
                "max_new_tokens",
                "prompt_count",
                "prompt_digest",
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
        )
        writer.writerow(
            [
                args.repo_id,
                model_source,
                args.projection_group,
                tensor_name,
                args.layer,
                args.q,
                effective_r,
                f"{args.q}x{effective_r}",
                value_policy,
                args.weight_source,
                args.weight_seed if args.weight_source == "random_bf16" else "",
                activation_metadata["activation_source"],
                activation_metadata["prompt_file"],
                activation_metadata["max_new_tokens"],
                activation_metadata["prompt_count"],
                activation_metadata["prompt_digest"],
                "numpy_cpu",
                "ok",
                args.warmup,
                args.repeats,
                cpu_stats["mean_s"],
                cpu_stats["median_s"],
                cpu_stats["stddev_s"],
                cpu_stats["min_s"],
                cpu_stats["max_s"],
            ]
        )
        writer.writerow(
            [
                args.repo_id,
                model_source,
                args.projection_group,
                tensor_name,
                args.layer,
                args.q,
                effective_r,
                f"{args.q}x{effective_r}",
                value_policy,
                args.weight_source,
                args.weight_seed if args.weight_source == "random_bf16" else "",
                activation_metadata["activation_source"],
                activation_metadata["prompt_file"],
                activation_metadata["max_new_tokens"],
                activation_metadata["prompt_count"],
                activation_metadata["prompt_digest"],
                gpu_backend,
                gpu_status,
                args.warmup,
                args.repeats,
                gpu_stats["mean_s"],
                gpu_stats["median_s"],
                gpu_stats["stddev_s"],
                gpu_stats["min_s"],
                gpu_stats["max_s"],
            ]
        )

    print(f"wrote {out}")
    print(f"cpu mean_s={cpu_stats['mean_s']} median_s={cpu_stats['median_s']}")
    print(f"gpu backend={gpu_backend} status={gpu_status} mean_s={gpu_stats['mean_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
