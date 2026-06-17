#!/usr/bin/env python3
"""Run Summer 2026 HF-weight Stage 5-7 experiments.

The runner executes real HF safetensors projection experiments for the selected
LLM projection groups:

- Stage 5: AiMRTL Mode2 functional validation
- Stage 6: AiM Mode1 timing on the same trace
- Stage 7: CPU/GPU reference timing
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/jhlee/miniconda3/envs/summer26/bin/python")
BUILD_RAMULATOR = ROOT / "build_mag32/extern/aim_simulator/ramulator2"
BUILD_OPT_RAMULATOR = ROOT / "build_opt/extern/aim_simulator/ramulator2"
BUILD_OPT_NATIVE_RAMULATOR = ROOT / "build_opt_native/extern/aim_simulator/ramulator2"
MODE1_CONFIG = ROOT / "src/configs/aim_timing_only.yaml"
LLM_OUT = ROOT / "tests/result/result_summer26/llm_shape"
REFERENCE_OUT = ROOT / "tests/result/result_summer26/reference_time/cpu_gpu"
PROMPT_CONFIG = ROOT / "tests/result/result_summer26/artifacts/prompt_inputs_maxnew16.json"


PROJECTION_GROUPS = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


MODELS = [
    {
        "model": "EXAONE-4.0-1.2B",
        "repo_id": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "file_id": "exaone",
        "groups": {
            "self_attn.q_proj": (2048, 2048, None),
            "self_attn.k_proj": (2048, 512, None),
            "self_attn.v_proj": (2048, 512, None),
            "self_attn.o_proj": (2048, 2048, None),
            "mlp.gate_proj": (2048, 4096, 4096),
            "mlp.up_proj": (2048, 4096, 4096),
            "mlp.down_proj": (4096, 2048, None),
        },
    },
    {
        "model": "Phi-4-mini-Instruct",
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "file_id": "phi4mini",
        "groups": {
            "self_attn.q_proj": (3072, 3072, 8192),
            "self_attn.k_proj": (3072, 1024, 8192),
            "self_attn.v_proj": (3072, 1024, 8192),
            "self_attn.o_proj": (3072, 3072, None),
            "mlp.gate_proj": (3072, 8192, 8192),
            "mlp.up_proj": (3072, 8192, 8192),
            "mlp.down_proj": (8192, 3072, None),
        },
    },
    {
        "model": "Llama3.2-1B-Instruct",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "file_id": "llama32_1b",
        "groups": {
            "self_attn.q_proj": (2048, 2048, 8192),
            "self_attn.k_proj": (2048, 512, 8192),
            "self_attn.v_proj": (2048, 512, 8192),
            "self_attn.o_proj": (2048, 2048, None),
            "mlp.gate_proj": (2048, 8192, 8192),
            "mlp.up_proj": (2048, 8192, 8192),
            "mlp.down_proj": (8192, 2048, None),
        },
    },
    {
        "model": "Llama3.1-8B-Instruct",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "file_id": "llama31_8b",
        "groups": {
            "self_attn.q_proj": (4096, 4096, 14336),
            "self_attn.k_proj": (4096, 1024, 14336),
            "self_attn.v_proj": (4096, 1024, 14336),
            "self_attn.o_proj": (4096, 4096, None),
            "mlp.gate_proj": (4096, 14336, 14336),
            "mlp.up_proj": (4096, 14336, 14336),
            "mlp.down_proj": (14336, 4096, None),
        },
    },
    {
        "model": "Qwen3-8B",
        "repo_id": "Qwen/Qwen3-8B",
        "file_id": "qwen3_8b",
        "groups": {
            "self_attn.q_proj": (4096, 4096, 12288),
            "self_attn.k_proj": (4096, 1024, 12288),
            "self_attn.v_proj": (4096, 1024, 12288),
            "self_attn.o_proj": (4096, 4096, None),
            "mlp.gate_proj": (4096, 12288, 12288),
            "mlp.up_proj": (4096, 12288, 12288),
            "mlp.down_proj": (12288, 4096, None),
        },
    },
]


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def group_id(group: str) -> str:
    return clean_tag(group)


def clean_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)


def run_id(repo_id: str, group: str, q: int, effective_r: int, tag: str) -> str:
    return f"{safe_name(repo_id)}_{group_id(group)}_p1_q{q}_r{effective_r}_layer0_seed42_{clean_tag(tag)}"


def tiles_per_row(q: int) -> int:
    return max(128, q // 16)


def estimate_aimd_bytes(q: int, r: int) -> int:
    return 16 * r * tiles_per_row(q) * 16 * 2


def default_ramulator() -> Path:
    if BUILD_RAMULATOR.exists():
        return BUILD_RAMULATOR
    if BUILD_OPT_NATIVE_RAMULATOR.exists():
        return BUILD_OPT_NATIVE_RAMULATOR
    return BUILD_OPT_RAMULATOR if BUILD_OPT_RAMULATOR.exists() else BUILD_RAMULATOR


def ramulator_env(ramulator: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_dir = str(ramulator.resolve().parent)
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = lib_dir if not current else f"{lib_dir}:{current}"
    return env


def chunk_ranges(q: int, r: int, chunk_rows: int) -> list[tuple[int, int]]:
    if chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if r <= chunk_rows:
        return [(0, r)]
    ranges = []
    start = 0
    while start < r:
        end = min(r, start + chunk_rows)
        ranges.append((start, end))
        start = end
    return ranges


def parse_cycles(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(errors="ignore").splitlines():
        if "memory_system_cycles:" in line:
            return line.split()[1]
    return ""


def run_checked(
    cmd: list[str],
    cwd: Path = ROOT,
    stdout_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    if stdout_path is None:
        return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode
    with stdout_path.open("w") as out:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, check=False).returncode


def run_stage5(
    model: dict[str, object],
    group: str,
    q: int,
    r: int,
    intermediate: int | None,
    row_start: int,
    row_end: int,
    channel_layout: str,
    ramulator: Path,
    activation_source: str,
    prompt_file: Path | None,
    force: bool,
) -> Path:
    repo_id = str(model["repo_id"])
    effective_r = row_end - row_start
    layout_tag = "rowshard32" if channel_layout == "row_sharded" else "broadcast32"
    activation_tag = "prompt_hash" if activation_source == "prompt_hash_bf16" else "seed42"
    tag = (
        f"active_rows_{layout_tag}_{activation_tag}"
        if row_start == 0 and row_end == r
        else f"active_rows_{layout_tag}_{activation_tag}_chunk{row_start}_{row_end}"
    )
    rid = run_id(repo_id, group, q, effective_r, tag)
    out_dir = LLM_OUT / rid
    report = out_dir / "functional_report.csv"
    if report.exists() and not force:
        print(f"[stage5] skip existing {rid}")
        return out_dir

    cmd = [
        str(PYTHON),
        "scripts/run_hf_weight_projection.py",
        "--repo-id",
        repo_id,
        "--projection-group",
        group,
        "--q",
        str(q),
        "--r",
        str(r),
        "--layer",
        "0",
        "--weight-source",
        "hf_safetensors",
        "--activation-source",
        activation_source,
        "--run-tag",
        tag,
        "--channel-layout",
        channel_layout,
        "--output-dir",
        str(LLM_OUT),
        "--ramulator",
        str(ramulator),
    ]
    if prompt_file is not None:
        cmd.extend(["--prompt-file", str(prompt_file)])
    if intermediate is not None:
        cmd.extend(["--intermediate", str(intermediate)])
    if row_start != 0 or row_end != r:
        cmd.extend(["--row-start", str(row_start), "--row-count", str(effective_r)])
    print(f"[stage5] run {rid}")
    rc = run_checked(cmd, env=ramulator_env(ramulator))
    if rc != 0:
        raise RuntimeError(f"Stage5 failed for {rid} with return code {rc}")
    return out_dir


def run_stage6_mode1(run_dir: Path, ramulator: Path, force: bool) -> None:
    out = run_dir / "mode1.txt"
    wall = run_dir / "mode1_wall_time.csv"
    if out.exists() and wall.exists() and not force:
        print(f"[stage6] skip existing {run_dir.name}")
        return

    traces = sorted(run_dir.glob("p1_q*_r*.trace"))
    if not traces:
        raise FileNotFoundError(f"trace not found in {run_dir}")
    trace = traces[0]
    print(f"[stage6] run Mode1 {run_dir.name}")
    start = time.perf_counter()
    rc = run_checked(
        [str(ramulator), "-f", str(MODE1_CONFIG), "-t", str(trace)],
        stdout_path=out,
        env=ramulator_env(ramulator),
    )
    elapsed = time.perf_counter() - start
    with wall.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "mode", "wall_time_s", "returncode"])
        writer.writerow([run_dir.name, "Mode1", f"{elapsed:.6f}", rc])
    if rc != 0:
        raise RuntimeError(f"Mode1 failed for {run_dir.name} with return code {rc}")
    print(f"[stage6] cycles={parse_cycles(out)} wall_time_s={elapsed:.3f}")


def run_stage7_reference(
    model: dict[str, object],
    group: str,
    q: int,
    r: int,
    intermediate: int | None,
    row_start: int,
    row_end: int,
    activation_source: str,
    prompt_file: Path | None,
    force: bool,
) -> None:
    repo_id = str(model["repo_id"])
    file_id = str(model["file_id"])
    activation_suffix = "_prompt_hash" if activation_source == "prompt_hash_bf16" else ""
    chunk_suffix = "" if row_start == 0 and row_end == r else f"_rows{row_start}_{row_end}"
    out = REFERENCE_OUT / f"hf_weight_{file_id}_{group_id(group)}{activation_suffix}{chunk_suffix}_cpu_gpu.csv"
    if out.exists() and not force:
        print(f"[stage7] skip existing {out.name}")
        return

    cmd = [
        str(PYTHON),
        "scripts/benchmark_hf_weight_reference.py",
        "--repo-id",
        repo_id,
        "--projection-group",
        group,
        "--q",
        str(q),
        "--r",
        str(r),
        "--layer",
        "0",
        "--weight-source",
        "hf_safetensors",
        "--activation-source",
        activation_source,
        "--warmup",
        "10",
        "--repeats",
        "5",
        "--output",
        str(out),
    ]
    if prompt_file is not None:
        cmd.extend(["--prompt-file", str(prompt_file)])
    if intermediate is not None:
        cmd.extend(["--intermediate", str(intermediate)])
    if row_start != 0 or row_end != r:
        cmd.extend(["--row-start", str(row_start), "--row-count", str(row_end - row_start)])
    print(f"[stage7] run CPU/GPU {out.name}")
    rc = run_checked(cmd)
    if rc != 0:
        raise RuntimeError(f"Stage7 failed for {repo_id} {group} with return code {rc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Summer 2026 HF-weight Stage 5-7 projection experiments")
    parser.add_argument("--models", nargs="*", choices=[m["model"] for m in MODELS], help="Model names to run")
    parser.add_argument(
        "--projection-groups",
        nargs="*",
        default=list(PROJECTION_GROUPS),
        choices=list(PROJECTION_GROUPS),
    )
    parser.add_argument(
        "--activation-source",
        choices=["random_bf16", "prompt_hash_bf16"],
        default="random_bf16",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROMPT_CONFIG,
        help="Prompt JSON used when --activation-source=prompt_hash_bf16",
    )
    parser.add_argument("--force-stage5", action="store_true")
    parser.add_argument("--force-mode1", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument(
        "--channel-layout",
        choices=["row_sharded", "broadcast"],
        default="row_sharded",
        help="row_sharded distributes output rows across 32 channels; broadcast keeps the legacy replicated trace",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1024,
        help="Split R dimension into fixed row chunks; final chunk keeps the remainder",
    )
    parser.add_argument(
        "--max-aimd-mib",
        type=int,
        default=0,
        help="Deprecated compatibility knob; fixed --chunk-rows is the official Summer 2026 policy",
    )
    parser.add_argument(
        "--ramulator",
        type=Path,
        default=default_ramulator(),
        help="Ramulator2 executable to use for Mode2/Mode1; defaults to build_mag32 when present",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    ramulator = args.ramulator.resolve()
    if not ramulator.exists():
        raise FileNotFoundError(ramulator)

    selected = [m for m in MODELS if not args.models or m["model"] in args.models]
    prompt_file = args.prompt_file.resolve() if args.activation_source == "prompt_hash_bf16" else None
    if prompt_file is not None and not prompt_file.exists():
        raise FileNotFoundError(prompt_file)
    LLM_OUT.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUT.mkdir(parents=True, exist_ok=True)

    for model in selected:
        for group in args.projection_groups:
            q, r, intermediate = model["groups"][group]  # type: ignore[index]
            ranges = chunk_ranges(q, r, args.chunk_rows)
            if len(ranges) > 1:
                print(
                    f"[chunk] {model['model']} {group} {q}x{r} -> {len(ranges)} chunks "
                    f"(chunk_rows={args.chunk_rows})"
                )
            for row_start, row_end in ranges:
                run_dir = run_stage5(
                    model,
                    group,
                    q,
                    r,
                    intermediate,
                    row_start,
                    row_end,
                    args.channel_layout,
                    ramulator,
                    args.activation_source,
                    prompt_file,
                    args.force_stage5,
                )
                run_stage6_mode1(run_dir, ramulator, args.force_mode1)
                run_stage7_reference(
                    model,
                    group,
                    q,
                    r,
                    intermediate,
                    row_start,
                    row_end,
                    args.activation_source,
                    prompt_file,
                    args.force_reference,
                )

    print("[done] Stage 5-7 selected projection experiments complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
