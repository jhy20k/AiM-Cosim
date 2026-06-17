#!/usr/bin/env python3
"""Download Hugging Face model config and safetensors into weight/<model>.

This script intentionally depends only on the Python standard library so it can
run in the current AiM-Cosim environment without installing transformers/torch.
Authentication is read from HF_TOKEN and is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


BASE_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
}

PROJECTION_CANDIDATES = {
    "self_attn.q_proj": [
        "self_attn.q_proj.weight",
        "self_attn.qkv_proj.weight",
    ],
    "self_attn.k_proj": [
        "self_attn.k_proj.weight",
        "self_attn.qkv_proj.weight",
    ],
    "self_attn.v_proj": [
        "self_attn.v_proj.weight",
        "self_attn.qkv_proj.weight",
    ],
    "self_attn.o_proj": [
        "self_attn.o_proj.weight",
    ],
    "mlp.gate_proj": [
        "mlp.gate_proj.weight",
        "mlp.gate_up_proj.weight",
    ],
    "mlp.up_proj": [
        "mlp.up_proj.weight",
        "mlp.gate_up_proj.weight",
    ],
    "mlp.down_proj": [
        "mlp.down_proj.weight",
    ],
    "q/o": [
        "self_attn.q_proj.weight",
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
    ],
    "k/v": [
        "self_attn.k_proj.weight",
        "self_attn.qkv_proj.weight",
        "self_attn.v_proj.weight",
    ],
    "gate/up": [
        "mlp.gate_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.up_proj.weight",
    ],
    "down": [
        "mlp.down_proj.weight",
    ],
}


def model_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def auth_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "aim-cosim-hf-weight-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request_json(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url, headers=auth_headers(token))
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sibling_sizes(model_info: dict) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    for item in model_info.get("siblings", []):
        name = item.get("rfilename", "")
        size = item.get("size")
        sizes[name] = size if isinstance(size, int) else None
    return sizes


def selected_files(model_info: dict, include_tokenizer: bool, include_safetensors: bool) -> list[tuple[str, int | None]]:
    wanted_exact = set(BASE_FILES)
    if include_tokenizer:
        wanted_exact.update(
            {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "merges.txt",
                "vocab.json",
            }
        )

    sizes = sibling_sizes(model_info)
    selected: list[tuple[str, int | None]] = []
    for name, size in sizes.items():
        if name in wanted_exact or (include_safetensors and name.endswith(".safetensors")):
            selected.append((name, size))
    return sorted(selected)


def read_index(index_path: Path) -> dict:
    with index_path.open() as f:
        return json.load(f)


def choose_tensor_from_index(index_data: dict, group: str, layer: int, tensor_name: str | None) -> str:
    weight_map = index_data.get("weight_map", {})
    if tensor_name:
        if tensor_name in weight_map:
            return tensor_name
        if ":" in tensor_name:
            base, _split = tensor_name.split(":", 1)
            if base in weight_map:
                return base
        raise KeyError(f"tensor not found in HF index: {tensor_name}")

    layer_prefix = f"model.layers.{layer}."
    for suffix in PROJECTION_CANDIDATES[group]:
        exact = layer_prefix + suffix
        if exact in weight_map:
            return exact

    for suffix in PROJECTION_CANDIDATES[group]:
        matches = [name for name in weight_map if name.endswith(suffix)]
        if matches:
            return sorted(matches)[0]

    sample = "\n".join(sorted(weight_map)[:80])
    raise KeyError(f"no tensor found for group {group}; first tensors:\n{sample}")


def files_for_projection(out_dir: Path, group: str, layer: int, tensor_name: str | None) -> tuple[list[str], list[str]]:
    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.exists():
        single = out_dir / "model.safetensors"
        return (["model.safetensors"] if single.exists() else [], [])

    index_data = read_index(index_path)
    weight_map = index_data.get("weight_map", {})
    selected_tensor = choose_tensor_from_index(index_data, group, layer, tensor_name)
    selected_file = weight_map[selected_tensor]
    return [selected_file], [selected_tensor]


def download_file(repo_id: str, revision: str, filename: str, dest: Path, token: str | None, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip existing: {dest}")
        return

    encoded = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{encoded}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers=auth_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total_i = int(total) if total and total.isdigit() else None
            seen = 0
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                seen += len(chunk)
                if total_i:
                    pct = 100.0 * seen / total_i
                    print(f"\rdownload {filename}: {seen / (1024**3):.2f}/{total_i / (1024**3):.2f} GiB ({pct:.1f}%)", end="")
                else:
                    print(f"\rdownload {filename}: {seen / (1024**3):.2f} GiB", end="")
            print()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        print(f"HF download error for {filename}: HTTP {exc.code}", file=sys.stderr)
        if exc.code in (401, 403):
            print("Check that HF_TOKEN is set and that the model license/access request is accepted.", file=sys.stderr)
        raise

    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF config and safetensors for AiM-Cosim LLM-weight experiments")
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--output-root", default="weight", help="Directory that will contain one subdirectory per model")
    parser.add_argument("--output-dir", help="Exact directory for this model; overrides --output-root")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include-tokenizer", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token-stdin", action="store_true", help="Read HF token from stdin instead of HF_TOKEN")
    parser.add_argument(
        "--only-needed-shards",
        action="store_true",
        help="Download metadata plus only the safetensors shard needed for one projection",
    )
    parser.add_argument("--projection-group", choices=sorted(PROJECTION_CANDIDATES), help="Projection group for --only-needed-shards")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tensor-name", help="Exact HF tensor name override for --only-needed-shards")
    args = parser.parse_args()

    if args.only_needed_shards and not (args.projection_group or args.tensor_name):
        parser.error("--only-needed-shards requires --projection-group or --tensor-name")

    token = sys.stdin.readline().strip() if args.token_stdin else os.environ.get("HF_TOKEN")
    api_url = f"https://huggingface.co/api/models/{args.repo_id}?revision={urllib.parse.quote(args.revision)}"
    try:
        info = request_json(api_url, token)
    except urllib.error.HTTPError as exc:
        print(f"HF API error for {args.repo_id}: HTTP {exc.code}", file=sys.stderr)
        if exc.code in (401, 403):
            print("Check that HF_TOKEN is set and that the model license/access request is accepted.", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / model_dir_name(args.repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "download_manifest.json"
    sizes = sibling_sizes(info)
    files = selected_files(info, args.include_tokenizer, include_safetensors=not args.only_needed_shards)

    selected_tensors: list[str] = []
    if args.only_needed_shards:
        for name, _ in files:
            if name in BASE_FILES:
                download_file(args.repo_id, args.revision, name, out_dir / name, token, args.force)
        shard_names, selected_tensors = files_for_projection(out_dir, args.projection_group or "q/o", args.layer, args.tensor_name)
        if not shard_names:
            shard_names = [name for name in sizes if name.endswith(".safetensors")]
        existing = {name for name, _ in files}
        for name in shard_names:
            if name not in existing:
                files.append((name, sizes.get(name)))

    with manifest_path.open("w") as f:
        json.dump(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "model_dir": str(out_dir),
                "only_needed_shards": args.only_needed_shards,
                "projection_group": args.projection_group,
                "layer": args.layer,
                "selected_tensors": selected_tensors,
                "files": [{"name": name, "size": size} for name, size in files],
            },
            f,
            indent=2,
        )

    print(f"repo: {args.repo_id}")
    print(f"output: {out_dir}")
    print(f"files: {len(files)}")

    try:
        for name, _ in files:
            download_file(args.repo_id, args.revision, name, out_dir / name, token, args.force)
    except urllib.error.HTTPError:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
