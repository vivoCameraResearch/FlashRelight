#!/usr/bin/env python3
"""Fetch / assemble the weights needed for FlashRelight inference.

Target layout (``--dest``, default ``./weights``)::

    weights/
    |-- config.json                        # Wan2.1-T2V-1.3B DiT config
    |-- Wan2.1_VAE.pth                     # Wan2.1 3D causal VAE        (0.5 GB)
    |-- models_t5_umt5-xxl-enc-bf16.pth    # umT5-XXL text encoder       (11 GB)
    |-- google/umt5-xxl/                   # umT5 tokenizer
    |-- flashdelight_rank0_full.pt         # stage 1: delighting         (6.9 GB)
    |-- flashrelight_rank0_full.pt         # stage 2: relighting         (7.0 GB)
    `-- BiRefNet-portrait/                 # optional matting model      (0.9 GB)

Examples
--------
    # download the project checkpoints and shared files, reusing existing copies
    python scripts/download_weights.py --dest weights

    # base model from HF, FlashRelight checkpoints from local paths
    python scripts/download_weights.py --dest weights \
        --delight_ckpt /path/to/flashdelight.pt \
        --relight_ckpt /path/to/flashrelight.pt

    # standalone delighting (Wan base files and the FlashDelight checkpoint)
    python scripts/download_weights.py --dest weights --delight_only --skip_matting_model

    # symlink instead of copy (saves 14 GB when the files are already local)
    python scripts/download_weights.py --dest weights --link \
        --delight_ckpt ... --relight_ckpt ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_FILES = [
    "config.json",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
]
BIREFNET_REPO = "ZhengPeng7/BiRefNet-portrait"

FLASHRELIGHT_REPO = os.environ.get("FLASHRELIGHT_HF_REPO", "JiahuiSheng/FlashRelight")


def place(src: str, dst: Path, link: bool) -> None:
    src_path = Path(src).expanduser()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        print(f"  = {dst.name} already present, skipping")
        return
    if link:
        dst.symlink_to(src_path.resolve())
        print(f"  -> linked {dst.name}")
    elif src_path.is_dir():
        shutil.copytree(src_path, dst)
        print(f"  -> copied {dst.name}/")
    else:
        shutil.copy2(src_path, dst)
        print(f"  -> copied {dst.name} ({dst.stat().st_size / 1e9:.1f} GB)")


def hf_download(repo: str, dest: Path, files=None, repo_dir: str | None = None) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    if files is None:
        target = dest / repo_dir if repo_dir else dest
        print(f"  snapshot {repo} -> {target}")
        snapshot_download(repo_id=repo, local_dir=str(target),
                          local_dir_use_symlinks=False)
        return

    for rel in files:
        out = dest / rel
        if out.exists():
            print(f"  = {rel} already present")
            continue
        print(f"  downloading {repo}:{rel}")
        hf_hub_download(repo_id=repo, filename=rel, local_dir=str(dest))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "weights"))
    p.add_argument("--delight_only", action="store_true",
                   help="prepare the shared Wan files and just the FlashDelight checkpoint")
    p.add_argument("--delight_ckpt", type=str, default=None,
                   help="local FlashDelight checkpoint (otherwise fetched from HF)")
    p.add_argument("--relight_ckpt", type=str, default=None,
                   help="local FlashRelight checkpoint (otherwise fetched from HF)")
    p.add_argument("--wan_dir", type=str, default=None,
                   help="local Wan2.1-T2V-1.3B directory (otherwise fetched from HF)")
    p.add_argument("--birefnet_dir", type=str, default=None,
                   help="local BiRefNet-portrait directory")
    p.add_argument("--link", action="store_true",
                   help="symlink local sources instead of copying")
    p.add_argument("--skip_matting_model", action="store_true")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"destination: {dest}")

    local = {"flashdelight_rank0_full.pt": args.delight_ckpt}
    checkpoint_options = "--delight_ckpt"
    if not args.delight_only:
        local["flashrelight_rank0_full.pt"] = args.relight_ckpt
        checkpoint_options += " / --relight_ckpt"
    for name, src in local.items():
        if src and not Path(src).expanduser().is_file():
            p.error(f"checkpoint file does not exist: {src}")
    missing_stages = [name for name, src in local.items()
                      if not src and not (dest / name).is_file()]
    if missing_stages and not FLASHRELIGHT_REPO:
        p.error("Provide the missing stage checkpoints before downloading the base model: "
                + ", ".join(missing_stages)
                + f". Use {checkpoint_options}; see weights/README.md.")

    print("\n[1/3] Wan2.1-T2V-1.3B base model")
    if args.wan_dir:
        for rel in WAN_FILES:
            place(str(Path(args.wan_dir) / rel), dest / rel, args.link)
    else:
        if FLASHRELIGHT_REPO == "JiahuiSheng/FlashRelight":
            hf_download(FLASHRELIGHT_REPO, dest, ["config.json"])
        hf_download(WAN_REPO, dest, WAN_FILES)

    print("\n[2/3] FlashRelight stage checkpoints")
    for name, src in local.items():
        if (dest / name).is_file():
            print(f"  = {name} already present")
        elif src:
            place(src, dest / name, args.link)
        else:
            hf_download(FLASHRELIGHT_REPO, dest, [name])

    print("\n[3/3] BiRefNet-portrait (optional, for in-the-wild matting)")
    if args.skip_matting_model:
        print("  skipped (--skip_matting_model); use --matting none at inference")
    elif args.birefnet_dir:
        place(args.birefnet_dir, dest / "BiRefNet-portrait", args.link)
    else:
        hf_download(BIREFNET_REPO, dest, None, repo_dir="BiRefNet-portrait")

    print("\nsummary:")
    required = [*WAN_FILES, *local]
    missing = [r for r in required if not (dest / r).is_file()]
    for r in required:
        mark = "ok " if (dest / r).is_file() else "MISSING"
        print(f"  [{mark}] {r}")
    if missing:
        sys.exit(f"\n{len(missing)} required file(s) missing.")
    example = ("--delight_only --source assets/sources/generated_portrait.mp4 --matting none"
               if args.delight_only else "--samples assets/samples.json")
    print(f"\nAll set. From the repository root, try:\n"
          f'  python scripts/infer.py --weights "{dest.resolve()}" {example} --out_dir outputs')


if __name__ == "__main__":
    main()
