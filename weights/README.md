# Model weights

Run the commands below from the repository root. The `weights/` directory should contain these files after setup:

```text
weights/
├── config.json
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── google/umt5-xxl/              # tokenizer files from Wan2.1
├── flashdelight_rank0_full.pt     # FlashRelight stage 1, 6.93 GB
├── flashrelight_rank0_full.pt     # full relighting only, 7.04 GB
└── BiRefNet-portrait/            # optional, for portrait matting
```

Full relighting uses both project checkpoints and the shared Wan2.1 configuration, VAE, text encoder, and tokenizer. [FlashRelight checkpoints](https://huggingface.co/JiahuiSheng/FlashRelight) · [Wan2.1 base files](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

```bash
python scripts/download_weights.py --dest weights --skip_matting_model
```

For standalone delighting, download just FlashDelight and the shared Wan2.1 files. [Standalone inference example](../README.md#standalone-delighting)

```bash
python scripts/download_weights.py --dest weights --delight_only --skip_matting_model
```

Omit `--skip_matting_model` to also download [BiRefNet-portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait) for foreground extraction.

For offline setup, point `WAN_DIR` to the local Wan2.1 directory and `STAGE_DIR` to the directory containing the project checkpoints.

```bash
python scripts/download_weights.py --dest weights --link \
  --wan_dir "$WAN_DIR" \
  --delight_ckpt "$STAGE_DIR/flashdelight_rank0_full.pt" \
  --relight_ckpt "$STAGE_DIR/flashrelight_rank0_full.pt" \
  --skip_matting_model
```

To enable matting offline, replace `--skip_matting_model` with `--birefnet_dir "$BIREFNET_DIR"`, where `BIREFNET_DIR` points to a local BiRefNet-portrait snapshot. The `--link` flag creates symbolic links. Omit it to copy the files. For a different weight directory at inference time, pass `--weights PATH`.
