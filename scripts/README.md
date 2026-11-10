# Scripts

- `download_weights.py` assembles the Wan2.1 base files and the required stage checkpoints. Use `--delight_only` to prepare FlashDelight by itself. See [weight setup](../weights/README.md).
- `infer.py` runs a single relighting job, a batch described in JSON or delighting alone. See the [quick start](../README.md#quick-start), or run `python scripts/infer.py --help`.
