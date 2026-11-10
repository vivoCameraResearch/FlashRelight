# Example sources

| Asset | Origin |
| --- | --- |
| `sources/generated_portrait.mp4`, `sources/generated_gesture.mp4` | Generated portrait inputs from the FlashRelight cross-domain evaluation. The two clips are 49 frames at 16 fps and are ready to use with `--matting none`. |
| `light_refs/studio_sweep.mp4` | Studio lighting reference from the authors' release example pack. |
| `envmaps/01_horror_uplight_reveal/` | Procedural uplighting sequence from the FlashRelight evaluation (example 1). |
| `envmaps/02_zwartkops_start_sunset/` | Animated panorama from evaluation example 2. [Zwartkops Start Sunset](https://polyhaven.com/a/zwartkops_start_sunset) |
| `envmaps/03_stone_alley_02/` | Animated panorama from evaluation example 3. [Stone Alley 02](https://polyhaven.com/a/stone_alley_02) |
| `envmaps/04_random_great_circle/` | Procedural light moving along a great circle from evaluation example 4. |
| `envmaps/meadow_2_4k.exr` | Single 4K panorama. [Meadow 2](https://polyhaven.com/a/meadow_2) |
| `envmaps/kloofendal_48d_partly_cloudy_puresky_4k.exr` | Single 4K sky panorama. [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky) |
| [Delighting a portrait](../docs/assets/delighting.gif) | Input and recorded first-stage output for `generated_portrait.mp4` from the authors' generated-video evaluation. |
| [Delighting a portrait with glasses](../docs/assets/delighting-gesture.webp) | `generated_gesture.mp4` alongside the first-stage output from the `great_circle` example. |
| [Data curation](../docs/assets/data-curation.png) and [stage-specific training pairs](../docs/assets/training-pairs.png) | Figures 2 and 4 from the FlashRelight paper. |
| README relighting animation | The bundled portrait with glasses and its outputs from the moving-light and studio-sweep examples. |

Poly Haven HDRIs are distributed under [CC0](https://polyhaven.com/license). The four jobs in [samples.json](samples.json) use paths relative to this directory. All six environment maps are available in the demo and through `--env_map`.

Each numbered directory contains 49 original 1024 × 512 EXR frames. The two
single panoramas are 4096 × 2048. All six maps retain their original HDR values.
During inference, FlashRelight resizes each map to 128 × 64 and multiplies
its linear RGB values by 0.1.
