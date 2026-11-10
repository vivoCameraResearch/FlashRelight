# Third-party software

FlashRelight builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1). The Wan transformer, VAE and text-encoder implementations in `flashrelight/models/` are adapted from the Wan project. The flow-matching sampler in `flashrelight/schedulers/` is adapted from the Wan implementation of the [Diffusers UniPC scheduler](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py). The original copyright notices are retained in the source. These upstream components use the [Apache License, Version 2.0](licenses/Apache-2.0.txt).

[BiRefNet-portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait) provides optional matting. The downloader retrieves it separately from Hugging Face. See the model card for its authors and MIT license.

The paper describes the use of [FaceOLAT](https://gvv-assets.mpi-inf.mpg.de/FaceOLAT/?page_id=12) for training. Please cite Rao et al., *3DPR: Single Image 3D Portrait Relighting with Generative Priors*, SIGGRAPH Asia 2025, for FaceOLAT.
