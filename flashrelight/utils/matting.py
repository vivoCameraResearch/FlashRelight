# flashrelight/utils/matting.py
#
# Optional portrait matting used to turn an arbitrary "in the wild" clip into the
# input distribution FlashRelight was trained on: a single subject composited on
# a black background.
#
# The default backend is BiRefNet-portrait (https://huggingface.co/ZhengPeng7/BiRefNet-portrait),
# loaded through `transformers` with `trust_remote_code=True`. It is entirely
# optional -- pass `matting="none"` to feed frames to the model untouched.
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_BIREFNET_ID = "ZhengPeng7/BiRefNet-portrait"


class PortraitMatter:
    """BiRefNet-portrait alpha matting wrapper.

    Parameters
    ----------
    model_path:
        Local directory or HuggingFace repo id of a BiRefNet checkpoint.
    device:
        Torch device string.
    resolution:
        Network input resolution (BiRefNet is trained at 1024x1024).
    half_precision:
        Run the segmenter in fp16 (CUDA only).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_BIREFNET_ID,
        device: str = "cuda",
        resolution: tuple[int, int] = (1024, 1024),
        half_precision: bool = True,
    ):
        from transformers import AutoModelForImageSegmentation

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.resolution = tuple(resolution)
        self.half_precision = bool(half_precision) and str(device).startswith("cuda")

        is_local = Path(model_path).exists()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local,
        ).to(self.device).eval()
        if self.half_precision:
            self.model = self.model.half()

        # BiRefNet reference preprocessing: resize -> ToTensor -> ImageNet norm
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 RGB ``[H,W,3]`` -> uint8 alpha ``[H,W]``."""
        h, w = rgb.shape[:2]
        net_h, net_w = self.resolution

        x = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = (x - self._mean) / self._std
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        if self.half_precision:
            x = x.half()

        out = self.model(x)
        logits = out[-1] if isinstance(out, (tuple, list)) else out
        prob = torch.sigmoid(logits)[0, 0].float()
        prob = torch.nn.functional.interpolate(
            prob[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return (prob.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)

    def matte_frames(
        self,
        frames_rgb: np.ndarray,
        threshold: Optional[int] = 127,
        feather: int = 0,
    ) -> np.ndarray:
        """Composite a uint8 clip ``[T,H,W,3]`` onto black using the predicted alpha.

        ``threshold=None`` keeps the soft alpha (smoother edges); the released
        checkpoints were trained on hard-thresholded masks, which is the default.
        """
        out = []
        for frame in frames_rgb:
            a = self.alpha(frame)
            if threshold is not None:
                _, a = cv2.threshold(a, int(threshold), 255, cv2.THRESH_BINARY)
            if feather > 0:
                k = int(feather) * 2 + 1
                a = cv2.GaussianBlur(a, (k, k), 0)
            alpha = (a.astype(np.float32) / 255.0)[..., None]
            out.append((frame.astype(np.float32) * alpha).round().clip(0, 255).astype(np.uint8))
        return np.stack(out, axis=0)

    def to(self, device: str) -> "PortraitMatter":
        self.device = device
        self.model.to(device)
        return self


_CACHE: dict = {}


def get_matter(
    model_path: str = DEFAULT_BIREFNET_ID,
    device: str = "cuda",
    **kwargs,
) -> PortraitMatter:
    """Process-wide cached :class:`PortraitMatter` (the model is ~900 MB)."""
    key = (str(model_path), str(device))
    if key not in _CACHE:
        _CACHE[key] = PortraitMatter(model_path=model_path, device=device, **kwargs)
    return _CACHE[key]
