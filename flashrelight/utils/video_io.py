# flashrelight/utils/video_io.py
#
# Video / image / env-map IO helpers used by the FlashRelight inference code.
# Everything here is numpy/opencv/imageio only -- no torch model dependency.
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

# OpenCV only decodes .exr when this flag is set *before* cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".MP4", ".MOV", ".AVI", ".MKV"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
HDR_EXTS = {".exr", ".hdr", ".EXR", ".HDR"}
DEFAULT_ENV_SCALE = 0.1

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def resample_sequence(items: Sequence, target_len: int) -> List:
    """Uniformly resample / pad a sequence to exactly ``target_len`` items."""
    n = len(items)
    if n == 0:
        raise ValueError("empty sequence")
    if n == 1:
        return [items[0]] * int(target_len)
    if n >= target_len:
        idx = np.round(np.linspace(0, n - 1, num=target_len)).astype(np.int64)
        return [items[i] for i in idx]
    return list(items) + [items[-1]] * (target_len - n)


def center_crop_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize to (target_h, target_w)."""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    interp = cv2.INTER_AREA if (img.shape[0] > target_h) else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def _to_rgb3(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame


# --------------------------------------------------------------------------- #
# video / image reading
# --------------------------------------------------------------------------- #
def probe_fps(video_path: PathLike, default: float = 16.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return float(fps) if fps and fps > 1e-6 else default


def read_video_frames_bgr(
    video_path: PathLike,
    max_seconds: Optional[float] = None,
    start_seconds: float = 0.0,
) -> List[np.ndarray]:
    """Decode a video into a list of BGR uint8 frames (optionally time-limited)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-6:
        src_fps = 30.0

    skip = int(max(0.0, start_seconds) * src_fps)
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = max(1, int(round(max_seconds * src_fps)))

    frames: List[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= skip:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        i += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"empty video: {video_path}")
    return frames


def read_video_rgb(
    video_path: PathLike,
    num_frames: int = 49,
    target_h: int = 832,
    target_w: int = 480,
    max_seconds: Optional[float] = None,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """Read a video as uint8 ``[num_frames, target_h, target_w, 3]`` RGB."""
    frames_bgr = read_video_frames_bgr(video_path, max_seconds=max_seconds,
                                       start_seconds=start_seconds)
    frames_bgr = resample_sequence(frames_bgr, num_frames)
    out = [center_crop_resize(cv2.cvtColor(_to_rgb3(f), cv2.COLOR_BGR2RGB), target_h, target_w)
           for f in frames_bgr]
    return np.stack(out, axis=0).astype(np.uint8)


def read_image_rgb(image_path: PathLike, target_h: int = 832, target_w: int = 480) -> np.ndarray:
    img = np.asarray(imageio.v2.imread(str(image_path)))
    img = _to_rgb3(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return center_crop_resize(img, target_h, target_w)


def read_source_rgb(
    path: PathLike,
    num_frames: int = 49,
    target_h: int = 832,
    target_w: int = 480,
    max_seconds: Optional[float] = None,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """Read a video *or* a still image into a uint8 ``[T,H,W,3]`` clip."""
    suffix = Path(path).suffix
    if suffix in IMAGE_EXTS:
        frame = read_image_rgb(path, target_h, target_w)
        return np.repeat(frame[None, ...], num_frames, axis=0)
    return read_video_rgb(path, num_frames, target_h, target_w,
                          max_seconds=max_seconds, start_seconds=start_seconds)


# --------------------------------------------------------------------------- #
# env-map reading
# --------------------------------------------------------------------------- #
def _read_hdr_rgb(path: Path) -> np.ndarray:
    # imdecode also works with non-ASCII filenames on Windows.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if image is None:
        raise ValueError(f"failed to decode HDR image: {path}")
    return cv2.cvtColor(_to_rgb3(image), cv2.COLOR_BGR2RGB).astype(np.float32)


def validate_env_scale(env_scale: float) -> float:
    scale = float(env_scale)
    if not np.isfinite(scale) or not 0 <= scale <= np.finfo(np.float32).max:
        raise ValueError("env_scale must be a finite, non-negative float32 value")
    return scale


def _rotate_env_panorama(image: np.ndarray, num_frames: int, speed: float, fps: float) -> np.ndarray:
    """Yaw a linear RGB panorama. Positive speed moves highlights to the right."""
    width = image.shape[1]
    degrees_per_frame = (speed / fps) % 360.0
    sequence = np.empty((num_frames, *image.shape), dtype=np.float32)
    for i in range(num_frames):
        offset = (i * degrees_per_frame % 360.0) * width / 360.0
        columns = int(offset)
        fraction = offset - columns
        shifted = np.roll(image, columns, axis=1)
        # Interpolate across the panorama seam, always from the original image.
        sequence[i] = shifted if fraction == 0 else (
            (1.0 - fraction) * shifted + fraction * np.roll(shifted, 1, axis=1))
    return sequence


def read_env_hdr(
    env_path: PathLike,
    num_frames: int = 49,
    target_h: int = 64,
    target_w: int = 128,
    *,
    env_rotation_speed: float = 0.0,
    fps: float = 16,
    env_scale: float = 1.0,
) -> np.ndarray:
    """Read a (dynamic) HDR environment map as float32 ``[T,3,H,W]`` radiance.

    Accepts
      * a folder of ``*.exr`` / ``*.hdr`` frames (sorted by filename),
      * a single equirectangular ``.exr`` / ``.hdr`` panorama,
      * a ``.npy`` array shaped ``[T,3,H,W]``, ``[T,H,W,3]``, ``[3,H,W]`` or ``[H,W,3]``.

    A single frame stays fixed by default. Set ``env_rotation_speed`` in
    degrees per output second to rotate it around the vertical axis.
    This I/O helper reads stored radiance by default. Model inference passes
    its exposure multiplier to scale linear RGB after resizing and before rotation.
    """
    env_path = Path(env_path)
    scale = validate_env_scale(env_scale)
    if num_frames < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("num_frames and fps must be positive and finite")
    if not np.isfinite(env_rotation_speed):
        raise ValueError("env_rotation_speed must be finite")

    if env_path.is_dir():
        items = sorted(p for p in env_path.iterdir() if p.is_file() and p.suffix.lower() in HDR_EXTS)
        if not items:
            raise ValueError(f"no .exr/.hdr frames found in {env_path}")
    elif env_path.suffix.lower() in HDR_EXTS:
        items = [env_path]
    elif env_path.suffix.lower() == ".npy":
        raw = np.load(env_path, allow_pickle=False)
        if raw.ndim == 3:                                   # single frame
            raw = raw[None, ...]
        if raw.ndim != 4:
            raise ValueError(f"unsupported env npy shape: {raw.shape}")
        if raw.shape[1] == 3:                               # [T,3,H,W]
            raw = raw.transpose(0, 2, 3, 1)
        elif raw.shape[-1] != 3:
            raise ValueError(f"cannot infer channel dim for env npy: {raw.shape}")
        items = raw
    else:
        raise ValueError(f"unsupported env-map input: {env_path}")

    if env_rotation_speed and len(items) != 1:
        raise ValueError("env_rotation_speed requires a single panorama. Use 0 for an existing sequence.")
    selected = items if len(items) == 1 else resample_sequence(items, num_frames)
    frames = []
    for item in selected:
        rgb = _read_hdr_rgb(item) if isinstance(item, Path) else item.astype(np.float32, copy=False)
        frames.append(cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA))
    arr = np.stack(frames)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(arr, 0.0, out=arr)
    if scale != 1.0:
        arr *= np.float32(scale)
    if len(items) == 1:
        if env_rotation_speed:
            arr = _rotate_env_panorama(arr[0], num_frames, env_rotation_speed, fps)
        else:
            arr = np.repeat(arr, num_frames, axis=0)
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))   # [T,3,H,W]


def env_hdr_to_u8(env_hdr: np.ndarray, target_h: int, target_w: Optional[int] = None) -> np.ndarray:
    """Tone-map HDR radiance ``[T,3,H,W]`` to a displayable uint8 ``[T,h,w,3]``."""
    out = []
    for i in range(env_hdr.shape[0]):
        img = env_hdr[i].transpose(1, 2, 0)
        img = img / (1.0 + img)                              # Reinhard
        img = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)     # gamma
        img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
        h0, w0 = img.shape[:2]
        w = target_w if target_w is not None else max(1, int(round(w0 * (target_h / max(1, h0)))))
        out.append(cv2.resize(img, (w, target_h), interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0)


def letterbox(frames_u8: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Fit a clip inside ``target_h x target_w`` on black, preserving aspect ratio."""
    frames_u8 = np.asarray(frames_u8)
    h, w = frames_u8.shape[1:3]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    y0, x0 = (target_h - new_h) // 2, (target_w - new_w) // 2
    out = np.zeros((frames_u8.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for i, f in enumerate(frames_u8):
        out[i, y0:y0 + new_h, x0:x0 + new_w] = cv2.resize(
            f, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


def generate_direction_map(h: int, w: int) -> np.ndarray:
    """Per-pixel unit direction of an equirectangular env map -> ``[3,H,W]``."""
    theta = np.linspace(0, np.pi, h)
    phi = np.linspace(0, 2 * np.pi, w)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.cos(theta_grid)
    z = np.sin(theta_grid) * np.sin(phi_grid)
    return np.stack([x, y, z], axis=0).astype(np.float32)


def env_hdr_to_3buffer(env_hdr: np.ndarray) -> tuple:
    """Build the three env buffers consumed by the relighting model.

    Returns ``(ldr, log, direction)``, each float32 ``[3,T,H,W]``:
      * ldr : Reinhard tone-mapped radiance, in [0,1]
      * log : ``log1p`` radiance normalised by its max, in [0,1]
      * dir : equirectangular direction map (constant over time), in [-1,1]
    """
    env = env_hdr.transpose(0, 2, 3, 1).astype(np.float32)        # [T,H,W,3]
    ldr = np.clip(env / (1.0 + env), 0.0, 1.0)
    log_val = np.log1p(env)
    log_val = log_val / (float(np.max(log_val)) + 1e-6)
    t, h, w, _ = ldr.shape
    dir_map = generate_direction_map(h, w).transpose(1, 2, 0)      # [H,W,3]
    dir_stack = np.tile(dir_map[None, ...], (t, 1, 1, 1))
    return (
        np.ascontiguousarray(ldr.transpose(3, 0, 1, 2)),
        np.ascontiguousarray(log_val.transpose(3, 0, 1, 2)),
        np.ascontiguousarray(dir_stack.transpose(3, 0, 1, 2)),
    )


# --------------------------------------------------------------------------- #
# writing / compositing
# --------------------------------------------------------------------------- #
def save_mp4(frames_u8: np.ndarray, out_path: PathLike, fps: int = 16, quality: int = 8) -> str:
    """Write uint8 ``[T,H,W,3]`` frames as a browser-friendly H.264 mp4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_u8 = np.asarray(frames_u8)
    # H.264 requires even dimensions
    h, w = frames_u8.shape[1:3]
    if h % 2 or w % 2:
        frames_u8 = frames_u8[:, :h - (h % 2), :w - (w % 2)]
    try:
        imageio.mimwrite(str(out_path), frames_u8, fps=fps, codec="libx264",
                         quality=quality, pixelformat="yuv420p",
                         macro_block_size=1, output_params=["-loglevel", "error"])
    except Exception:  # pragma: no cover - fall back to the default writer
        imageio.mimwrite(str(out_path), frames_u8, fps=fps, quality=quality)
    return str(out_path)


def add_label(frames_u8: np.ndarray, text: str) -> np.ndarray:
    """Burn a short caption into the top-left corner of every frame."""
    arr = np.asarray(frames_u8).copy()
    h, w = arr.shape[1], arr.shape[2]
    scale = max(0.4, min(w, h) / 480.0)
    thick = max(1, int(round(scale * 2)))
    org = (max(8, int(0.02 * w)), max(20, int(0.07 * h)))
    for i in range(arr.shape[0]):
        cv2.putText(arr[i], text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(arr[i], text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return arr


def make_text_panel(text: str, height: int, width: int, num_frames: int) -> np.ndarray:
    """Render a wrapped text prompt as a static video panel."""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "TEXT PROMPT", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    scale = max(0.35, min(width, height) / 900.0)
    max_chars = max(12, int(width / (18 * scale)))
    words, line, lines = (text or "").split(), "", []
    for word in words:
        probe = f"{line} {word}".strip()
        if len(probe) > max_chars:
            lines.append(line)
            line = word
        else:
            line = probe
    if line:
        lines.append(line)
    y = 70
    for ln in lines[:24]:
        cv2.putText(canvas, ln, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += int(26 * max(1.0, scale))
    return np.repeat(canvas[None, ...], num_frames, axis=0)


def hstack_panels(panels: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate equal-length uint8 clips side by side."""
    panels = [np.asarray(p) for p in panels if p is not None and len(p)]
    if not panels:
        raise ValueError("no panels to stack")
    t = min(p.shape[0] for p in panels)
    h = max(p.shape[1] for p in panels)
    resized = []
    for p in panels:
        p = p[:t]
        if p.shape[1] != h:
            w = max(1, int(round(p.shape[2] * h / p.shape[1])))
            p = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in p], axis=0)
        resized.append(p)
    return np.concatenate(resized, axis=2)


def delight_comparison(source_u8: np.ndarray, delit_u8: np.ndarray) -> np.ndarray:
    """Show the source portrait and the delighted result side by side."""
    return hstack_panels([add_label(source_u8, "SOURCE"), add_label(delit_u8, "DELIT")])


def list_media(directory: PathLike, exts: Optional[set] = None) -> List[Path]:
    exts = exts or (VIDEO_EXTS | IMAGE_EXTS)
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix in exts])
