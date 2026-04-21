"""Segédfüggvények: seeding, kép I/O, videó frame extrakció."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    # A ROCm-es PyTorch is 'cuda' típust jelent — a hip / cuda attribútumból
    # derítjük ki, melyik backend fut.
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            hip = getattr(torch.version, "hip", None)
            cuda = getattr(torch.version, "cuda", None)
            backend = f"ROCm {hip}" if hip else f"CUDA {cuda}"
            return f"{name} ({backend})"
        except Exception:
            return "cuda device"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def load_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image_rgb(array: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.dtype != np.uint8:
        raise ValueError(f"uint8-at vártam, ez viszont {img.dtype}")
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).contiguous()


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (t * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    return arr


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 60,
    max_frames: int | None = None,
    prefix: str = "frame",
    image_ext: str = "png",
) -> list[Path]:
    """Minden every_n-edik frame-et kiírja PNG-ként. Az egymás utáni frame-ek
    szinte azonosak, tanulási szempontból redundánsak, ezért ritkítunk."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Videó nem található: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nem sikerült megnyitni a videót: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"  Videó: {video_path.name}")
    print(f"  FPS:   {fps:.2f}, összes frame: {total_frames}")
    print(f"  Kinyerés: minden {every_n}. frame, max {max_frames or 'korlátlan'}")

    written: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out_path = output_dir / f"{prefix}_{saved_idx:05d}.{image_ext}"
            save_image_rgb(frame_rgb, out_path)
            written.append(out_path)
            saved_idx += 1

            if max_frames is not None and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"  Kinyert frame-ek: {saved_idx} db -> {output_dir}")
    return written


def iter_image_paths(directory: str | Path, exts: Iterable[str] = ("png", "jpg", "jpeg")) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    paths: list[Path] = []
    for ext in exts:
        paths.extend(directory.glob(f"*.{ext}"))
        paths.extend(directory.glob(f"*.{ext.upper()}"))

    # Windows case-insensitive FS-en a kis/nagybetűs globok ugyanazt duplázzák.
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in sorted(paths):
        resolved = str(p.resolve())
        key = resolved.lower() if os.name == "nt" else resolved
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
