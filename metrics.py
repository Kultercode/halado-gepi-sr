"""PSNR, SSIM és inferencia idő metrikák."""

from __future__ import annotations

import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _tensor_to_hwc_numpy(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        if x.size(0) != 1:
            raise ValueError("Batch metrikát külön-külön kell számolni.")
        x = x.squeeze(0)
    if x.dim() != 3 or x.size(0) != 3:
        raise ValueError(f"Vártam (3, H, W)-t, kaptam {tuple(x.shape)}")
    return x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def shave_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    return img[border:-border, border:-border, :]


def psnr(sr: torch.Tensor, hr: torch.Tensor, shave: int = 0) -> float:
    sr_np = _tensor_to_hwc_numpy(sr)
    hr_np = _tensor_to_hwc_numpy(hr)
    if shave > 0:
        sr_np = shave_border(sr_np, shave)
        hr_np = shave_border(hr_np, shave)
    return float(sk_psnr(hr_np, sr_np, data_range=1.0))


def ssim(sr: torch.Tensor, hr: torch.Tensor, shave: int = 0) -> float:
    sr_np = _tensor_to_hwc_numpy(sr)
    hr_np = _tensor_to_hwc_numpy(hr)
    if shave > 0:
        sr_np = shave_border(sr_np, shave)
        hr_np = shave_border(hr_np, shave)
    return float(
        sk_ssim(
            hr_np,
            sr_np,
            data_range=1.0,
            channel_axis=-1,
        )
    )


class InferenceTimer:
    """Context manager inferencia idő méréséhez. GPU-n szinkronizál."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.times: list[float] = []

    def reset(self) -> None:
        self.times.clear()

    def __enter__(self) -> "InferenceTimer":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)

    def summary(self) -> dict:
        if not self.times:
            return {"mean_s": 0.0, "std_s": 0.0, "n": 0}
        arr = np.array(self.times)
        return {
            "mean_s": float(arr.mean()),
            "std_s": float(arr.std()),
            "min_s": float(arr.min()),
            "max_s": float(arr.max()),
            "n": len(arr),
        }
