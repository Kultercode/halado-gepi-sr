"""Super-Resolution dataset-ek HR képekből, on-the-fly LR generálással."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import iter_image_paths, load_image_rgb, np_to_tensor


def bicubic_downscale(hr: torch.Tensor, scale: int) -> torch.Tensor:
    hr_b = hr.unsqueeze(0)
    lr_b = F.interpolate(
        hr_b,
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    lr_b = lr_b.clamp(0.0, 1.0)
    return lr_b.squeeze(0)


def augment_pair(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[-1])
        hr = torch.flip(hr, dims=[-1])
    k = random.randint(0, 3)
    if k > 0:
        lr = torch.rot90(lr, k=k, dims=[-2, -1])
        hr = torch.rot90(hr, k=k, dims=[-2, -1])
    return lr, hr


class SRPatchDataset(Dataset):
    """Tanító dataset: random HR patch + hozzá generált LR pár."""

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int,
        hr_patch_size: int,
        augment: bool = True,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        if hr_patch_size % scale != 0:
            raise ValueError(
                f"hr_patch_size ({hr_patch_size}) nem osztható scale-lel ({scale})"
            )

        self.hr_paths = iter_image_paths(hr_dir)
        if len(self.hr_paths) == 0:
            raise FileNotFoundError(f"Nem találtam HR képet: {hr_dir}")

        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch or len(self.hr_paths)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_hr(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            self._cache[idx] = load_image_rgb(self.hr_paths[idx])
        return self._cache[idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Az index csak epoch-méretet jelent — a konkrét képet random választjuk.
        img_idx = random.randint(0, len(self.hr_paths) - 1)
        hr_img = self._load_hr(img_idx)

        h, w, _ = hr_img.shape
        ps = self.hr_patch_size

        if h < ps or w < ps:
            return self.__getitem__(index)

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        hr_crop = hr_img[top:top + ps, left:left + ps, :]

        hr = np_to_tensor(hr_crop)
        lr = bicubic_downscale(hr, self.scale)

        if self.augment:
            lr, hr = augment_pair(lr, hr)

        return lr, hr


class SRFullImageDataset(Dataset):
    """Teljes képes kiértékeléshez. start_idx/end_idx szeleteli a mappát."""

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int,
        crop_to_max: int | None = 512,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> None:
        super().__init__()
        all_paths = iter_image_paths(hr_dir)
        if len(all_paths) == 0:
            raise FileNotFoundError(f"Nem találtam HR képet: {hr_dir}")
        self.hr_paths = all_paths[start_idx:end_idx]
        if len(self.hr_paths) == 0:
            raise ValueError(
                f"A [{start_idx}:{end_idx}] szelet üres ({len(all_paths)} kép összesen)"
            )
        self.scale = scale
        self.crop_to_max = crop_to_max

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        hr_img = load_image_rgb(self.hr_paths[index])
        h, w, _ = hr_img.shape

        if self.crop_to_max is not None:
            m = self.crop_to_max
            if h > m or w > m:
                top = max(0, (h - m) // 2)
                left = max(0, (w - m) // 2)
                hr_img = hr_img[top:top + m, left:left + m, :]
                h, w, _ = hr_img.shape

        # Méret scale-lel oszthatóvá igazítása
        new_h = (h // self.scale) * self.scale
        new_w = (w // self.scale) * self.scale
        hr_img = hr_img[:new_h, :new_w, :]

        hr = np_to_tensor(hr_img)
        lr = bicubic_downscale(hr, self.scale)

        return lr, hr, self.hr_paths[index].name


class GameplayFramesDataset(SRFullImageDataset):
    """Alias a gameplay frame mappára (domain-specifikus kiértékelés)."""
    pass
