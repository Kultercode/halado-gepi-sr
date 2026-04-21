"""Gyors sanity check: minden modul importálható, mindhárom modell fut."""

from __future__ import annotations

import sys

import torch

import config
from metrics import psnr, ssim
from models import AVAILABLE_MODELS, build_model
from utils import count_parameters, describe_device, set_seed


def main() -> int:
    set_seed(0)
    print("Eszköz:", describe_device(config.DEVICE))
    print("Scale:", config.SCALE)
    print("HR patch:", config.HR_PATCH_SIZE)
    print()

    lr_h = config.HR_PATCH_SIZE // config.SCALE
    lr = torch.rand(2, 3, lr_h, lr_h)
    expected_sr_h = lr_h * config.SCALE

    for name in AVAILABLE_MODELS:
        model = build_model(name, scale=config.SCALE)
        n_params = count_parameters(model)
        sr = model(lr)
        ok = sr.shape == (2, 3, expected_sr_h, expected_sr_h)
        status = "OK" if ok else "HIBA"
        print(f"  {name:<8}  params={n_params:>10,d}  "
              f"LR {tuple(lr.shape)} -> SR {tuple(sr.shape)}  [{status}]")
        if not ok:
            return 1

    hr = torch.rand(3, 64, 64)
    sr = hr + 0.01 * torch.randn_like(hr)
    sr = sr.clamp(0, 1)
    print()
    print(f"  PSNR (ugyanarra a képre kis zajjal): {psnr(sr, hr):.2f} dB")
    print(f"  SSIM (ugyanarra a képre kis zajjal): {ssim(sr, hr):.4f}")
    print(f"  PSNR (tök random képpel):            {psnr(torch.rand_like(hr), hr):.2f} dB")

    print("\n[OK] Minden alap működik.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
