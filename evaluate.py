"""Modellek kiértékelése: PSNR/SSIM/inferencia idő + összehasonlító ábrák."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from dataset import SRFullImageDataset
from metrics import InferenceTimer, psnr, ssim
from models import build_model
from utils import count_parameters, describe_device
from visualize import plot_metric_bars, plot_sr_comparison, save_figure


def load_model(
    model_name: str,
    scale: int = config.SCALE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device = config.DEVICE,
) -> torch.nn.Module:
    model = build_model(model_name, scale=scale).to(device)
    if model_name == "bicubic":
        return model.eval()

    ckpt_path = checkpoint_dir / f"{model_name}_x{scale}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Nem találom a checkpointot: {ckpt_path}\n"
            f"Tanítsd be előbb: python train.py {model_name} --scale {scale}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Betöltve: {ckpt_path.name} (epoch {ckpt['epoch']}, val PSNR {ckpt['val_psnr']:.2f} dB)")
    return model.eval()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: SRFullImageDataset,
    device: torch.device,
    model_name: str,
    shave: int = config.EVAL_SHAVE_BORDER,
) -> dict:
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    timer = InferenceTimer(device)

    model.eval()
    for lr, hr, name in tqdm(dataset, desc=f"  {model_name:8s}", leave=False):
        lr_b = lr.unsqueeze(0).to(device)

        with timer:
            sr_b = model(lr_b)

        sr = sr_b.squeeze(0).cpu()
        psnr_vals.append(psnr(sr, hr, shave=shave))
        ssim_vals.append(ssim(sr, hr, shave=shave))

    time_summary = timer.summary()
    return {
        "psnr": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "inference_mean_s": time_summary["mean_s"],
        "inference_std_s": time_summary["std_s"],
        "num_samples": len(psnr_vals),
        "params": count_parameters(model),
    }


def evaluate_all(
    test_dir: str | Path,
    scale: int = config.SCALE,
    models: tuple[str, ...] = ("bicubic", "srcnn", "edsr"),
    device: torch.device = config.DEVICE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    figures_dir: Path = config.FIGURES_DIR,
    tag: str = "test",
    split: str = "test",
) -> dict:
    """split: "test" | "val" | "all"."""
    print("=" * 60)
    print(f"KIÉRTÉKELÉS: {tag}  (split={split}, ×{scale})")
    print("=" * 60)
    print(f"  Eszköz:   {describe_device(device)}")

    if split == "test":
        dataset = SRFullImageDataset(
            hr_dir=test_dir, scale=scale, crop_to_max=1024,
            start_idx=config.VAL_SPLIT_END,
        )
    elif split == "val":
        dataset = SRFullImageDataset(
            hr_dir=test_dir, scale=scale, crop_to_max=1024,
            end_idx=config.VAL_SPLIT_END,
        )
    elif split == "all":
        dataset = SRFullImageDataset(hr_dir=test_dir, scale=scale, crop_to_max=1024)
    else:
        raise ValueError(f"Ismeretlen split: {split!r}")
    print(f"  Képek:    {len(dataset)}")

    results: dict[str, dict] = {}
    for name in models:
        print(f"\n--- {name.upper()} ---")
        model = load_model(name, scale=scale, checkpoint_dir=checkpoint_dir, device=device)
        results[name] = evaluate_model(model, dataset, device, name)
        m = results[name]
        print(
            f"  PSNR: {m['psnr']:.2f} ± {m['psnr_std']:.2f} dB  |  "
            f"SSIM: {m['ssim']:.4f} ± {m['ssim_std']:.4f}  |  "
            f"t: {m['inference_mean_s']*1000:.1f} ms/img  |  "
            f"params: {m['params']:,}"
        )

    results_path = figures_dir.parent / f"results_{tag}_x{scale}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrikák mentve: {results_path}")

    fig = plot_metric_bars(
        results,
        metrics=("psnr", "ssim"),
        title=f"Modell-összehasonlítás ({tag}, ×{scale})",
    )
    save_figure(fig, figures_dir / f"bars_{tag}_x{scale}.png")
    print(f"Bar chart mentve: {figures_dir / f'bars_{tag}_x{scale}.png'}")

    _visualize_examples(
        dataset, models, scale, device, checkpoint_dir, figures_dir, tag,
        num_examples=min(3, len(dataset)),
    )

    print("\n" + "-" * 72)
    print(f"{'Modell':<12} {'PSNR (dB)':>12} {'SSIM':>10} {'t (ms)':>10} {'Params':>12}")
    print("-" * 72)
    for name, m in results.items():
        print(
            f"{name:<12} {m['psnr']:>12.2f} {m['ssim']:>10.4f} "
            f"{m['inference_mean_s']*1000:>10.1f} {m['params']:>12,}"
        )
    print("-" * 72)

    return results


@torch.no_grad()
def _visualize_examples(
    dataset: SRFullImageDataset,
    model_names: tuple[str, ...],
    scale: int,
    device: torch.device,
    checkpoint_dir: Path,
    figures_dir: Path,
    tag: str,
    num_examples: int = 3,
) -> None:
    models = {n: load_model(n, scale=scale, checkpoint_dir=checkpoint_dir, device=device)
              for n in model_names}

    n = len(dataset)
    indices = np.linspace(0, n - 1, num_examples, dtype=int)

    for i, idx in enumerate(indices):
        lr, hr, name = dataset[int(idx)]
        lr_b = lr.unsqueeze(0).to(device)

        predictions: dict[str, torch.Tensor] = {}
        for mname, model in models.items():
            sr = model(lr_b).squeeze(0).cpu()
            predictions[mname] = sr

        H, W = hr.shape[-2:]
        zw, zh = W // 4, H // 4
        zy, zx = (H - zh) // 2, (W - zw) // 2

        fig = plot_sr_comparison(
            lr, hr, predictions,
            title=f"{tag} – {Path(name).stem}  (×{scale})",
            zoom_box=(zy, zx, zh, zw),
        )
        save_figure(fig, figures_dir / f"compare_{tag}_x{scale}_{i+1:02d}.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SR modellek kiértékelése.")
    parser.add_argument("--scale", type=int, default=config.SCALE)
    parser.add_argument("--test-dir", type=Path,
                        default=config.DIV2K_DIR / "DIV2K_valid_HR")
    parser.add_argument("--tag", default="div2k_test")
    parser.add_argument(
        "--split",
        choices=["test", "val", "all"],
        default="test",
        help="'test' = [VAL_SPLIT_END:] elkülönített teszt, "
             "'val' = [:VAL_SPLIT_END] tanítási val, "
             "'all' = teljes mappa (pl. gameplay frame-ekhez).",
    )
    parser.add_argument("--models", nargs="+",
                        default=["bicubic", "srcnn", "edsr"])
    args = parser.parse_args()

    evaluate_all(
        test_dir=args.test_dir,
        scale=args.scale,
        models=tuple(args.models),
        tag=args.tag,
        split=args.split,
    )


if __name__ == "__main__":
    main()
