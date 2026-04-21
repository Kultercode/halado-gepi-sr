"""Egy modell (SRCNN vagy EDSR) tanítása a DIV2K tanító halmazon."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import SRPatchDataset, SRFullImageDataset
from metrics import psnr
from models import build_model
from utils import count_parameters, describe_device, set_seed
from visualize import plot_training_curves, save_figure


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for lr, hr in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)
        loss = loss_fn(sr, hr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataset: SRFullImageDataset,
    device: torch.device,
    max_samples: int = 20,
) -> float:
    model.eval()
    psnr_values: list[float] = []

    n = min(len(dataset), max_samples)
    for i in range(n):
        lr, hr, _ = dataset[i]
        lr = lr.unsqueeze(0).to(device)
        sr = model(lr).squeeze(0).cpu()
        psnr_values.append(psnr(sr, hr, shave=config.EVAL_SHAVE_BORDER))

    return sum(psnr_values) / max(len(psnr_values), 1)


def train_model(
    model_name: str,
    train_hr_dir: str | Path,
    val_hr_dir: str | Path,
    scale: int = config.SCALE,
    batch_size: int = config.BATCH_SIZE,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    patch_size: int = config.HR_PATCH_SIZE,
    samples_per_epoch: int | None = None,
    device: torch.device = config.DEVICE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    figures_dir: Path = config.FIGURES_DIR,
) -> dict:
    set_seed(config.SEED)

    print("=" * 60)
    print(f"TANÍTÁS: {model_name.upper()}  (×{scale})")
    print("=" * 60)
    print(f"  Eszköz:        {describe_device(device)}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Epochok:       {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  HR patch:      {patch_size}x{patch_size}")
    print(f"  LR patch:      {patch_size // scale}x{patch_size // scale}")

    train_ds = SRPatchDataset(
        hr_dir=train_hr_dir,
        scale=scale,
        hr_patch_size=patch_size,
        augment=True,
        samples_per_epoch=samples_per_epoch,
    )
    # Csak a val szeletet látja, a test részt nem.
    val_ds = SRFullImageDataset(
        hr_dir=val_hr_dir, scale=scale, crop_to_max=512,
        end_idx=config.VAL_SPLIT_END,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    print(f"  Train képek:   {len(train_ds.hr_paths)} (samples/epoch: {len(train_ds)})")
    print(f"  Val képek:     {len(val_ds)}")

    model = build_model(model_name, scale=scale).to(device)
    n_params = count_parameters(model)
    print(f"  Paraméterek:   {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_GAMMA,
    )

    if config.LOSS_TYPE == "l1":
        loss_fn = nn.L1Loss()
    elif config.LOSS_TYPE == "l2":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Ismeretlen LOSS_TYPE: {config.LOSS_TYPE}")

    history = {"train_loss": [], "val_psnr": [], "lr": [], "epoch_time_s": []}
    best_psnr = -float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, num_epochs
        )

        val_ps = validate(model, val_ds, device) if epoch % config.VAL_EVERY == 0 else history["val_psnr"][-1] if history["val_psnr"] else 0.0

        scheduler.step()
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_ps)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time_s"].append(elapsed)

        is_best = val_ps > best_psnr
        if is_best:
            best_psnr = val_ps
            best_epoch = epoch
            ckpt_path = checkpoint_dir / f"{model_name}_x{scale}_best.pt"
            torch.save({
                "model_name": model_name,
                "scale": scale,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_psnr": val_ps,
            }, ckpt_path)

        print(
            f"Epoch {epoch:3d}/{num_epochs}: "
            f"loss={train_loss:.4f}  val_PSNR={val_ps:.4f} dB  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"t={elapsed:.1f}s  "
            f"{'** best **' if is_best else ''}"
        )

    history["best_val_psnr"] = best_psnr
    history["best_epoch"] = best_epoch
    history["total_params"] = n_params

    history_path = checkpoint_dir / f"{model_name}_x{scale}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory mentve: {history_path}")

    fig = plot_training_curves(history, title=f"{model_name.upper()} (×{scale}) tanítás")
    fig_path = figures_dir / f"training_{model_name}_x{scale}.png"
    save_figure(fig, fig_path)
    print(f"Ábra mentve:   {fig_path}")

    print(f"\nLegjobb val PSNR: {best_psnr:.2f} dB (epoch {best_epoch})")
    return history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SR modell tanítása.")
    parser.add_argument("model", choices=["srcnn", "edsr"])
    parser.add_argument("--scale", type=int, default=config.SCALE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--train-dir", type=Path,
                        default=config.DIV2K_DIR / "DIV2K_train_HR")
    parser.add_argument("--val-dir", type=Path,
                        default=config.DIV2K_DIR / "DIV2K_valid_HR")
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        train_hr_dir=args.train_dir,
        val_hr_dir=args.val_dir,
        scale=args.scale,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        samples_per_epoch=args.samples_per_epoch,
    )


if __name__ == "__main__":
    main()
