"""Matplotlib ábrák: tanítási görbék, kép-kép összehasonlítás, metrika bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import tensor_to_np


plt.style.use("seaborn-v0_8-whitegrid")


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Tanítási görbék",
) -> plt.Figure:
    fig, (ax_loss, ax_psnr) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], "o-", color="steelblue", label="train")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("L1 loss")
    ax_loss.set_title("Tanítási loss")
    ax_loss.legend()

    if "val_psnr" in history and history["val_psnr"]:
        vpsnr = history["val_psnr"]
        ax_psnr.plot(np.arange(1, len(vpsnr) + 1), vpsnr, "o-", color="coral", label="val")
        ax_psnr.set_xlabel("Epoch")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.set_title("Validációs PSNR")
        ax_psnr.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_loss_comparison(
    histories: dict[str, list[float]],
    title: str = "Loss görbék összehasonlítása",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, loss in histories.items():
        ax.plot(np.arange(1, len(loss) + 1), loss, "-", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sr_comparison(
    lr: torch.Tensor,
    hr: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    title: str = "",
    zoom_box: tuple[int, int, int, int] | None = None,
) -> plt.Figure:
    """LR | HR | predikciók egymás mellett. zoom_box = (y, x, h, w) HR koordinátában."""
    n_cols = 2 + len(predictions)
    n_rows = 1 if zoom_box is None else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    lr_np = tensor_to_np(lr)
    hr_np = tensor_to_np(hr)

    axes[0, 0].imshow(lr_np)
    axes[0, 0].set_title(f"LR bemenet\n{lr.shape[-2]}×{lr.shape[-1]}")
    axes[0, 1].imshow(hr_np)
    axes[0, 1].set_title(f"HR ground truth\n{hr.shape[-2]}×{hr.shape[-1]}")
    for i, (name, pred) in enumerate(predictions.items()):
        axes[0, 2 + i].imshow(tensor_to_np(pred))
        axes[0, 2 + i].set_title(name)

    for ax in axes[0]:
        ax.axis("off")

    if zoom_box is not None:
        y, x, h, w = zoom_box

        # HR koordinátát LR-re vetítjük nearest-neighbor "zoommal".
        lr_h, lr_w = lr_np.shape[:2]
        hr_h, hr_w = hr_np.shape[:2]
        sy = max(hr_h // max(lr_h, 1), 1)
        sx = max(hr_w // max(lr_w, 1), 1)

        y0_lr = max(0, min(lr_h, y // sy))
        x0_lr = max(0, min(lr_w, x // sx))
        y1_lr = max(y0_lr + 1, min(lr_h, (y + h + sy - 1) // sy))
        x1_lr = max(x0_lr + 1, min(lr_w, (x + w + sx - 1) // sx))

        lr_zoom = lr_np[y0_lr:y1_lr, x0_lr:x1_lr]
        lr_zoom_up = np.repeat(np.repeat(lr_zoom, sy, axis=0), sx, axis=1)
        lr_zoom_up = lr_zoom_up[:h, :w]

        axes[1, 0].imshow(lr_zoom_up)
        axes[1, 0].set_title("LR zoom")
        axes[1, 1].imshow(hr_np[y:y + h, x:x + w])
        axes[1, 1].set_title("HR zoom")
        for i, (name, pred) in enumerate(predictions.items()):
            pred_np = tensor_to_np(pred)
            axes[1, 2 + i].imshow(pred_np[y:y + h, x:x + w])
            axes[1, 2 + i].set_title(f"{name} zoom")
        for ax in axes[1]:
            ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metric_bars(
    results: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = ("psnr", "ssim"),
    title: str = "Modell teljesítmény",
) -> plt.Figure:
    names = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#4c72b0", "#dd8452", "#55a467", "#c44e52"][:len(names)]

    for ax, metric in zip(axes, metrics):
        values = [results[n].get(metric, 0.0) for n in names]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}" if metric != "psnr" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
