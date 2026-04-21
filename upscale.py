"""Inferencia CLI: kép vagy videó felskálázása egy betanított modellel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from evaluate import load_model
from utils import (describe_device, human_bytes, load_image_rgb,
                   np_to_tensor, save_image_rgb, tensor_to_np)


@torch.no_grad()
def upscale_tensor(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int,
    device: torch.device,
    tile_size: int | None = None,
    tile_overlap: int = 16,
) -> torch.Tensor:
    """Tile-alapú inferencia nagy képekhez, átfedéssel az él-artifaktok ellen."""
    model.eval()
    _, h, w = lr.shape

    if tile_size is None or (h <= tile_size and w <= tile_size):
        lr_b = lr.unsqueeze(0).to(device)
        sr = model(lr_b).squeeze(0).cpu()
        return sr

    sr_h = h * scale
    sr_w = w * scale
    sr = torch.zeros(3, sr_h, sr_w, dtype=torch.float32)
    weight = torch.zeros(1, sr_h, sr_w, dtype=torch.float32)

    stride = tile_size - tile_overlap
    def _positions(length: int, tile: int, step: int) -> list[int]:
        pos = list(range(0, max(length - tile, 0) + 1, step))
        if not pos or pos[-1] + tile < length:
            pos.append(max(length - tile, 0))
        return pos

    ys = _positions(h, tile_size, stride)
    xs = _positions(w, tile_size, stride)

    for y in ys:
        for x in xs:
            lr_tile = lr[:, y:y + tile_size, x:x + tile_size]
            lr_b = lr_tile.unsqueeze(0).to(device)
            sr_tile = model(lr_b).squeeze(0).cpu()

            sy, sx = y * scale, x * scale
            th, tw = sr_tile.shape[-2:]
            sr[:, sy:sy + th, sx:sx + tw] += sr_tile
            weight[:, sy:sy + th, sx:sx + tw] += 1.0

    sr = sr / weight.clamp(min=1e-6)
    return sr.clamp(0.0, 1.0)


def upscale_image(
    input_path: Path,
    output_path: Path,
    model_name: str,
    scale: int,
    tile_size: int | None,
    device: torch.device,
) -> None:
    print(f"  Bemenet:  {input_path}")
    img = load_image_rgb(input_path)
    h, w, _ = img.shape
    print(f"  LR méret: {w}×{h}")

    lr = np_to_tensor(img)
    model = load_model(model_name, scale=scale, device=device)

    t0 = time.perf_counter()
    sr = upscale_tensor(model, lr, scale, device, tile_size=tile_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sr_np = tensor_to_np(sr)
    save_image_rgb(sr_np, output_path)
    print(f"  SR méret: {sr_np.shape[1]}×{sr_np.shape[0]}")
    print(f"  Idő:      {elapsed:.2f} s")
    print(f"  Kimenet:  {output_path}")


def upscale_video(
    input_path: Path,
    output_path: Path,
    model_name: str,
    scale: int,
    tile_size: int | None,
    device: torch.device,
    max_frames: int | None = None,
) -> None:
    """Frame-by-frame felskálázás. Nem temporálisan konzisztens."""
    print(f"  Bemenet:  {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nem nyitható meg a videó: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out = w_in * scale
    h_out = h_in * scale

    print(f"  Input:    {w_in}×{h_in} @ {fps:.1f} fps, {total} frame")
    print(f"  Output:   {w_out}×{h_out}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Nem tudtam videó writer-t nyitni: {output_path}")

    model = load_model(model_name, scale=scale, device=device)

    total_to_process = min(total, max_frames) if max_frames else total
    pbar = tqdm(total=total_to_process, desc="  frame-ek", unit="f")
    t0 = time.perf_counter()

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lr = np_to_tensor(frame_rgb)

            sr = upscale_tensor(model, lr, scale, device, tile_size=tile_size)
            sr_np = tensor_to_np(sr)
            writer.write(cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            pbar.update(1)
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        pbar.close()

    elapsed = time.perf_counter() - t0
    size = output_path.stat().st_size if output_path.exists() else 0
    print(f"  Idő:      {elapsed:.1f} s  ({frame_idx/elapsed:.2f} fps)")
    print(f"  Kimenet:  {output_path}  ({human_bytes(size)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kép vagy videó felskálázása.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_img = sub.add_parser("image", help="Egy kép felskálázása.")
    p_img.add_argument("input", type=Path)
    p_img.add_argument("output", type=Path)

    p_vid = sub.add_parser("video", help="Egy videó felskálázása.")
    p_vid.add_argument("input", type=Path)
    p_vid.add_argument("output", type=Path)
    p_vid.add_argument("--max-frames", type=int, default=None,
                       help="Első N frame feldolgozása (gyors teszthez).")

    for p in (p_img, p_vid):
        p.add_argument("--model", choices=["bicubic", "srcnn", "edsr"], default="edsr")
        p.add_argument("--scale", type=int, default=config.SCALE)
        p.add_argument(
            "--tile-size",
            type=int,
            default=None,
            help="LR tile méret nagy képekhez (pl. 256). OOM esetén csökkentsd.",
        )

    args = parser.parse_args()
    device = config.DEVICE
    print(f"Eszköz: {describe_device(device)}")

    if args.mode == "image":
        upscale_image(
            args.input, args.output,
            model_name=args.model, scale=args.scale,
            tile_size=args.tile_size, device=device,
        )
    else:
        upscale_video(
            args.input, args.output,
            model_name=args.model, scale=args.scale,
            tile_size=args.tile_size, device=device,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
