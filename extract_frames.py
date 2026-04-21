"""CLI: gameplay videókból frame-ek kinyerése."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from utils import extract_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Videóból / videó mappából frame-ek kinyerése."
    )
    parser.add_argument("input", type=Path,
                        help="Bemeneti videófájl vagy mappa.")
    parser.add_argument("--output-dir", type=Path,
                        default=config.GAMEPLAY_FRAMES_DIR)
    parser.add_argument("--every-n", type=int, default=60,
                        help="Minden hányadik frame (60fps-nél 60 = 1 frame/s).")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Felső korlát videónként.")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"])
    args = parser.parse_args()

    input_path: Path = args.input

    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        exts = ("mp4", "mkv", "mov", "avi", "webm")
        videos = []
        for e in exts:
            videos.extend(input_path.glob(f"*.{e}"))
            videos.extend(input_path.glob(f"*.{e.upper()}"))
        videos = sorted(videos)
        if not videos:
            raise SystemExit(f"Nem találtam videót: {input_path}")
    else:
        raise SystemExit(f"Input nem létezik: {input_path}")

    print(f"Feldolgozandó videók: {len(videos)}")
    print(f"Kimeneti mappa:       {args.output_dir}")
    print()

    total = 0
    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video.name}")
        out_dir = args.output_dir / video.stem
        written = extract_frames(
            video_path=video,
            output_dir=out_dir,
            every_n=args.every_n,
            max_frames=args.max_frames,
            image_ext=args.ext,
        )
        total += len(written)
        print()

    print(f"Összes kinyert frame: {total}")
    print(f"A frame-ek itt:       {args.output_dir}")


if __name__ == "__main__":
    main()
