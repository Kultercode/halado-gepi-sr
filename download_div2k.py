"""DIV2K dataset letöltő: 800 train HR + 100 valid HR kép (~3.6 GB)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

import config
from utils import human_bytes


DIV2K_URLS = {
    "DIV2K_train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk = 1024 * 64
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                pbar.update(len(data))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"  kibontás {zip_path.name}"):
            zf.extract(m, extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(description="DIV2K letöltése és kibontása.")
    parser.add_argument("--output-dir", type=Path, default=config.DIV2K_DIR)
    parser.add_argument("--keep-zips", action="store_true",
                        help="Ne törölje a ZIP-eket kibontás után.")
    parser.add_argument("--splits", nargs="+",
                        default=list(DIV2K_URLS.keys()),
                        choices=list(DIV2K_URLS.keys()))
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Ez a script a következőket tölti le:")
    for s in args.splits:
        print(f"  - {s}  ({DIV2K_URLS[s]})")
    print(f"Cél mappa: {out_dir.resolve()}")
    print("A teljes letöltés ~3.6 GB, kibontva még kb. annyi hely kell.")
    resp = input("Folytatjuk? [y/N] ").strip().lower()
    if resp not in ("y", "yes", "igen"):
        print("Megszakítva.")
        sys.exit(0)

    for split in args.splits:
        target_dir = out_dir / split
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[{split}] már kibontva itt: {target_dir} — kihagyva")
            continue

        zip_path = out_dir / f"{split}.zip"

        if zip_path.exists():
            print(f"[{split}] ZIP már létezik: {zip_path} ({human_bytes(zip_path.stat().st_size)})")
        else:
            print(f"\n[{split}] letöltés...")
            download_file(DIV2K_URLS[split], zip_path)

        print(f"[{split}] kibontás...")
        extract_zip(zip_path, out_dir)

        if not args.keep_zips:
            zip_path.unlink()

    print("\nKész. Mappaszerkezet:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
