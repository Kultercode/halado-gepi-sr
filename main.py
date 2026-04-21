#!/usr/bin/env python3
"""
Juhász Kristóf – Haladó gépi tanulás beadandó
=============================================
Kép super-resolution (DLSS/FSR-jellegű upscaler) összehasonlító program.

Modellek:
  1. Bicubic interpoláció
  2. SRCNN
  3. EDSR baseline

Adathalmaz:
  DIV2K, 800 tanító + 100 validációs HR kép, plusz saját
  gameplay frame-ek domain-specifikus kiértékeléshez.

Használat:
  python main.py prepare                      # adatok letöltése + frame kivonás
  python main.py train srcnn                  # SRCNN tanítása
  python main.py train edsr                   # EDSR tanítása
  python main.py evaluate                     # mindhárom modell értékelése DIV2K test-en
  python main.py evaluate --tag gameplay \\
         --test-dir data_raw/gameplay_frames  # saját frame-eken
  python main.py all                          # teljes futtatás (prepare kivételével)

Egyedi inferenciához ld. külön:
  python upscale.py image in.png out.png --model edsr
  python upscale.py video in.mp4 out.mp4 --model edsr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from utils import describe_device, set_seed


def cmd_prepare(args: argparse.Namespace) -> None:
    if args.div2k:
        from download_div2k import main as div2k_main
        print("=" * 60)
        print("DIV2K LETÖLTÉSE")
        print("=" * 60)
        sys.argv = ["download_div2k.py", "--output-dir", str(config.DIV2K_DIR)]
        div2k_main()

    if args.video is not None:
        from extract_frames import main as extract_main
        print("=" * 60)
        print("GAMEPLAY FRAME-EK KINYERÉSE")
        print("=" * 60)
        sys.argv = [
            "extract_frames.py",
            str(args.video),
            "--output-dir", str(config.GAMEPLAY_FRAMES_DIR),
            "--every-n", str(args.every_n),
            "--max-frames", str(args.max_frames),
        ]
        extract_main()


def cmd_train(args: argparse.Namespace) -> None:
    from train import train_model

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


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import evaluate_all

    evaluate_all(
        test_dir=args.test_dir,
        scale=args.scale,
        models=tuple(args.models),
        tag=args.tag,
        split=args.split,
    )


def cmd_all(args: argparse.Namespace) -> None:
    from train import train_model
    from evaluate import evaluate_all

    train_dir = args.train_dir
    val_dir = args.val_dir

    print("\n>>> 1/3: SRCNN tanítása\n")
    train_model(
        model_name="srcnn",
        train_hr_dir=train_dir,
        val_hr_dir=val_dir,
        scale=args.scale,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        samples_per_epoch=args.samples_per_epoch,
    )

    print("\n>>> 2/3: EDSR tanítása\n")
    train_model(
        model_name="edsr",
        train_hr_dir=train_dir,
        val_hr_dir=val_dir,
        scale=args.scale,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        samples_per_epoch=args.samples_per_epoch,
    )

    print("\n>>> 3/3: kiértékelés a DIV2K teszt szeletén\n")
    evaluate_all(
        test_dir=val_dir,
        scale=args.scale,
        models=("bicubic", "srcnn", "edsr"),
        tag="div2k_test",
        split="test",
    )

    gp_dir = config.GAMEPLAY_FRAMES_DIR
    if gp_dir.exists() and any(gp_dir.rglob("*.png")):
        print("\n>>> BÓNUSZ: kiértékelés saját gameplay frame-eken\n")
        gp_subdirs = [d for d in gp_dir.iterdir() if d.is_dir()]
        target = gp_subdirs[0] if gp_subdirs else gp_dir
        evaluate_all(
            test_dir=target,
            scale=args.scale,
            models=("bicubic", "srcnn", "edsr"),
            tag=f"gameplay_{target.name}",
            split="all",
        )
    else:
        print("\nNincs gameplay frame a data_raw/gameplay_frames/ alatt — kihagyva.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Adatok letöltése / előkészítése.")
    p_prep.add_argument("--div2k", action="store_true", help="DIV2K letöltése.")
    p_prep.add_argument("--video", type=Path, default=None,
                        help="Gameplay videó vagy mappa.")
    p_prep.add_argument("--every-n", type=int, default=60)
    p_prep.add_argument("--max-frames", type=int, default=500)
    p_prep.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Egy modell tanítása.")
    p_train.add_argument("model", choices=["srcnn", "edsr"])
    p_train.add_argument("--scale", type=int, default=config.SCALE)
    p_train.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p_train.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_train.add_argument("--samples-per-epoch", type=int, default=2000)
    p_train.add_argument("--train-dir", type=Path,
                         default=config.DIV2K_DIR / "DIV2K_train_HR")
    p_train.add_argument("--val-dir", type=Path,
                         default=config.DIV2K_DIR / "DIV2K_valid_HR")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Modellek összehasonlító értékelése.")
    p_eval.add_argument("--scale", type=int, default=config.SCALE)
    p_eval.add_argument("--test-dir", type=Path,
                        default=config.DIV2K_DIR / "DIV2K_valid_HR")
    p_eval.add_argument("--tag", default="div2k_test")
    p_eval.add_argument("--split", choices=["test", "val", "all"], default="test",
                        help="'test' = [VAL_SPLIT_END:], 'val' = [:VAL_SPLIT_END], "
                             "'all' = teljes mappa.")
    p_eval.add_argument("--models", nargs="+",
                        default=["bicubic", "srcnn", "edsr"])
    p_eval.set_defaults(func=cmd_evaluate)

    p_all = sub.add_parser("all", help="Teljes pipeline (tanítás + értékelés).")
    p_all.add_argument("--scale", type=int, default=config.SCALE)
    p_all.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p_all.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p_all.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_all.add_argument("--samples-per-epoch", type=int, default=2000)
    p_all.add_argument("--train-dir", type=Path,
                       default=config.DIV2K_DIR / "DIV2K_train_HR")
    p_all.add_argument("--val-dir", type=Path,
                       default=config.DIV2K_DIR / "DIV2K_valid_HR")
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    set_seed(config.SEED)
    print("=" * 60)
    print("SR program")
    print(f"Eszköz: {describe_device(config.DEVICE)}")
    print("=" * 60)

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
