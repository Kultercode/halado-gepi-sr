"""Központi konfiguráció: minden tunable paraméter és elérési út egy helyen."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data_raw"
DIV2K_DIR = DATA_DIR / "DIV2K"
GAMEPLAY_FRAMES_DIR = DATA_DIR / "gameplay_frames"
VIDEO_DIR = DATA_DIR / "videos"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
UPSCALED_DIR = OUTPUT_DIR / "upscaled"

for _d in (CHECKPOINT_DIR, OUTPUT_DIR, FIGURES_DIR, UPSCALED_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = DEVICE.type == "cuda"
SEED = 42


SCALE = 2
HR_PATCH_SIZE = 96
RGB_RANGE = 1.0


BATCH_SIZE = 16
NUM_EPOCHS = 100

LEARNING_RATE = 1e-4
LR_DECAY_STEP = 30
LR_DECAY_GAMMA = 0.5

LOSS_TYPE = "l1"
VAL_EVERY = 1
LOG_EVERY_STEPS = 50


SRCNN_CHANNELS = (64, 32)
SRCNN_KERNELS = (9, 5, 5)

EDSR_NUM_FEATURES = 64
EDSR_NUM_BLOCKS = 16
EDSR_RES_SCALE = 0.1


# A PSNR/SSIM előtt a szegélyt levágjuk, mert a bicubic a kép szélén
# pontatlan — ez az SR irodalom bevett konvenciója.
EVAL_SHAVE_BORDER = SCALE

VIDEO_BATCH_SIZE = 4

# A DIV2K hivatalos valid halmaza (100 kép) kettéosztva: az első VAL_SPLIT_END
# kép a validáció tanítás közben, a maradék a független teszt halmaz.
# (A hivatalos DIV2K test labelje nem publikus.)
VAL_SPLIT_END = 80
