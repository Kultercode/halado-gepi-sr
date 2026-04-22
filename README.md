# halado-gepi-sr

Image super-resolution (SR) projekt a *Haladó gépi tanulás* tárgy beadandójához.
Egy DLSS/FSR-jellegű kép- és videófelskálázó, három modell összehasonlításával:
**bicubic baseline**, **SRCNN** (VDSR-stílusú residual tanulással), és **EDSR baseline**.

- Framework: PyTorch
- Dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Skálázási tényező: 2× (4× is működne, csak át kell állítani a `config.py`-ben)

## Telepítés

A `torch` és `torchvision` csomagokat a hardvernek megfelelő index-ből kell telepíteni:

```bash
# NVIDIA CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD (ROCm, Linux):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# AMD (Windows, RDNA3/RDNA4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu


Ha ezek ezek közül egyik sem jó, akkor sima pip install torch torchvision is működhet
```

Utána a többi függőség:

```bash
pip install -r requirements.txt
```

Gyors ellenőrzés, hogy minden importál és a GPU látszik:

```bash
python smoke_test.py
```

## Használat

A projekt egyetlen belépési pontja a `main.py`. Minden részfeladat ezen keresztül érhető el.

### 1. Adatok előkészítése

DIV2K letöltése (~3.6 GB):

```bash
python main.py prepare --div2k
```

Opcionálisan saját videóból frame-ek kinyerése:

```bash
python main.py prepare --video path/to/video.mp4 --every-n 60 --max-frames 500
```

### 2. Tanítás

```bash
# SRCNN (~50 epoch, ~20 perc)
python main.py train srcnn --epochs 50 --batch-size 32 --learning-rate 3e-4

# EDSR (~50 epoch, ~20 perc)
python main.py train edsr --epochs 50 --batch-size 32 --learning-rate 2e-4
```

A legjobb validációs PSNR szerinti checkpointok a `checkpoints/` alá mentődnek,
a tanítási görbék pedig az `outputs/figures/` alá.

### 3. Kiértékelés

A DIV2K valid halmaz (100 kép) ketté van osztva: `[:80]` = validáció tanítás
közben, `[80:]` = elkülönített teszt halmaz.

```bash
# Teszt halmaz (default)
python main.py evaluate --tag div2k_test

# Validációs halmazon (összehasonlításhoz)
python main.py evaluate --tag div2k_val --split val

# Saját gameplay frame-eken
python main.py evaluate --tag gameplay --test-dir data_raw/gameplay_frames/videó_név --split all
```

### 4. Kép vagy videó felskálázása

Gyakorlati inferencia a tanított modellel:

```bash
python upscale.py image input.jpg output.png --model edsr --scale 2
python upscale.py video input.mp4 output.mp4 --model edsr --scale 2

# Nagy képeknél, ha Out Of Memory lenne:
python upscale.py image input.jpg output.png --model edsr --tile-size 256
```

### 5. Teljes pipeline egyben

SRCNN + EDSR tanítása és mindhárom modell kiértékelése (~35-40 perc):

```bash
python main.py all --epochs 50 --batch-size 32 --learning-rate 2e-4
```

## Projekt struktúra

```
main.py              # CLI entry point (prepare/train/evaluate/all)
config.py            # Hiperparaméterek, elérési utak
models.py            # Bicubic / SRCNN / EDSR modellek
dataset.py           # Patch-alapú tanító + teljes képes eval dataset
train.py             # Tanítási loop, checkpoint, LR scheduler
evaluate.py          # PSNR / SSIM / inference time metrikák
upscale.py           # Kép / videó inferencia CLI
metrics.py           # PSNR, SSIM, InferenceTimer
visualize.py         # Matplotlib ábrák
utils.py             # Seed, I/O, frame extrakció
extract_frames.py    # Videóból frame kinyerés
download_div2k.py    # DIV2K letöltő
smoke_test.py        # Gyors sanity check
```

Futás közben az alábbi mappák jönnek létre automatikusan:

```
data_raw/                # DIV2K képek + saját gameplay frame-ek
checkpoints/             # Betanított modellek (.pt) + history JSON-ok
outputs/
    figures/             # Tanítási görbék, bar chart, összehasonlító képek
    upscaled/            # Az upscale.py CLI kimenetei
```

## Modellek

| Modell  | Paraméterek | Leírás |
|---------|-------------|--------|
| Bicubic | 0           | Klasszikus interpoláció — baseline |
| SRCNN   | 69 251      | 3-rétegű CNN, VDSR-stílusú residual tanulással (Dong 2014 + Kim 2016) |
| EDSR    | 1 369 859   | 16 residual blokk, 64 feature (Lim et al., 2017 — baseline verzió) |

## Példa eredmények (2× skála, DIV2K teszt split, 20 kép)

| Modell  | PSNR (dB) | SSIM    | Inferencia (ms/kép) |
|---------|-----------|---------|---------------------|
| Bicubic | 30.01     | 0.8962  | 4.5                 |
| SRCNN   | **31.61** | **0.9207** | 16.4             |
| EDSR    | 30.99     | 0.9103  | 45.3                |

Mérési hardver: AMD Radeon RX 9070 XT (16 GB VRAM), ROCm 7.2, PyTorch.

## Előre betanított modellek

A betanított checkpointok (SRCNN és EDSR ×2) a
[`pretrained` branch](https://github.com/Kultercode/halado-gepi-sr/tree/pretrained)
alatt megtalálhatóak, a `checkpoints/` mappában. Klónozás után:

```bash
git checkout pretrained -- checkpoints/
python main.py evaluate --tag div2k_test
```

Így a tanítást nem kell lefuttatni, egyből futtatható a kiértékelés.

## Referenciák

- Dong et al., 2014 — *Learning a Deep Convolutional Network for Image Super-Resolution* (SRCNN)
- Kim et al., 2016 — *Accurate Image Super-Resolution Using Very Deep Convolutional Networks* (VDSR, residual learning)
- Lim et al., 2017 — *Enhanced Deep Residual Networks for Single Image Super-Resolution* (EDSR)
- Agustsson & Timofte, 2017 — *NTIRE 2017 Challenge on Single Image Super-Resolution* (DIV2K dataset)
