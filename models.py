"""Super-Resolution modellek: Bicubic baseline, SRCNN, EDSR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BicubicUpsampler(nn.Module):
    """Egyszerű bicubic interpoláció, tanítható paraméter nélkül."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        sr = F.interpolate(
            lr,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        return sr.clamp(0.0, 1.0)


class SRCNN(nn.Module):
    """SRCNN (Dong et al., 2014) VDSR-stílusú (Kim et al., 2016) residual tanulással.

    RGB képen az eredeti paper-hű SRCNN színkonvergenciája lassú, mert a paper
    Y csatornán dolgozik. A VDSR által bevezetett pixel-térbeli residual
    tanulással (a háló a bicubic alaphoz képesti különbséget tanulja) ez a
    probléma megszűnik.
    """

    def __init__(
        self,
        scale: int,
        channels: tuple[int, int] = config.SRCNN_CHANNELS,
        kernels: tuple[int, int, int] = config.SRCNN_KERNELS,
    ) -> None:
        super().__init__()
        self.scale = scale

        c1, c2 = channels
        k1, k2, k3 = kernels

        self.conv1 = nn.Conv2d(3, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.conv3 = nn.Conv2d(c2, 3, kernel_size=k3, padding=k3 // 2)

        self._init_weights()

    def _init_weights(self) -> None:
        # Induláskor a kimeneti réteg kicsi -> residual ≈ 0, azaz SR ≈ bicubic.
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        x = F.relu(self.conv1(base), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        residual = self.conv3(x)
        # Tanítás közben nem clampelünk (gradient elvágás elkerülése),
        # kiértékelésnél / mentésnél a tensor_to_np clampel.
        return base + residual


class _ResBlock(nn.Module):
    """EDSR-stílusú residual block, BatchNorm nélkül (SR-en ront)."""

    def __init__(self, num_features: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(F.relu(self.conv1(x), inplace=True))
        return x + residual * self.res_scale


class _Upsampler(nn.Sequential):
    """Pixel shuffle alapú felskálázó. 2^k vagy 3× skálát támogat."""

    def __init__(self, scale: int, num_features: int) -> None:
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(num_features, num_features * 4, 3, padding=1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, 3, padding=1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Nem támogatott scale: {scale}")
        super().__init__(*layers)


class EDSR(nn.Module):
    """EDSR baseline (Lim et al., 2017): head -> B residual block -> tail."""

    def __init__(
        self,
        scale: int,
        num_features: int = config.EDSR_NUM_FEATURES,
        num_blocks: int = config.EDSR_NUM_BLOCKS,
        res_scale: float = config.EDSR_RES_SCALE,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, num_features, 3, padding=1)

        body = [_ResBlock(num_features, res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, 3, padding=1))
        self.body = nn.Sequential(*body)

        self.upsample = _Upsampler(scale, num_features)
        self.tail = nn.Conv2d(num_features, 3, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


AVAILABLE_MODELS = ("bicubic", "srcnn", "edsr")


def build_model(name: str, scale: int | None = None) -> nn.Module:
    if scale is None:
        scale = config.SCALE
    name = name.lower()
    if name == "bicubic":
        return BicubicUpsampler(scale)
    if name == "srcnn":
        return SRCNN(scale)
    if name == "edsr":
        return EDSR(scale)
    raise ValueError(f"Ismeretlen modell: {name!r}. Elérhető: {AVAILABLE_MODELS}")
