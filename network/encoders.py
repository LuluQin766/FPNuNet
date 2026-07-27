"""Paper-aligned SAM/UNI adapters and lightweight RGB encoders."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .dct import dct_high_pass


class DCTPromptGenerator(nn.Module):
    """Input-dependent high-pass prompts used by both foundation encoders."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        patch_size: int,
        scale_factor: int = 32,
        high_pass_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        bottleneck = embed_dim // scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.high_pass_fraction = high_pass_fraction
        self.embedding_generator = nn.Linear(embed_dim, bottleneck)
        self.handcrafted_generator = nn.Conv2d(
            3, bottleneck, kernel_size=patch_size, stride=patch_size
        )
        self.lightweight_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(bottleneck, bottleneck), nn.GELU()) for _ in range(depth)]
        )
        self.shared_mlp = nn.Linear(bottleneck, embed_dim)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> list[torch.Tensor]:
        high = dct_high_pass(image, self.high_pass_fraction)
        handcrafted = self.handcrafted_generator(high).flatten(2).transpose(1, 2)
        embedded = self.embedding_generator(tokens)
        if handcrafted.shape[1] != embedded.shape[1]:
            side = int(embedded.shape[1] ** 0.5)
            handcrafted = F.interpolate(
                handcrafted.transpose(1, 2).reshape(image.shape[0], -1, *self._square_hw(handcrafted)),
                size=(side, side),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        mixed = handcrafted + embedded
        return [self.shared_mlp(layer(mixed)) for layer in self.lightweight_mlps]

    @staticmethod
    def _square_hw(tokens: torch.Tensor) -> tuple[int, int]:
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("Prompt tokens must form a square spatial grid")
        return side, side


def _tokens_from_sam_patch_embed(patch_embed: nn.Module, image: torch.Tensor) -> torch.Tensor:
    x = patch_embed(image)
    if x.ndim != 4:
        raise ValueError("SAM patch_embed must return a 4-D tensor")
    # Segment Anything uses BHWC; tolerate BCHW for simple test doubles.
    if x.shape[-1] > x.shape[1]:
        return x
    return x.permute(0, 2, 3, 1).contiguous()


class SAMStructuralEncoder(nn.Module):
    """Frozen SAM ViT-B with trainable 8x8 patch projection and DCT prompts."""

    def __init__(self, image_encoder: nn.Module, input_size: int = 128) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.input_size = input_size
        self.embed_dim = int(getattr(image_encoder, "embed_dim", 768))
        self.depth = len(image_encoder.blocks)
        self.prompt_generator = DCTPromptGenerator(
            self.embed_dim, self.depth, patch_size=8
        )
        for parameter in image_encoder.parameters():
            parameter.requires_grad = False
        pos = getattr(image_encoder, "pos_embed", None)
        if pos is not None and pos.shape[1:3] != (16, 16):
            resized = F.interpolate(
                pos.detach().permute(0, 3, 1, 2),
                size=(16, 16),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            image_encoder.pos_embed = nn.Parameter(resized, requires_grad=False)
        projection = getattr(image_encoder.patch_embed, "proj", None)
        if projection is None or tuple(projection.kernel_size) != (8, 8):
            image_encoder.patch_embed.proj = nn.Conv2d(
                3, self.embed_dim, kernel_size=8, stride=8
            )
        for parameter in image_encoder.patch_embed.proj.parameters():
            parameter.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = _tokens_from_sam_patch_embed(self.image_encoder.patch_embed, image)
        b, h, w, c = x.shape
        prompts = self.prompt_generator(image, x.reshape(b, h * w, c))
        pos = getattr(self.image_encoder, "pos_embed", None)
        if pos is not None:
            pos_bchw = pos.permute(0, 3, 1, 2)
            pos_bchw = F.interpolate(pos_bchw, size=(h, w), mode="bilinear", align_corners=False)
            x = x + pos_bchw.permute(0, 2, 3, 1)
        for index, block in enumerate(self.image_encoder.blocks):
            x = block(x + prompts[index].reshape(b, h, w, c))
        return x.permute(0, 3, 1, 2).contiguous()


class UNISemanticEncoder(nn.Module):
    """Frozen UNI ViT-L with alternating DCT prompt injection."""

    def __init__(
        self,
        backbone: nn.Module,
        prompt_layers: Sequence[int] = tuple(range(0, 24, 2)),
        hook_layers: Sequence[int] = (3, 12, 21),
        detach_skips: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompt_layers = tuple(prompt_layers)
        self.hook_layers = tuple(hook_layers)
        self.detach_skips = detach_skips
        self.embed_dim = int(getattr(backbone, "embed_dim", 1024))
        self.prompt_generator = DCTPromptGenerator(
            self.embed_dim, len(self.prompt_layers), patch_size=16
        )
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = self.backbone.patch_embed(image)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        prompts = self.prompt_generator(image, tokens)
        captured: dict[int, torch.Tensor] = {}
        prompt_index = 0
        x = tokens
        for index, block in enumerate(self.backbone.blocks):
            if index in self.prompt_layers:
                x = x + prompts[prompt_index]
                prompt_index += 1
            x = block(x)
            if index in self.hook_layers:
                captured[index] = x.detach() if self.detach_skips else x
        if len(captured) != len(self.hook_layers):
            raise RuntimeError("UNI backbone did not expose all requested hook layers")
        main = self._tokens_to_map(x)
        main = F.interpolate(main, size=(16, 16), mode="bilinear", align_corners=False)
        target_shapes = ((64, 32, 32), (16, 64, 64), (4, 128, 128))
        skips = [
            self._reshape_preserving_elements(captured[layer], shape)
            for layer, shape in zip(self.hook_layers, target_shapes)
        ]
        return main, skips

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("UNI token count must be square")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    @staticmethod
    def _reshape_preserving_elements(
        tokens: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        channels, height, width = shape
        expected = channels * height * width
        if tokens.shape[1] * tokens.shape[2] != expected:
            raise ValueError(
                f"UNI tokens contain {tokens.shape[1] * tokens.shape[2]} elements; "
                f"cannot reshape to {shape}"
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], channels, height, width)


class WaveletFeatureEncoder(nn.Module):
    """Fixed Haar decomposition followed by a lightweight learned projection."""

    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1, 1], [1, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [-1, 1]],
                [[1, -1], [-1, 1]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("haar", filters[:, None], persistent=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(12, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        weight = self.haar.repeat(image.shape[1], 1, 1, 1)
        bands = F.conv2d(image, weight, stride=2, groups=image.shape[1])
        bands = F.interpolate(bands, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(bands)


class MultiScaleContextEncoder(nn.Module):
    """Three spatial scales fused into one full-resolution descriptor."""

    def __init__(self, out_channels: int = 8, branch_channels: int = 64) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, branch_channels, 3, padding=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = []
        for scale, branch in zip((1.0, 0.5, 0.25), self.branches):
            x = image if scale == 1.0 else F.interpolate(image, scale_factor=scale, mode="bilinear")
            x = branch(x)
            outputs.append(F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False))
        return self.projection(self.fuse(torch.cat(outputs, dim=1)))
