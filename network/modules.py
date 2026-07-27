"""PFAE fusion and collaborative decoders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dct import dct_2d, idct_2d


class ChannelSpatialGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 16)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * self.spatial(spatial)


class DCTFrequencyAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.gate = ChannelSpatialGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = dct_2d(x.float()).to(x.dtype)
        attention_frequency = frequency
        if frequency.shape[-2] * frequency.shape[-1] > 256:
            attention_frequency = F.adaptive_avg_pool2d(frequency, (16, 16))
        tokens = attention_frequency.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape_as(attention_frequency)
        if attended.shape[-2:] != frequency.shape[-2:]:
            attended = F.interpolate(
                attended, size=frequency.shape[-2:], mode="bilinear", align_corners=False
            )
        return idct_2d(self.gate(attended).float()).to(x.dtype)


class DCTLightweightAttention(nn.Module):
    """Parameter-light DCT gating used outside the single heavy stage."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = ChannelSpatialGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = self.gate(dct_2d(x.float()).to(x.dtype))
        return idct_2d(frequency.float()).to(x.dtype)


class PFAEBlock(nn.Module):
    """Progressive dilated refinement with DCT attention and residual fusion."""

    def __init__(self, channels: int, stages: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for index, dilation in enumerate((3, 5, 7)[:stages]):
            self.stages.append(
                nn.ModuleDict(
                    {
                        "spatial": nn.Sequential(
                            nn.Conv2d(
                                channels,
                                channels,
                                3,
                                padding=dilation,
                                dilation=dilation,
                                groups=1 if index == 0 else 2 if channels % 2 == 0 else 1,
                            ),
                            nn.BatchNorm2d(channels),
                            nn.GELU(),
                        ),
                        "frequency": DCTFrequencyAttention(channels, heads)
                        if index == 0
                        else DCTLightweightAttention(channels),
                    }
                )
            )
        self.aggregate = nn.Conv2d(channels * stages, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        outputs = []
        for stage in self.stages:
            x = stage["spatial"](x)
            x = x + stage["frequency"](x)
            outputs.append(x)
        return residual + self.aggregate(torch.cat(outputs, dim=1))


class PFAEGlobalFusionNeck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_proj = nn.Conv2d(768, 128, 1)
        self.uni_proj = nn.Conv2d(1024, 128, 1)
        self.wfe_proj = nn.Conv2d(8, 16, 1)
        self.msce_proj = nn.Conv2d(8, 16, 1)
        self.fusion_proj = nn.Conv2d(288, 256, 1)
        self.pfae = PFAEBlock(256, stages=3, heads=4)
        self.high_gate = nn.Sequential(nn.Conv2d(16, 256, 3, stride=8, padding=1), nn.Sigmoid())
        self.high_value = nn.Conv2d(16, 256, 3, stride=8, padding=1)
        self.position = nn.Sequential(
            nn.Conv2d(258, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU()
        )

    def forward(
        self,
        sam: torch.Tensor,
        uni: torch.Tensor,
        wfe: torch.Tensor,
        msce: torch.Tensor,
    ) -> torch.Tensor:
        size = sam.shape[-2:]
        local = torch.cat((wfe, msce), dim=1)
        fused = torch.cat(
            (
                self.sam_proj(sam),
                self.uni_proj(F.interpolate(uni, size=size, mode="bilinear", align_corners=False)),
                F.adaptive_avg_pool2d(self.wfe_proj(wfe), size),
                F.adaptive_avg_pool2d(self.msce_proj(msce), size),
            ),
            dim=1,
        )
        base = self.fusion_proj(fused)
        detail = self.high_gate(local) * self.high_value(local)
        if detail.shape[-2:] != size:
            detail = F.interpolate(detail, size=size, mode="bilinear", align_corners=False)
        y = base + self.pfae(base) + detail
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size[0], device=y.device, dtype=y.dtype),
            torch.linspace(-1, 1, size[1], device=y.device, dtype=y.dtype),
            indexing="ij",
        )
        coords = torch.stack((xx, yy)).expand(y.shape[0], -1, -1, -1)
        return self.position(torch.cat((y, coords), dim=1))


class PFAESkipEnhancement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = PFAEBlock(channels, stages=2, heads=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BinaryDecodeStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = PixelShuffleUp(in_channels, out_channels)
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.skip_projection(skip)
        return self.refine(torch.cat((x, skip), dim=1))


class BinaryDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                BinaryDecodeStage(256, 64, 128),
                BinaryDecodeStage(128, 16, 64),
                BinaryDecodeStage(64, 4, 32),
            ]
        )
        self.binary_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, 32, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1),
                )
                for channels in (128, 64, 32)
            ]
        )
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]):
        features = []
        predictions = []
        for stage, head, skip in zip(self.stages, self.binary_heads, skips):
            x = stage(x, skip)
            features.append(x)
            predictions.append(head(x))
        return predictions[-1], self.boundary_head(features[-1]), features


class BilinearUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BinaryGuidedSkipFusion(nn.Module):
    """Cross-attend to a UNI skip and modulate it with binary features."""

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        heads: int = 4,
        max_attention_size: int = 32,
    ) -> None:
        super().__init__()
        self.max_attention_size = max_attention_size
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.binary_gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, feature: torch.Tensor, skip: torch.Tensor, binary_feature: torch.Tensor
    ) -> torch.Tensor:
        skip = self.skip_projection(skip)
        height, width = feature.shape[-2:]
        attention_size = (
            min(height, self.max_attention_size),
            min(width, self.max_attention_size),
        )
        query_map = F.adaptive_avg_pool2d(feature, attention_size)
        skip_map = F.adaptive_avg_pool2d(skip, attention_size)
        query = query_map.flatten(2).transpose(1, 2)
        key_value = skip_map.flatten(2).transpose(1, 2)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        attended = attended.transpose(1, 2).reshape_as(query_map)
        attended = F.interpolate(
            attended, size=(height, width), mode="bilinear", align_corners=False
        )
        gate = self.binary_gate(binary_feature)
        return self.fuse(feature + feature * (1.0 - gate) + attended * gate)


class DenseRefinement(nn.Module):
    """Two-unit dense refinement used by the paper HV decoder."""

    def __init__(self, channels: int, growth_rate: int = 16) -> None:
        super().__init__()
        self.units = nn.ModuleList()
        current = channels
        for _ in range(2):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(current),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(current, growth_rate, 1, bias=False),
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(growth_rate, growth_rate, 3, padding=1, bias=False),
                )
            )
            current += growth_rate
        self.out_channels = current
        self.output = nn.Sequential(nn.BatchNorm2d(current), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            x = torch.cat((x, unit(x)), dim=1)
        return self.output(x)


class HVDecoderV4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (128, 64, 32)
        self.upsamples = nn.ModuleList(
            [BilinearUp(256, 128), BilinearUp(128, 64), BilinearUp(64, 32)]
        )
        self.fusions = nn.ModuleList(
            [
                BinaryGuidedSkipFusion(128, 64),
                BinaryGuidedSkipFusion(64, 16),
                BinaryGuidedSkipFusion(32, 4),
            ]
        )
        self.dense_blocks = nn.ModuleList([DenseRefinement(c) for c in channels])
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(block.out_channels, channels[index], 1, bias=False),
                    nn.BatchNorm2d(channels[index]),
                    nn.ReLU(inplace=True),
                )
                for index, block in enumerate(self.dense_blocks)
            ]
        )
        self.head_features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(block.out_channels, 32, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                for block in self.dense_blocks
            ]
        )
        self.heads = nn.ModuleList([nn.Conv2d(32, 2, 1) for _ in channels])

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], binary_features: list[torch.Tensor]):
        predictions = []
        for up, fusion, dense, projection, head_feature, head, skip, binary in zip(
            self.upsamples,
            self.fusions,
            self.dense_blocks,
            self.projections,
            self.head_features,
            self.heads,
            skips,
            binary_features,
        ):
            x = fusion(up(x), skip, binary)
            dense_feature = dense(x)
            predictions.append(head(head_feature(dense_feature)))
            x = projection(dense_feature)
        return predictions[-1]


class PromptTokenGenerator(nn.Module):
    """Learnable prompt tokens with lightweight inter-prompt reasoning."""

    def __init__(self, channels: int, num_prompts: int = 6, heads: int = 4) -> None:
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_prompts, channels) * 0.02)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(prompts, prompts, prompts, need_weights=False)
        prompts = prompts + attended
        return prompts + self.ffn(self.norm(prompts))


class PromptToFeatureAttention(nn.Module):
    """Type prompts are Query; pooled spatial features are Key and Value."""

    def __init__(self, channels: int, heads: int = 4, pool_kernel: int = 2) -> None:
        super().__init__()
        self.pool_kernel = pool_kernel
        self.query_projection = nn.Linear(channels, channels)
        self.key_projection = nn.Linear(channels, channels)
        self.value_projection = nn.Linear(channels, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm = nn.LayerNorm(channels)
        self.output_projection = nn.Linear(channels, channels)

    def forward(
        self, feature: torch.Tensor, prompt_tokens: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, _, _ = feature.shape
        pooled = F.avg_pool2d(feature, kernel_size=self.pool_kernel)
        spatial_tokens = pooled.flatten(2).transpose(1, 2)
        query = self.query_projection(prompt_tokens)
        key = self.key_projection(spatial_tokens)
        value = self.value_projection(spatial_tokens)
        attended, _ = self.attention(query, key, value, need_weights=False)
        attended = attended + self.ffn(self.norm(attended))
        global_prompt = self.output_projection(attended).mean(dim=1)
        global_prompt = global_prompt.reshape(batch, channels, 1, 1)
        return feature + global_prompt


class TypeDecoderV4(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.upsamples = nn.ModuleList(
            [PixelShuffleUp(256, 128), PixelShuffleUp(128, 64), PixelShuffleUp(64, 32)]
        )
        self.prompt_generators = nn.ModuleList(
            [PromptTokenGenerator(128), PromptTokenGenerator(64), PromptTokenGenerator(32)]
        )
        self.prompt_attention = nn.ModuleList(
            [PromptToFeatureAttention(128), PromptToFeatureAttention(64), PromptToFeatureAttention(32)]
        )
        self.fusions = nn.ModuleList(
            [
                BinaryGuidedSkipFusion(128, 64),
                BinaryGuidedSkipFusion(64, 16),
                BinaryGuidedSkipFusion(32, 4),
            ]
        )
        self.head_features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, 32, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                for channels in (128, 64, 32)
            ]
        )
        self.heads = nn.ModuleList(
            [nn.Conv2d(32, num_classes, 1) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], binary_features: list[torch.Tensor]):
        predictions = []
        for up, prompt_generator, prompt, fusion, head_feature, head, skip, binary in zip(
            self.upsamples,
            self.prompt_generators,
            self.prompt_attention,
            self.fusions,
            self.head_features,
            self.heads,
            skips,
            binary_features,
        ):
            x = up(x)
            x = prompt(x, prompt_generator(x.shape[0]))
            x = fusion(x, skip, binary)
            predictions.append(head(head_feature(x)))
        return predictions[-1], predictions[:-1]
