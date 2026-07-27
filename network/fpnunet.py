"""The single, paper-aligned public FPNuNet model."""

from __future__ import annotations

import torch
from torch import nn

from .encoders import (
    MultiScaleContextEncoder,
    SAMStructuralEncoder,
    UNISemanticEncoder,
    WaveletFeatureEncoder,
)
from .modules import (
    BinaryDecoder,
    HVDecoderV4,
    PFAEGlobalFusionNeck,
    PFAESkipEnhancement,
    TypeDecoderV4,
)


class FPNuNet(nn.Module):
    """Frequency-aware prompt-guided network for IHC nuclear analysis."""

    def __init__(self, sam_image_encoder: nn.Module, uni_backbone: nn.Module) -> None:
        super().__init__()
        self.sam_encoder = SAMStructuralEncoder(sam_image_encoder)
        self.uni_encoder = UNISemanticEncoder(uni_backbone)
        self.wfe = WaveletFeatureEncoder(8)
        self.msce = MultiScaleContextEncoder(8)
        self.fusion_neck = PFAEGlobalFusionNeck()
        self.skip_neck_64 = PFAESkipEnhancement(16)
        self.skip_neck_128 = PFAESkipEnhancement(4)
        self.binary_decoder = BinaryDecoder()
        self.hv_decoder = HVDecoderV4()
        self.type_decoder = TypeDecoderV4(num_classes=5)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, received {tuple(image.shape)}")
        if image.shape[-2:] != (128, 128):
            raise ValueError("FPNuNet expects 128x128 patches")
        sam = self.sam_encoder(image)
        uni, skips = self.uni_encoder(image)
        skips[1] = self.skip_neck_64(skips[1])
        skips[2] = self.skip_neck_128(skips[2])
        wfe = self.wfe(image)
        msce = self.msce(image)
        fused = self.fusion_neck(sam, uni, wfe, msce)
        binary, boundary, binary_features = self.binary_decoder(fused, skips)
        hv = self.hv_decoder(fused, skips, binary_features)
        nuclear_type, type_aux = self.type_decoder(fused, skips, binary_features)
        return {
            "bin": binary,
            "boundary": boundary,
            "hv": hv,
            "tp": nuclear_type,
            "type_aux": type_aux,
        }

    def parameter_summary(self) -> dict[str, dict[str, int]]:
        modules = {
            "sam_encoder": self.sam_encoder,
            "uni_encoder": self.uni_encoder,
            "wfe": self.wfe,
            "msce": self.msce,
            "fusion_neck": self.fusion_neck,
            "skip_necks": nn.ModuleList([self.skip_neck_64, self.skip_neck_128]),
            "binary_decoder": self.binary_decoder,
            "hv_decoder": self.hv_decoder,
            "type_decoder": self.type_decoder,
        }
        return {
            name: {
                "total": sum(p.numel() for p in module.parameters()),
                "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            for name, module in modules.items()
        }
