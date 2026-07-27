from __future__ import annotations

import numpy as np
import torch
from torch import nn

from main import FPNuNetLoss, build_optimizer
from main.postprocess import logits_to_instances
from network.dct import dct_2d, dct_high_pass, idct_2d
from network.encoders import MultiScaleContextEncoder, WaveletFeatureEncoder
from network.fpnunet import FPNuNet
from network.modules import PFAEBlock


class MockSAMPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, 8, stride=8)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class MockSAMEncoder(nn.Module):
    embed_dim = 768

    def __init__(self):
        super().__init__()
        self.patch_embed = MockSAMPatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 16, 768))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])


class MockUNIPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 1024, 16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MockUNIEncoder(nn.Module):
    embed_dim = 1024

    def __init__(self):
        super().__init__()
        self.patch_embed = MockUNIPatchEmbed()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(24)])


def test_dct_round_trip_and_high_pass():
    x = torch.randn(2, 3, 16, 16)
    reconstructed = idct_2d(dct_2d(x))
    torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)
    high = dct_high_pass(x)
    assert high.shape == x.shape
    assert torch.isfinite(high).all()


def test_lightweight_encoders_and_pfae():
    x = torch.randn(1, 3, 128, 128)
    wfe = WaveletFeatureEncoder()
    msce = MultiScaleContextEncoder()
    assert wfe(x).shape == (1, 8, 128, 128)
    assert msce(x).shape == (1, 8, 128, 128)
    assert sum(parameter.numel() for parameter in wfe.parameters()) == 952
    assert sum(parameter.numel() for parameter in msce.parameters()) == 7_384
    feature = torch.randn(1, 16, 16, 16)
    assert PFAEBlock(16, stages=2, heads=1)(feature).shape == feature.shape


def test_model_contract_loss_backward_and_freezing():
    model = FPNuNet(MockSAMEncoder(), MockUNIEncoder())
    image = torch.randn(1, 3, 128, 128)
    output = model(image)
    assert set(output) == {"bin", "boundary", "hv", "tp", "type_aux"}
    assert output["bin"].shape == (1, 1, 128, 128)
    assert output["boundary"].shape == (1, 1, 128, 128)
    assert output["hv"].shape == (1, 2, 128, 128)
    assert output["tp"].shape == (1, 5, 128, 128)
    assert len(output["type_aux"]) == 2

    target = {
        "bin": torch.randint(0, 2, (1, 1, 128, 128)).float(),
        "boundary": torch.randint(0, 2, (1, 1, 128, 128)).float(),
        "hv": torch.randn(1, 2, 128, 128),
        "tp": torch.randint(0, 5, (1, 128, 128)),
    }
    losses = FPNuNetLoss()(output, target)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert not any(p.grad is not None for p in model.sam_encoder.image_encoder.blocks.parameters())
    assert not any(p.grad is not None for p in model.uni_encoder.backbone.blocks.parameters())
    assert any(p.grad is not None for p in model.sam_encoder.prompt_generator.parameters())
    assert any(p.grad is not None for p in model.fusion_neck.parameters())
    assert any(p.grad is not None for p in model.type_decoder.parameters())
    optimizer, scheduler = build_optimizer(model)
    assert [group["name"] for group in optimizer.param_groups] == [
        "prompt_generators",
        "trainable_components",
    ]
    assert [group["initial_lr"] for group in optimizer.param_groups] == [1e-3, 5e-4]
    assert scheduler is not None


def test_postprocess_smoke():
    binary = torch.full((1, 128, 128), -8.0)
    binary[:, 30:90, 30:90] = 8.0
    hv = torch.zeros(2, 128, 128)
    nuclear_type = torch.zeros(5, 128, 128)
    nuclear_type[2, 30:90, 30:90] = 4.0
    instances, types = logits_to_instances(binary, hv, nuclear_type)
    assert instances.shape == (128, 128)
    assert types.shape == (128, 128)
    assert np.isfinite(instances).all()
