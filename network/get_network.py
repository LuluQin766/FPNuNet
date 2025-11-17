# network/get_network.py
# from copy import deepcopy
# from functools import partial
from segment_anything.modeling.common import LayerNorm2d

import os
import torch
from torch import nn
# sim added for model encoder test
# from torchsummary import summary

class EncoderWrapper(nn.Module):
    def __init__(self, model, ft_dim, out_dim, neck=True, re_norm = False, mean=None, std=None):
        super().__init__()
        self.model = model
        self.neck = neck
        self.re_norm = re_norm

        if neck:
            self.neck_layer = nn.Sequential(
                nn.Conv2d(ft_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
            )
        else:
            self.neck_layer = None

        self.register_buffer("in_mean", torch.Tensor((0.485, 0.456, 0.406)).view(-1, 1, 1), False)
        self.register_buffer("in_std", torch.Tensor((0.229, 0.224, 0.225)).view(-1, 1, 1), False)
        self.register_buffer("mean", torch.Tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.Tensor(std).view(-1, 1, 1), False)

        # Hook layers for skip features
        self.hook_layers = [3, 6, 9]
        self.inter_features = {}
        for i in self.hook_layers:
            self.model.blocks[i].register_forward_hook(self._get_hook(i))

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.inter_features[layer_idx] = output
        return hook

    def forward(self, x, no_grad=True): #  x [B, 3, 1024, 1024]
        self.inter_features = {}  # reset hooks

        if self.re_norm:
            x = x * self.in_std + self.in_mean
            x = (x - self.mean) / self.std
        
        # print(f"Input x.shape: {x.shape}")

        if no_grad:
            with torch.no_grad():
                x = self.model(x, dense=True)
        else:
            x = self.model(x, dense=True)
        
        # print(f" self.model output: x.shape: {x.shape}")

        # x: [B, 4096, C], project to [B, C, 64, 64] if neck
        if self.neck:
            x = x.permute(0, 2, 1).reshape(x.shape[0], -1, 64, 64)
            x_out = self.neck_layer(x)
        else:
            x_out = x

        # print(f"Encoder x_out shape: {x_out.shape}")

        # Generate intermediate skip features
        skip_feats = []
        for i in self.hook_layers:
            # print("--------- Available inter_features keys:", self.inter_features.keys())
            # print("Trying to access key:", i)
            f = self.inter_features[i]  # [B, 4096, C]
            # print(f"\n ---- inter_features[{i}] feature shape: {f.shape}")
            f = f[:, 1:, :]  # remove CLS token
            feat_size = int((f.shape[1]) ** 0.5)
            # print(f" Layer {i} feature shape after permute: {f.shape}, feat_size: {feat_size}")
            f = f.permute(0, 2, 1).reshape(x.shape[0], -1, feat_size, feat_size)
            # print(f" Layer {i} feature shape after reshape: {f.shape}")
            skip_feats.append(f)

        return x_out, skip_feats


def get_hipt(pretrained=None, neck=True):
    # from .hipt.vision_transformer import vit_small
    # from .hipt.hipt_prompt import load_ssl_weights
    
    # debug use
    from hipt.vision_transformer import vit_small
    from hipt.hipt_prompt import load_ssl_weights

    model = vit_small(patch_size=16)
    model = load_ssl_weights(model, pretrained)

    model = EncoderWrapper(model, 384, 256, neck=neck,
                           re_norm=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    return model

def has_uni():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

# === Uni encoder ===
def get_uni(pretrained=None, neck=True):
    from .hipt.vision_transformer import vit_large
    from .hipt.hipt_prompt import load_uni_weights
    model = vit_large()
    model = load_uni_weights(model, pretrained)
    
    model = EncoderWrapper(model, 1024, 256, neck=neck,
                           re_norm=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    return model

if __name__ == '__main__':
    # x should be B, 4096, dim
    # pretrained = "/mnt/zmy/code/sam-path/pretrained/vit256_small_dino.pth"
    # encoder = get_hipt(pretrained, neck=True)
    # # x [6, 3, 1024, 1024]
    # input_size = (6, 3, 1024, 1024)
    
    # input_feature = torch.randn(input_size)
    # out = encoder(input_feature)
    # print(out.shape)  # [6, 4096, 384] neck=False   # [6, 256, 64, 64] neck=True
    
    # test uni
    _, pretrained = has_uni()
    neck = False
    print(f"\n ---- Pretrained: {pretrained}, neck: {neck}")
    encoder = get_uni(pretrained, neck=neck)

    input_size = (2, 3, 128, 128)
    input_feature = torch.randn(input_size)

    print(f" ---- Input feature shape: {input_feature.shape}")

    out, skips = encoder(input_feature)
    print('\n ---- Output feature shape:', out.shape)
    for i, s in enumerate(skips):
        print(f"Skip {i} shape: {s.shape}")
    print(f"Skip {len(skips)} shape: {out.shape}\n")
