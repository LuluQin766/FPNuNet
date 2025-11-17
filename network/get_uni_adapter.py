# network/get_network.py
# from copy import deepcopy
# from functools import partial
# from segment_anything.modeling.common import LayerNorm2d

from math import e
import sys
sys.path.append('/root/SAM2PATH-main')

import os
import torch
from torch import nn
# sim added for model encoder tes
# from torchsummary import summary
from typing import Optional, Tuple, Type
from segment_anything_local.modeling.image_encoder_SAMAdapter import PromptGenerator
# from segment_anything_local.modeling.image_encoder_SAMAdapter_debug import Block, PromptGenerator   # with debug info

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False
# DEBUG = True    # print debug info

def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n ✅ Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

class PromptInjectedUNIWrapper(nn.Module):
    def __init__(
            self, 
            model, 
            ft_dim: int = 1024, 
            out_dim: int = 256, 
            prompt_layers=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # 之前的设置
            neck=True, 
            re_norm=False, 
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5),
            img_size: int = 1024,
            patch_size: int = 16,
            embed_dim: int = 1024,
            depth: int = 12,
            hook_layers = [3, 6, 9]
        ) -> None:
        super().__init__()
        self.model = model  # ViT-L backbone
        self.ft_dim = ft_dim
        self.out_dim = out_dim
        self.prompt_layers = prompt_layers
        self.neck = neck
        self.re_norm = re_norm

        # === 冻结 model 主干参数 ===
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        print("🧊 UNI backbone frozen. Only PromptGenerator is trainable.")

        self.register_buffer("in_mean", torch.Tensor((0.485, 0.456, 0.406)).view(-1, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.Tensor((0.229, 0.224, 0.225)).view(-1, 1, 1), persistent=False)
        self.register_buffer("mean", torch.Tensor(mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("std", torch.Tensor(std).view(-1, 1, 1), persistent=False)

        assert hasattr(self.model, 'blocks'), "The provided model must have a 'blocks' attribute (e.g., ViT)"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        if neck:
            self.neck_layer = nn.Sequential(
                nn.Conv2d(ft_dim, out_dim, kernel_size=1, bias=False),
                nn.LayerNorm([out_dim, 64, 64]),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([out_dim, 64, 64]),
            )
        else:
            self.neck_layer = None
        
        self.scale_factor = 32  # orginal image size is 1024, we use 32x32 patches
        self.prompt_type = 'highpass'
        self.tuning_stage = 1234
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        # === Prompt Generator ===
        self.prompt_generator = PromptGenerator(
            scale_factor=self.scale_factor,
            prompt_type=self.prompt_type,
            embed_dim=self.embed_dim,
            tuning_stage=self.tuning_stage,
            depth=self.depth,
            input_type=self.input_type,
            freq_nums=self.freq_nums,
            handcrafted_tune=self.handcrafted_tune,
            embedding_tune=self.embedding_tune,
            adaptor=self.adaptor,
            img_size=self.img_size,
            patch_size=self.patch_size
        )
        # === 只训练 PromptGenerator 参数 ===
        for name, param in self.prompt_generator.named_parameters():
            param.requires_grad = True
            if DEBUG:
                print(f"✅ Prompt param trainable: {name}")

        # 注册中间层输出 hook
        # self.hook_layers = [3, 6, 9]  # 之前的设置
        self.hook_layers = hook_layers
        self.inter_features = {}
        for i in self.hook_layers:
            self.model.blocks[i].register_forward_hook(self._get_hook(i))
        
        if DEBUG:
            self.print_args()

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.inter_features[layer_idx] = output
        return hook

    def print_args(self):
        print(f"\n ======== get_uni_adapter.py PromptInjectedUNIWrapper.print_args =========")
        print(f" Args: self.ft_dim={self.ft_dim}, self.out_dim={self.out_dim}, self.prompt_layers={self.prompt_layers}, self.neck={self.neck}, self.re_norm={self.re_norm}")
        print(f" Args: self.mean={self.mean}, self.std={self.std}")
        print(f" Args: self.img_size={self.img_size}, self.patch_size={self.patch_size}, self.embed_dim={self.embed_dim}, self.depth={self.depth}")
        print(f" Args: self.scale_factor={self.scale_factor}, self.prompt_type={self.prompt_type},")
        print(f" Args: self.tuning_stage={self.tuning_stage}, self.input_type={self.input_type}, self.freq_nums={self.freq_nums}, self.handcrafted_tune={self.handcrafted_tune}, self.embedding_tune={self.embedding_tune}, self.adaptor={self.adaptor}")
        print("\n")

    def forward(self, x: torch.Tensor):
        if DEBUG:
            print(f"\n ========= get_uni_adapter.py PromptInjectedUNIWrapper.forward =========")
            self.print_args()
            print(f" input, x.shape: {x.shape}, x.dtype: {x.dtype}, x.device: {x.device}, value range: [{x.min().item()}, {x.max().item()}]")
        
        self.inter_features = {}  # 重置 hook cache
        B = x.shape[0]
        x_raw = x.clone()

        # === (1) Re-normalization ===
        if self.re_norm:
            x = x * self.in_std + self.in_mean
            x = (x - self.mean) / self.std
            if DEBUG:
                print(f"After re-norm, shape: {x.shape}, value range: [{x.min().item()}, {x.max().item()}]")

        # === (2) Patch embedding & Prompt generation ===
        patch_embed = self.model.patch_embed(x)  # [B, N, C]
        if DEBUG:
            print(f" patch_embed.shape: {patch_embed.shape}")

        embed_feat = self.prompt_generator.init_embeddings(patch_embed, use_permute=False)
        if DEBUG:
            print(f" embed_feat.shape: {embed_feat.shape}")

        handcrafted_feat = self.prompt_generator.init_handcrafted(x_raw)
        if DEBUG:
            print(f" handcrafted_feat.shape: {handcrafted_feat.shape}")

        prompts = self.prompt_generator.get_prompt(handcrafted_feat, embed_feat)  # list of [B, N, C]
        if DEBUG:
            if isinstance(prompts, list):
                print(f" ----- prompts len: {len(prompts)}")
                for i, p in enumerate(prompts):
                    print(f" Prompt {i} shape: {p.shape}")
                print("\n")
            else:
                print(f" prompts shape: {prompts.shape}")

        # === (3) Transformer Blocks with Prompt Injection ===
        x = patch_embed  # [B, N, C]
        if DEBUG:
            print(f"\n ------ Transformer blocks with prompt injection ------")
            print(f" Initial x.shape: {x.shape}")
        for i, blk in enumerate(self.model.blocks):
            if DEBUG:
                print(f" ---- Block {i}:")
            if i in self.prompt_layers:
                prompt_idx = self.prompt_layers.index(i)
                x = x + prompts[prompt_idx]   # Prompt 注入
            x = blk(x)
            if DEBUG:
                print(f" Layer {i} feature shape: {x.shape}")
            if i in self.hook_layers:
                self.inter_features[i] = x.detach()

        # === (4) Neck transformation ===
        if self.neck_layer is not None:
            x = x.permute(0, 2, 1).reshape(B, -1, 64, 64)
            if DEBUG:
                print(f" x.shape after permute: {x.shape}")
            x_out = self.neck_layer(x)
        else:
            x_out = x

        if DEBUG:
            print(f" Output x_out.shape: {x_out.shape}")

        # === (5) Extract skip features for decoder ===
        if DEBUG:
            print(f"\n ------ Extracting skip features from layers: {self.hook_layers}")

        skip_feats = []
        for i in self.hook_layers:
            f = self.inter_features[i]  # [B, N, C]
            N, C = f.shape[1], f.shape[2]
            feat_size = int(N ** 0.5)

            if DEBUG:
                print(f"\n---- inter_features[{i}]: shape = {f.shape}, feat_size = ({feat_size}, {feat_size})")

            if feat_size * feat_size != N:
                print(f"⚠️ Warning: Cannot reshape layer {i} feature to 2D (non-square patch count: N={N})")
                feat_2d = None
            else:
                feat_2d = f.permute(0, 2, 1).reshape(B, C, feat_size, feat_size)

            skip_feats.append(feat_2d)
            if DEBUG:
                print(f"Skip {i} → tokens: {f.shape}, feature_map: {feat_2d.shape if feat_2d is not None else None}")

            # skip_feats.append({
            #     "tokens": f,       # [B, N, C]
            #     "feature_map": feat_2d  # [B, C, H, W] or None
            # })
            # if DEBUG:
            #     print(f"Skip {i} → tokens: {f.shape}, feature_map: {feat_2d.shape if feat_2d is not None else None}")

        return x_out, skip_feats

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
def get_uni_adapter(pretrained=None, neck=True):
    from .hipt.vision_transformer import vit_large
    from .hipt.hipt_prompt import load_uni_weights
    model = vit_large()
    model = load_uni_weights(model, pretrained)

    model = PromptInjectedUNIWrapper(
        model=model,
        ft_dim=1024,
        out_dim=256,
        neck=neck,
        re_norm=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        hook_layers=[3, 12, 21],  # Mv21之后的模型改为这三次，之前的模型改为[3, 6, 9]
    )
    return model

if __name__ == '__main__':
    # test uni
    _, pretrained = has_uni()
    neck = False
    print(f"\n ---- Pretrained: {pretrained}, neck: {neck}")
    encoder = get_uni_adapter(pretrained, neck=neck)

    print_trainable_params(encoder)

    print("\n\n ----------- Model architecture:")
    print(encoder)
    print(" ----------- end of Model architecture ---------- \n\n")

    input_feature = torch.randn(size=(2, 3, 128, 128))

    print(f" ---- Input feature shape: {input_feature.shape}")

    out, skips = encoder(input_feature)
    print('\n ---- Output feature shape:', out.shape)
    for i, s in enumerate(skips):
        print(f"  Skip {i} shape: {s.shape}")

    # print(f" ---- Skip {len(skips)}")
    # for i, s in enumerate(skips):
    #     print(f"Skip {i} shape: {s['tokens'].shape}, feature_map shape: {s['feature_map'].shape if s['feature_map'] is not None else None}")
    print(f"\n")
