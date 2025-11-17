# from re import T
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Type

# import sys
# sys.path.append('/root/SAM2PATH-main')
# from segment_anything_local import sam_model_registry, SamPredictor

DEBUG = False
# DEBUG = True # print the input and output shape of each layer

# 获取当前文件名
import os
file_name = os.path.basename(__file__)
if "debug" in file_name:
    DEBUG = True

if DEBUG:
    print(f"\n ------ Debug mode is {DEBUG}, setting on {file_name} ------ \n")

def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n ✅ Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

def get_grid_from_patch_size(input_HW, patch_size):
    """ Given input size (H, W) and patch size (patch_h, patch_w), return actual token grid."""
    H, W = input_HW
    patch_h, patch_w = patch_size
    gh = H // patch_h
    gw = W // patch_w
    return (gh, gw)

# ------------------------------
# Basic Attention Building Blocks
# ------------------------------

class AdaptiveTransformerFusionNeck(nn.Module):
    """
    AdaptiveTransformerFusionNeck fuses two input feature maps (e.g., from the SAM encoder and an extra encoder)
    using a transformer decoder block and an adaptive gating mechanism.
    
    - The transformer decoder treats the SAM features (x1) as the target sequence and the extra encoder features (x2) as memory.
    - A gating branch computes a per-location weight from concatenated features to adaptively fuse the attended
      output and the original SAM features.
    - Finally, the fused features are projected to the desired fusion channel dimension and normalized.
    """
    def __init__(self, in_channels, fusion_channels, num_heads=4, dropout=0.1, num_layers=2):
        super(AdaptiveTransformerFusionNeck, self).__init__()
        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        
        # Define a transformer decoder block.
        decoder_layer = nn.TransformerDecoderLayer(d_model=in_channels, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Gating branch: generates gating weights from concatenated features.
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()
        
        # Projection layer to adjust channel dimension to fusion_channels.
        self.proj_conv = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
        # Use LayerNorm after flattening the spatial dimensions.
        self.ln = nn.LayerNorm(fusion_channels)
        
    def forward(self, x1, x2):
        # x1 and x2: [B, in_channels, H, W]
        B, C, H, W = x1.size()
        
        # Flatten spatial dimensions for transformer input: shape -> [H*W, B, C]
        x1_flat = x1.view(B, C, H * W).permute(2, 0, 1)
        x2_flat = x2.view(B, C, H * W).permute(2, 0, 1)
        
        # Transformer decoder: use x1 as target, x2 as memory.
        attended = self.transformer_decoder(tgt=x1_flat, memory=x2_flat)  # shape: [H*W, B, C]
        attended = attended.permute(1, 2, 0).view(B, C, H, W)
        
        # Compute gating weights from the concatenation of x1 and x2.
        concat_features = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        gate = self.gate_sigmoid(self.gate_conv(concat_features))  # [B, C, H, W], values in [0, 1]
        
        # Fuse: weighted sum of attended output and original x1.
        fused = gate * attended + (1 - gate) * x1
        
        # Project to fusion_channels.
        fused_proj = self.proj_conv(fused)  # [B, fusion_channels, H, W]
        
        # Flatten for layer normalization.
        fused_flat = fused_proj.flatten(2).transpose(1, 2)  # [B, H*W, fusion_channels]
        fused_norm = self.ln(fused_flat)
        fused_norm = fused_norm.transpose(1, 2).view(B, self.fusion_channels, H, W)
        
        return fused_norm


class BinarySkipFusionBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.skip_dim = skip_dim
        self.out_dim = out_dim

        self.conv_skip = nn.Sequential(
            nn.Conv2d(skip_dim, out_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv2d((in_dim + out_dim), out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fusion_emb, skip_feat):
        if DEBUG:
            print(f"\n - BinarySkipFusionBlock: forward()")
            print(f' - self.in_dim={self.in_dim}, self.skip_dim={self.skip_dim}, self.out_dim={self.out_dim}')
            print(f" - BinarySkipFusionBlock: input fusion_emb.shape={fusion_emb.shape}")
            print(f" - BinarySkipFusionBlock: input skip_feat.shape={skip_feat.shape}")

        skip_feat = self.conv_skip(skip_feat)
        if DEBUG:
                print(f" - BinarySkipFusionBlock conv_skip: skip_feat.shape={skip_feat.shape}")

        x = torch.cat([fusion_emb, skip_feat], dim=1)
        if DEBUG:
            print(f" - BinarySkipFusionBlock concat: x.shape={x.shape}")
        x = self.conv_fusion(x)
        if DEBUG:
            print(f" - BinarySkipFusionBlock conv_fusion: x.shape={x.shape}")
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim * 2, dim, 1)
        self.v_proj = nn.Conv2d(dim * 2, dim, 1)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, fusion_emb, skip_feat, bin_feat):
        B, C, H, W = fusion_emb.shape
        q = self.q_proj(fusion_emb).flatten(2).transpose(1, 2)  # B, HW, C
        kv = torch.cat([skip_feat, bin_feat], dim=1)
        k = self.k_proj(kv).flatten(2).transpose(1, 2)  # B, HW, C
        v = self.v_proj(kv).flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return self.out_proj(attn_out)

class UnifiedFusionAttentionBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, use_bin_feat=True):
        super().__init__()
        self.in_dim = in_dim
        self.skip_dim = skip_dim
        self.out_dim = out_dim
        self.use_bin_feat = use_bin_feat

        self.conv_skip = nn.Sequential(
            nn.Conv2d(skip_dim, out_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.cross_attn = CrossAttentionBlock(out_dim)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fusion_emb, skip_feat, bin_feat=None):
        if DEBUG:
            print(f"\n -====== UnifiedFusionAttentionBlock: forward()")
            print(f" -  self.in_dim={self.in_dim}, self.skip_dim={self.skip_dim}, self.out_dim={self.out_dim}, self.use_bin_feat={self.use_bin_feat}")
            print(f" - UnifiedFusionAttentionBlock: input fusion_emb.shape={fusion_emb.shape}")
            print(f" - UnifiedFusionAttentionBlock: input skip_feat.shape={skip_feat.shape}")
            print(f" - UnifiedFusionAttentionBlock: input bin_feat.shape={bin_feat.shape}")
        
        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock: upsampling skip_feat to match fusion_emb")
            print(f" - UnifiedFusionAttentionBlock: input fusion_emb.shape={fusion_emb.shape}")
            print(f" - UnifiedFusionAttentionBlock: input skip_feat.shape={skip_feat.shape}")
        # B = fusion_emb.shape[0]
        # skip_feat = skip_feat.reshape([B, self.reshaped_shape[0], self.reshaped_shape[1], self.reshaped_shape[2]])
        # if DEBUG:
            # print(f" - UnifiedFusionAttentionBlock reshape: skip_feat.shape={skip_feat.shape}")
        skip_feat = self.conv_skip(skip_feat)
        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock conv_skip: skip_feat.shape={skip_feat.shape}")

        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock input fusion_emb, skip_feat, bin_feat.shape={fusion_emb.shape}, {skip_feat.shape}, {bin_feat.shape}")
        attn_feat = self.cross_attn(fusion_emb, skip_feat, bin_feat)

        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock: attn_feat.shape={attn_feat.shape}")
        
        gate = self.gate_conv(torch.cat([attn_feat, skip_feat, bin_feat], dim=1))
        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock: gate.shape={gate.shape}")
            
        fused = gate * attn_feat + (1 - gate) * skip_feat
        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock: fused.shape={fused.shape}")
        fused = self.fuse_conv(fused)
        if DEBUG:
            print(f" - UnifiedFusionAttentionBlock: output fused.shape={fused.shape}\n")
        return fused

# -------------------------
# Binary Decoder with 2 heads: binary head and boundary head, with auxiliary outputs
# -------------------------
class BinaryDecoder(nn.Module):
    def __init__(self, feat_dims, skip_dims=[1024, 256, 64, 16], 
                 dropout=0.1, output_aux_tokens=False):
        super().__init__()
        self.num_stages = len(feat_dims)
        self.feat_dims = feat_dims
        self.skip_dims = skip_dims
        self.output_aux_tokens = output_aux_tokens

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i-1], feat_dims[i]*4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(feat_dims[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(1, self.num_stages)
        ])

        self.fusion_blocks = nn.ModuleList([
            BinarySkipFusionBlock(feat_dims[i], skip_dims[i], feat_dims[i]) for i in range(1, self.num_stages)
        ])
        
        self.bin_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i], feat_dims[-1], kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(feat_dims[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dims[-1], 1, kernel_size=1, padding=0, bias=True)
            )
            for i in range(1, self.num_stages)
        ])

        self.final_boundary = nn.Conv2d(feat_dims[-1], 1, kernel_size=1)

    def forward(self, fusion_emb, skip_feats):
        if DEBUG:
            print(f"\n ------------- BinaryDecoder: forward()")
            print(f" - self.num_stages={self.num_stages}, self.feat_dims={self.feat_dims}, self.skip_dims={self.skip_dims}")
            print(f" - BinaryDecoder: input fusion_emb.shape={fusion_emb.shape}")
            print(f" - BinaryDecoder: input, skip_feats.shape={len(skip_feats)}")
            for i in range(len(skip_feats)):
                print(f" ----- BinaryDecoder: input, skip_feats[{i}].shape={skip_feats[i].shape}")
            # fusion_emb: [B, C, H, W]
            # skip_feats: list of [B, C, H, W]
            print(f" - BinaryDecoder: len(self.upsamples)={len(self.upsamples)}")

        x = fusion_emb
        bin_feats = []
        bin_aux_outs = []
        for i in range(len(self.upsamples)):
            if DEBUG:
                print(f"\n ----- BinaryDecoder: stage {i}")
                print(f" - BinaryDecoder: input x.shape={x.shape}")
                print(f" - BinaryDecoder: input skip_feats[{i}].shape={skip_feats[i].shape}")

            x = self.upsamples[i](x)
            if DEBUG:
                print(f" - BinaryDecoder: upsample[{i}]: x.shape={x.shape}")

            x = self.fusion_blocks[i](x, skip_feats[i])
            if DEBUG:
                print(f" - BinaryDecoder stage output : fusion_block[{i}]: x.shape={x.shape}")
            bin_feats.append(x)
            bin_aux_outs.append(self.bin_heads[i](x))

        if DEBUG:
            print(f"\n - BinaryDecoder: bin_feats.shape={len(bin_feats)}")
            for i in range(len(bin_feats)):
                print(f" - BinaryDecoder: bin_feats[{i}].shape={bin_feats[i].shape}")
            print(f"\n - BinaryDecoder: bin_aux_outs.shape={len(bin_aux_outs)}")
            for i in range(len(bin_aux_outs)):
                print(f" - BinaryDecoder: bin_aux_outs[{i}].shape={bin_aux_outs[i].shape}")

        bin_map = bin_aux_outs[-1]
        boundary = self.final_boundary(bin_feats[-1])
        if DEBUG:
            print(f" - BinaryDecoder output: bin_map.shape={bin_map.shape}")
            print(f" - BinaryDecoder output: boundary.shape={boundary.shape}\n")

        if self.output_aux_tokens:
            return bin_map, boundary, bin_feats, bin_aux_outs
        else:
            del bin_aux_outs
            torch.cuda.empty_cache()
            return bin_map, boundary, bin_feats, None


# -------------------------
# HV Decoder with bin guidance
# -------------------------
class HVDecoder(nn.Module):
    def __init__(self, feat_dims, out_ch=2, skip_dims=[1024, 256, 64, 16], output_aux_tokens=False):
        """
        feat_dims: List[int], channel dimensions for each stage (deep → shallow).
        out_ch: Output HV map channels (default: 2)
        """
        super().__init__()
        self.feat_dims = feat_dims
        self.num_stages = len(feat_dims)
        self.out_ch = out_ch
        self.skip_dims = skip_dims
        self.output_aux_tokens = output_aux_tokens

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i-1], feat_dims[i]*4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(feat_dims[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(1, self.num_stages)
        ])

        print(f"\n\n - HVDecoder: feat_dims={feat_dims}, out_ch={out_ch}, skip_dims={skip_dims}")
        self.fusion_stage_0 = UnifiedFusionAttentionBlock(feat_dims[1], skip_dims[1], feat_dims[1])
        self.fusion_stage_1 = UnifiedFusionAttentionBlock(feat_dims[2], skip_dims[2], feat_dims[2])

        self.final_fusion = TokenReducedCrossAttention(feat_dims[-1], skip_dims[-1])

        self.hv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i], feat_dims[-1], kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(feat_dims[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dims[-1], out_ch, kernel_size=1, padding=0, bias=True)
            )
            for i in range(1, self.num_stages)
        ])


    def forward(self, x, skip_feats, bin_feats):
        """
        x: initial decoder input feature [B, C, H, W]
        skip_feats: list of skip features from encoder (deep to shallow)
        bin_feats: list of corresponding binary decoder features (deep to shallow)
        """
        if DEBUG:
            print(f"\n\n ----------- HVDecoder: forward()")
            print(f" - HVDecoder self.feat_dims={self.feat_dims}, self.out_ch={self.out_ch}, self.num_stages={self.num_stages}")
            print(f" - HVDecoder: input x.shape={x.shape}")
            print(f" - HVDecoder: input skip_feats.shape={len(skip_feats)}")
            for i in range(len(skip_feats)):
                print(f" - HVDecoder: skip_feats[{i}].shape={skip_feats[i].shape}")
            print(f" - HVDecoder: input bin_feats.shape={len(bin_feats)}")
            for i in range(len(bin_feats)):
                print(f" - HVDecoder: bin_feats[{i}].shape={bin_feats[i].shape}")

        outputs = []
        for i in range(len(self.upsamples)):
            if DEBUG:
                print(f"\n ---- HVDecoder: stage {i}")
                print(f" - HVDecoder: input x.shape={x.shape}")
                print(f" - HVDecoder: input skip_feats[{i}]: skip_feats.shape={skip_feats[i].shape}")
            x = self.upsamples[i](x)
            if DEBUG:
                print(f" - HVDecoder: upsample[{i}]: x.shape={x.shape}")

            if i == 0:
                x = self.fusion_stage_0(x, skip_feats[i], bin_feats[i])
            elif i == 1:
                x = self.fusion_stage_1(x, skip_feats[i], bin_feats[i])
            else:
                x = self.final_fusion(x, skip_feats[i], bin_feats[i])

            if DEBUG:
                print(f" - HVDecoder stage output : fusion_block[{i}]: x.shape={x.shape}")

            outputs.append(self.hv_heads[i](x))

        final_hv = outputs[-1]
        if DEBUG:
            print(f"\n - HVDecoder output: final_hv.shape={final_hv.shape}")
            print(f" - HVDecoder output: outputs.shape={len(outputs)}")
            for i in range(len(outputs)):
                print(f" - HVDecoder output: outputs[{i}].shape={outputs[i].shape}")

        if self.output_aux_tokens:
            return final_hv, outputs
        else:
            del outputs
            torch.cuda.empty_cache()
            return final_hv, None

# -------------------------
# Type Decoder with Prompt and bin guidance
# -------------------------
class TokenReducedCrossAttention(nn.Module):
    def __init__(self, dim, skip_dim, token_dim=32, num_heads=4):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim * 2, token_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # Token数目缩减到 256
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Conv2d(dim, dim, 1)

        self.conv_skip = nn.Sequential(
            nn.Conv2d(skip_dim, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fusion_emb, skip_feat, bin_feat):
        if DEBUG:
            print(f"\n -====== TokenReducedCrossAttention: forward()")
            print(f" - Input fusion_emb.shape={fusion_emb.shape}")  # [2, 32, 256, 256]
            print(f" - Input skip_feat.shape={skip_feat.shape}")    # [2, 1024, 32, 32]
            print(f" - Input bin_feat.shape={bin_feat.shape}")      # [2, 32, 256, 256]

        B, C, H, W = fusion_emb.shape

        # === skip_feat reshape + conv ===
        skip_feat = self.conv_skip(skip_feat)
        if DEBUG:
            print(f" - Processed skip_feat.shape={skip_feat.shape}")    # [2, 32, 256, 256]

        # === Q: 来自 fusion_emb ===
        B, C, H, W = fusion_emb.shape
        q = self.q_proj(fusion_emb).flatten(2).transpose(1, 2)  # [B, 256 * 256, 32] = [2, 65536, 32]
        if DEBUG:
            print(f" - q from fusion_emb, flattened q.shape={q.shape}")

        # === KV: 来自 skip_feat + bin_feat ===
        kv_feat = torch.cat([skip_feat, bin_feat], dim=1)   # [B, 64, 256, 256]
        if DEBUG:
            print(f" - kv_feat from skip_feat, bin_feat, kv_feat.shape={kv_feat.shape}")

        kv_pooled = self.pool(kv_feat)  # [B, 2C, 16, 16]   
        if DEBUG:
            print(f" - kv_pooled from kv_feat, kv_pooled.shape={kv_pooled.shape}")

        kv = self.kv_proj(kv_pooled).flatten(2).transpose(1, 2)  # [B, 256, token_dim]
        if DEBUG:
            print(f" - kv from kv_pooled, flattened kv.shape={kv.shape}")

        # === Cross Attention ===
        attn_out, _ = self.attn(q, kv, kv)  # [B, HW, C]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(attn_out)

        if DEBUG:
            print(f" - attn_out.shape={attn_out.shape}, final out.shape={out.shape}")

        return out

class PromptCrossAttnBlock(nn.Module):
    def __init__(self, dec_ch, prompt_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(dec_ch, prompt_dim)
        self.key_proj = nn.Linear(prompt_dim, prompt_dim)
        self.value_proj = nn.Linear(prompt_dim, prompt_dim)
        self.attn = nn.MultiheadAttention(embed_dim=prompt_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(prompt_dim, dec_ch)

    def forward(self, x, prompt_tokens):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        q = self.query_proj(x_flat)
        k = self.key_proj(prompt_tokens).expand(B, -1, -1)
        v = self.value_proj(prompt_tokens).expand(B, -1, -1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.out_proj(attn_out).transpose(1, 2).view(B, C, H, W)
        return x + attn_out  # residual

class GATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x
        x, _ = self.attn(x, x, x)
        x = residual + x
        x = x + self.ffn(self.norm(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x
        x = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] + residual
        x = self.ffn(self.norm2(x)) + x
        return x

class PromptTokenGenerator(nn.Module):
    def __init__(self, prompt_len, embed_dim, num_heads=4, use_gat=True):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_len, embed_dim))  # [L, C]

        if use_gat:
            self.processor = GATLayer(embed_dim, heads=num_heads)
        else:
            self.processor = TransformerBlock(embed_dim, heads=num_heads)

    def forward(self, B):
        # Input: batch size
        prompt = self.prompt.unsqueeze(0).expand(B, -1, -1)  # [B, L, C]
        prompt = self.processor(prompt)  # context-aware prompt
        return prompt  # [B, L, C]


class PromptCrossTransformerBlockV3(nn.Module):
    def __init__(self, dec_ch, prompt_dim=64, num_heads=4, pool_kernel=2):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.query_proj = nn.Linear(dec_ch, prompt_dim)
        self.key_proj = nn.Linear(prompt_dim, prompt_dim)
        self.value_proj = nn.Linear(prompt_dim, prompt_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=prompt_dim, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, prompt_dim * 4),
            nn.GELU(),
            nn.Linear(prompt_dim * 4, prompt_dim)
        )
        self.out_proj = nn.Linear(prompt_dim, dec_ch)
        self.norm = nn.LayerNorm(prompt_dim)

    def forward(self, x, prompt_tokens):
        B, C, H, W = x.shape

        # ↓↓↓ 降采样空间分辨率 ↓↓↓
        x_pooled = F.avg_pool2d(x, kernel_size=self.pool_kernel)  # [B, C, H', W']
        Hp, Wp = x_pooled.shape[2:]

        x_flat = x_pooled.flatten(2).transpose(1, 2)  # [B, HW, C]
        q = self.query_proj(x_flat)  # [B, HW, prompt_dim]
        k = self.key_proj(prompt_tokens).expand(B, -1, -1)  # [B, L, prompt_dim]
        v = self.value_proj(prompt_tokens).expand(B, -1, -1)

        # ↓↓↓ 关闭权重输出避免显存浪费 ↓↓↓
        attn_out, _ = self.attn(q, k, v, need_weights=False)  # [B, HW, prompt_dim]

        attn_out = attn_out + self.ffn(self.norm(attn_out))  # residual FFN
        attn_out = self.out_proj(attn_out)                   # [B, HW, dec_ch]
        attn_out = attn_out.transpose(1, 2).view(B, C, Hp, Wp)  # reshape

        # 上采样回原始大小（与 x 相加）
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

        return x + attn_out  # residual



class PromptBinarySkipFusionBlockV3(nn.Module):
    def __init__(self, in_dim, skip_dim, use_bin_feat=True, attn_mode='cross', num_heads=4):
        super().__init__()
        self.use_bin_feat = use_bin_feat

        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        if attn_mode == 'cross':
            self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        else:
            self.attn = None

        if use_bin_feat:
            self.bin_gate = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=1),
                nn.Sigmoid()
            )

        # 残差加法后，再轻量融合
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, skip_feat, bin_feat=None):
        B, C, H, W = feat.shape
        skip = self.skip_proj(skip_feat)  # [B, C, H, W]

        if self.attn is not None:
            q = feat.flatten(2).transpose(1, 2)
            k = skip.flatten(2).transpose(1, 2)
            v = skip.flatten(2).transpose(1, 2)
            attn_out, _ = self.attn(q, k, v, need_weights=False)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        else:
            attn_out = skip

        if self.use_bin_feat and bin_feat is not None:
            gate = self.bin_gate(bin_feat)
            attn_out = feat * (1 - gate) + attn_out * gate

        # ✅ 用残差加法代替 concat，降低显存
        fused = feat + attn_out
        fused = self.fuse(fused)

        return fused


class TypeDecoderV3(nn.Module):
    def __init__(
            self, feat_dims, num_classes, 
            skip_dims=[1024, 256, 64, 16], 
            prompt_dims=[256, 128, 64, 32], 
            num_tokens = 256, 
            num_heads=4, 
            prompt_len=6, 
            output_aux_tokens=True, 
            use_multi_scale_fusion=True
       ):
        """
        Args:
            feat_dims (List[int]): Feature dimensions for each decoder stage (deep to shallow).
            num_classes (int): Number of nucleus types (excluding background).
            prompt_dim (int): Dimension of prompt tokens.
            num_heads (int): Multi-head attention count for prompt-guided attention.
            prompt_len (int): Number of learnable prompt tokens.
        """
        super().__init__()
        self.num_stages = len(feat_dims)
        self.num_classes = num_classes
        self.prompt_len = prompt_len
        self.skip_dims = skip_dims
        self.prompt_dims = prompt_dims  # [256, 128, 64, 32], 实际只取后三层
        self.num_tokens = num_tokens
        self.output_aux_tokens = output_aux_tokens

        # === Learnable Prompt Tokens ===
        # Prompt token generators
        self.prompt_tokens = nn.ModuleList([
            PromptTokenGenerator(prompt_len, prompt_dims[i], use_gat=True) for i in range(1, self.num_stages)
        ])
        
        # === Cross Attention blocks for prompt guidance
        self.cross_attn_blocks = nn.ModuleList([
            PromptCrossTransformerBlockV3(feat_dims[i], prompt_dims[i], num_heads) for i in range(1, self.num_stages)
        ])
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i-1], feat_dims[i]*4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(feat_dims[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(1, self.num_stages)
        ])

        # === Feature fusion: skip + binary feature + fusion_emb
        self.fusion_stage_1 = PromptBinarySkipFusionBlockV3(
            in_dim=feat_dims[1], skip_dim=skip_dims[1], use_bin_feat=True, attn_mode='cross')
        self.fusion_stage_2 = PromptBinarySkipFusionBlockV3(
            in_dim=feat_dims[2], skip_dim=skip_dims[2], use_bin_feat=True, attn_mode='cross')
        self.final_fusion = PromptBinarySkipFusionBlockV3(
            in_dim=feat_dims[3], skip_dim=skip_dims[3], use_bin_feat=True, attn_mode='cross')

        # === Per-stage feature heads for multi-stage deep supervision
        self.head_feats = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dims[i], feat_dims[-1], kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(feat_dims[-1]),
                nn.ReLU(inplace=True),
            )
            for i in range(1, self.num_stages)
        ])

        # 用于每个 stage 的中间输出监督
        self.head_preds = nn.ModuleList([
            nn.Conv2d(feat_dims[-1], num_classes + 1, kernel_size=1, padding=0, bias=True)
            for _ in range(1, self.num_stages)
        ])

    def forward(self, x, skip_feats, bin_feats):
        """
        Args:
            x: Tensor, initial decoder input from fusion neck [B, C, H, W]
            skip_feats: List[Tensor], encoder skip connections per stage
            bin_feats: List[Tensor], BinaryDecoder output features per stage (same shape as skip_feats)
        Returns:
            outputs: List[Tensor], type prediction maps per stage (for deep supervision)
        """
        if DEBUG:
            print(f"\n\n ----------- TypeDecoder: forward()")
            print(f" self.num_stages={self.num_stages}, self.num_classes={self.num_classes}, self.skip_dims={self.skip_dims} ")
            print(f" self.prompt_dims={self.prompt_dims}, self.prompt_len={self.prompt_len}")
            print(f" - TypeDecoder: input x.shape={x.shape}")
            print(f" - TypeDecoder: input skip_feats.shape={len(skip_feats)}")
            for i in range(len(skip_feats)):
                print(f" - TypeDecoder: skip_feats[{i}].shape={skip_feats[i].shape}")
            print(f" - TypeDecoder: input bin_feats.shape={len(bin_feats)}")
            for i in range(len(bin_feats)):
                print(f" - TypeDecoder: bin_feats[{i}].shape={bin_feats[i].shape}")
        
        B = x.size(0)
        outputs_feat = []
        outputs_pred = []

        for i in range(len(self.upsamples)):
            if DEBUG:
                print(f"\n ---- TypeDecoder: stage {i}")
                print(f" - TypeDecoder: input x.shape={x.shape}")

            x = self.upsamples[i](x)  # upsample spatial size
            if DEBUG:
                print(f" - TypeDecoder: upsample[{i}]: x.shape={x.shape}")
            
            # 使用动态生成的 prompt
            prompt_token = self.prompt_tokens[i](B)  # [B, L, C]
            if DEBUG:
                print(f" - TypeDecoder: cross_attn_blocks input x shape={x.shape}, prompt_token shape={prompt_token.shape}")

            # === Prompt Attention 引导 ===
            x = self.cross_attn_blocks[i](x, prompt_token)
            if DEBUG:
                print(f" - TypeDecoder: cross_attn_block[{i}]: x.shape={x.shape}")

            # === skip + binary 引导融合 ===
            if i == 0:
                x = self.fusion_stage_1(x, skip_feats[i], bin_feats[i])
            elif i == 1:
                x = self.fusion_stage_2(x, skip_feats[i], bin_feats[i])
            else:
                x = self.final_fusion(x, skip_feats[i], bin_feats[i])

            if DEBUG:
                print(f" - TypeDecoder stage output : fusion_block[{i}]: x.shape={x.shape}")

            # === 输出 per-stage type prediction ===
            feat = self.head_feats[i](x)      # [B, C, H, W]
            pred = self.head_preds[i](feat)   # [B, num_classes+1, H, W]
            if DEBUG:
                print(f" - TypeDecoder: head[{i}]: feat.shape={feat.shape}, pred.shape={pred.shape}")

            outputs_feat.append(feat)
            outputs_pred.append(pred)

        if DEBUG:
            print(f"\n ---- TypeDecoder: outputs_feat, outputs_pred")
            for i in range(len(outputs_feat)):
                print(f" - TypeDecoder: outputs_feat[{i}].shape={outputs_feat[i].shape}, outputs_pred[{i}].shape={outputs_pred[i].shape}")
            print(f"\n")

        final_tp = outputs_pred[-1]
        
        if DEBUG:
            print(f"\n ---- TypeDecoder: final_tp.shape={final_tp.shape}")
            print(f" ---- TypeDecoder: outputs_pred len={len(outputs_pred)}")
            for i in range(len(outputs_pred)):
                print(f" - TypeDecoder: outputs_pred[{i}].shape={outputs_pred[i].shape}")
            print(f"\n")
        
        del outputs_feat
        torch.cuda.empty_cache()
        
        if self.output_aux_tokens:
            return final_tp, outputs_pred  # 保留辅助监督的logits
        else:
            del outputs_pred
            torch.cuda.empty_cache()
            return final_tp, None