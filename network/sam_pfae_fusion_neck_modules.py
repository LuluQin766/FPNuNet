"""
PFAE Fusion‑Neck v2 & SAMNuSCNetV232
===================================

This file supersedes the previous *PFAE Family Toolkit* and provides **production‑ready implementations** for:

1. **safe frequency transforms** (GPU → CPU fallback + torch‑dct support).
2. **PFAEGlobalFusionNeckV2** – main semantic fusion path (img + SAM encoder + UNI encoder).
3. **PFAESkipEnhanceNeck** – lightweight skip‑feature enhancer.
4. **SAMNuSCNetV232** – end‑to‑end nuclei segmentation & classification network wired with the new necks.

> ⚠️ *All BatchNorm2d layers are GroupNorm‑ready via `convert_bn_to_gn`.*
"""
from __future__ import annotations
from re import T
# from regex import D, P
from sympy import use
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional, Type
from einops import rearrange
import torch_dct
import copy
import os

DEBUG = False
# DEBUG = True     # set to True to print debug messages

# 获取当前文件名
file_name = os.path.basename(__file__)
if "debug" in file_name:
    DEBUG = True
    
if DEBUG:
    print(f"\n ------ Debug mode is {DEBUG}, setting on {file_name} ------ \n")


# -----------------------------------------------------------------------------
# 1. Safe frequency transforms (GPU→CPU fallback + DCT)
# -----------------------------------------------------------------------------
def _safe_fft2(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for torch.fft.fft2 with cuFFT fallback to CPU to avoid CUDA bug in PyTorch 2.0.0.
    """
    try:
        return torch.fft.fft2(x)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cufft" in msg or "fft" in msg:
            if DEBUG:
                print(f"[safe_fft2] FFT on GPU failed: {e}")
                print("[safe_fft2] Falling back to CPU FFT.")
            out = torch.fft.fft2(x.cpu())
            return out.to(x.device)
        else:
            raise

def _safe_ifft2(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for torch.fft.ifft2 with cuFFT fallback to CPU to avoid CUDA bug in PyTorch 2.0.0.
    """
    try:
        return torch.fft.ifft2(x)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cufft" in msg or "ifft" in msg:
            if DEBUG:
                print(f"[safe_ifft2] IFFT on GPU failed: {e}")
                print("[safe_ifft2] Falling back to CPU IFFT.")
            out = torch.fft.ifft2(x.cpu())
            return out.to(x.device)
        else:
            raise

def _safe_dct2(x: torch.Tensor) -> torch.Tensor:
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()

    try:
        # force CUDA if available
        if x.is_cuda:
            y = torch_dct.dct(x, norm='ortho')
            y = y.transpose(-2, -1)
            y = torch_dct.dct(y, norm='ortho')
            y = y.transpose(-2, -1)
        else:
            raise RuntimeError("Only CUDA supported in torch_dct fallback")
    except RuntimeError as e:
        # print("[safe_dct2] FFT on GPU failed, falling back to CPU:", e)
        x_cpu = x.detach().cpu()
        y = torch_dct.dct(x_cpu, norm='ortho')
        y = y.transpose(-2, -1)
        y = torch_dct.dct(y, norm='ortho')
        y = y.transpose(-2, -1)
        return y.to(x.device)

    return y

def _safe_idct2(x: torch.Tensor) -> torch.Tensor:
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()

    try:
        # force CUDA if available
        if x.is_cuda:
            y = torch_dct.idct(x, norm='ortho')
            y = y.transpose(-2, -1)
            y = torch_dct.idct(y, norm='ortho')
            y = y.transpose(-2, -1)
        else:
            raise RuntimeError("Only CUDA supported in torch_dct fallback")
    except RuntimeError as e:
        # print("[safe_idct2] FFT on GPU failed, falling back to CPU:", e)
        x_cpu = x.detach().cpu()
        y = torch_dct.idct(x_cpu, norm='ortho')
        y = y.transpose(-2, -1)
        y = torch_dct.idct(y, norm='ortho')
        y = y.transpose(-2, -1)
        return y.to(x.device)

    return y

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)
    return normalized_tensor

# -----------------------------------------------------------------------------
# Utility: CBAM (Channel + Spatial Attention)
# -----------------------------------------------------------------------------
class _ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden, in_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(attn))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = _ChannelAttention(in_channels, reduction)
        self.sa = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -----------------------------------------------------------------------------
# Utility: CoordConv  (adds normalized coordinate channels)
# -----------------------------------------------------------------------------
class ColorFusionStem(nn.Module):
    def __init__(self, out_ch=3, mid_ch=32, n_heads=4):
        super().__init__()
        self.rgb = nn.Sequential(
            nn.Conv2d(3, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch),  # depth-wise
            nn.BatchNorm2d(mid_ch), nn.GELU(),
        )
        self.hrd = copy.deepcopy(self.rgb)
        self.attn = nn.MultiheadAttention(mid_ch, n_heads, batch_first=True)
        self.proj = nn.Conv2d(mid_ch, out_ch, 1)  # returns to 3-ch

    def forward(self, x6):               # B,6,H,W
        rgb, hrd = torch.split(x6, 3, dim=1)
        fr, fh = self.rgb(rgb), self.hrd(hrd)     # B,32,H,W
        B,C,H,W = fr.shape
        q = fr.flatten(2).transpose(1,2)          # B,(H*W),C
        kv = fh.flatten(2).transpose(1,2)
        fused,_ = self.attn(q, kv, kv)            # B,(H*W),C
        fused = fused.transpose(1,2).view(B,C,H,W)
        return self.proj(fused)                   # B,3,H,W

class ColorFusionPatchStemV2(nn.Module):
    """
    Learnable front-end that maps RGB+HED (6 ch) → 3 ch
    and preserves an identity fall-back.
    """
    def __init__(self, mid_ch=32, n_heads=4):
        super().__init__()
        # two shallow depth-wise stems
        def _stem():
            return nn.Sequential(
                nn.Conv2d(3, mid_ch, 3, padding=1),
                nn.BatchNorm2d(mid_ch), nn.GELU(),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch),
                nn.BatchNorm2d(mid_ch), nn.GELU(),
            )
        self.rgb = _stem()
        self.hed = _stem()
        # cross-attention (query = RGB)
        self.attn = nn.MultiheadAttention(mid_ch, n_heads, batch_first=True)
        # residual gating & projection back to 3 channels
        self.proj = nn.Conv2d(mid_ch, 3, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts as identity

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_rgb, x_hed):                # x_rgb shape: [B,3,H,W], x_hed shape: [B,3,H,W]
        fr, fh = self.rgb(x_rgb), self.hed(x_hed)         # B,mid,H,W
        B,C,H,W = fr.shape
        q = fr.flatten(2).transpose(1,2)              # B, HW, C
        k = v = fh.flatten(2).transpose(1,2)
        attn,_ = self.attn(q, k, v)                   # B, HW, C
        attn = attn.transpose(1,2).view(B,C,H,W)
        fused = self.proj(attn)                       # B,3,H,W
        return x_rgb + self.gamma * fused               # residual identity


class LocalAttentionConv(nn.Module):
    """Local spatial attention via 1×1 conv + sigmoid gating."""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.gamma  = nn.Parameter(torch.zeros(1))

    def forward(self, q_feat, kv_feat):
        # q_feat, kv_feat: [B, C, H, W]
        q = self.q_proj(q_feat)
        k = self.k_proj(kv_feat)
        v = self.v_proj(kv_feat)
        attn = torch.sigmoid(q * k)          # element-wise gating
        out = self.gamma * (attn * v) + q_feat
        return out

class ColorFusionStemV3(nn.Module):
    """
    RGB+HED (6-ch) → 3-ch patch input, with stain-conditioned channel gating.
    Args
    ----
    mid_ch   : hidden dim for each branch
    stain_flag : B,     0=HE, 1=IHC  (float or long)
    """
    def __init__(self, mid_ch=32, stain_flag: torch.Tensor=1):
        super().__init__()
        self.mid_ch = mid_ch
        # self.n_heads = n_heads

        # 处理 stain_flag 为 tensor
        if isinstance(stain_flag, int):
            stain_flag = torch.tensor(float(stain_flag), dtype=torch.float32)
        elif isinstance(stain_flag, float):
            stain_flag = torch.tensor(stain_flag, dtype=torch.float32)

        self.register_buffer("stain_flag", stain_flag)  # 不参与训练，自动 device 迁移

        self.stain_flag = stain_flag

        # shared tiny stem
        def _stem():
            return nn.Sequential(
                nn.Conv2d(3, mid_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_ch), nn.GELU(),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch),
                nn.BatchNorm2d(mid_ch), nn.GELU(),
            )
        self.rgb_stem = _stem()            # RGB → mid
        self.h_stem   = nn.Sequential(     # 1-ch → mid
            nn.Conv2d(1, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.GELU())
        self.e_stem   = copy.deepcopy(self.h_stem)
        self.d_stem   = copy.deepcopy(self.h_stem)

        # learnable stain-specific scalars (initialised with prior)
        self.scale_he  = nn.Parameter(torch.tensor([1.0, 0.5, 0.0]))  # H,E,D
        self.scale_ihc = nn.Parameter(torch.tensor([1.0, 0.0, 1.0]))

        # cross-attention: RGB query, fused-HED key/value
        self.attn = LocalAttentionConv(mid_ch)

        # project back to 3-ch and residual gate γ
        self.proj  = nn.Conv2d(mid_ch, 3, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    # ------------------------------------------------------------------ #
    def forward(self, x6: torch.Tensor):
        """
        x6         : B, 6, H, W  (RGB||HED)
        """
        if DEBUG:
            print(f"\n ------- ColorFusionStemV3 forward -------- ")
            print(f" Args: mid_ch={self.mid_ch}, stain_flag={self.stain_flag}")
            print(f" input: x6.shape={x6.shape}")

        x_rgb, x_hed = torch.split(x6, 3, dim=1)      # B,3,H,W each
        h, e, d  = x_hed[:,0:1], x_hed[:,1:2], x_hed[:,2:3]

        fr = self.rgb_stem(x_hed)                   # B,mid,H,W
        fh = self.h_stem(h)
        fe = self.e_stem(e)
        fd = self.d_stem(d)

        # ---------- stain-conditioned fusion weights ----------
        s = self.stain_flag.view(1, 1, 1, 1).to(x6.device, dtype=x6.dtype)  # broadcast
        w = (1 - s) * self.scale_he + s * self.scale_ihc                    # shape (B,3)
        if DEBUG:
            print(f" stain_flag: {self.stain_flag.shape} = {self.stain_flag}")
            print(f" fr: {fr.shape}, fh: {fh.shape}, fe: {fe.shape}, fd: {fd.shape}")
            print(f" s: {s.shape}, w: {w.shape}")
        
        wH, wE, wD = torch.chunk(w, chunks=3, dim=-1)  # each shape: [1,1,1,1]

        if DEBUG:
            print(f" wH: {wH.shape}, wE: {wE.shape}, wD: {wD.shape}")

        fhed = wH*fh + wE*fe + wD*fd               # B,mid,H,W

        # ---------- cross-attention RGB↔HED ----------
        # cross-attention via conv-based gating
        attn = self.attn(fr, fhed)  # B,C,H,W

        out = x_rgb + self.gamma * self.proj(attn)   # residual identity
        if DEBUG:
            print(f" output: out.shape={out.shape}\n")
        return out
    
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

# -----------------------------------------------------------------------------
# Utility: CoordConv  (adds normalized coordinate channels)
# -----------------------------------------------------------------------------
class CoordConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, with_r: bool = False):
        super().__init__()
        extra = 2 + (1 if with_r else 0)
        self.with_r = with_r
        self.proj = nn.Conv2d(in_channels + extra, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij",
        )
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        if self.with_r:
            rr = torch.sqrt(xx ** 2 + yy ** 2).expand(b, 1, h, w)
            x = torch.cat([x, xx, yy, rr], dim=1)
        else:
            x = torch.cat([x, xx, yy], dim=1)
        return self.proj(x)


# -----------------------------------------------------------------------------
# Heavy Spectral Stage (from Original PFAE) – Optional
# -----------------------------------------------------------------------------
class PFAEHeavyStage(nn.Module):
    """A single heavy spectral enhancement stage derived from the Original PFAE.

    Intended for *one* insertion in PFAEv5Hybrid (usually stage_idx=0 at coarsest scale).
    Includes dual-path frequency modeling: inter-frequency attention + CBAM-gated local spectrum.
    """
    def __init__(
            self, 
            channels: int, 
            num_heads: int = 8, 
            temperature: float = 1.0
        ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * temperature)
        self.cbam = CBAMBlock(channels)
        self.project_out = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, freq_fn=_safe_fft2, ifreq_fn=_safe_ifft2) -> torch.Tensor:

        b, c, h, w = x.shape
        # Frequency domain rep
        fx = freq_fn(x.float())  # complex
        # Inter-frequency attention: treat channels as (head, c_per_head)
        cph = c // self.num_heads
        if c % self.num_heads != 0:
            # pad channels
            pad = self.num_heads * cph + (self.num_heads - (c % self.num_heads)) - c
            fx = F.pad(fx, (0,0,0,0,0,pad))  # pad C-dim; cheap fix
            c = fx.shape[1]
            cph = c // self.num_heads
        q = rearrange(fx, 'b (hds cph) H W -> b hds cph (H W)', hds=self.num_heads)
        k = q  # same feature
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # custom norm (magnitude wise)
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v  # complex matmul via broadcast
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)

        out_f = torch.abs(ifreq_fn(out_f))  # magnitude restore

        # Local gated frequency (CBAM on real amplitude)
        fx_local = freq_fn(x.float())
        fx_gated = self.cbam(fx_local.real) * fx_local  # broadcast over complex
        out_f_l = torch.abs(ifreq_fn(fx_gated))

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out


# -----------------------------------------------------------------------------
# PFAEv5Hybrid
# -----------------------------------------------------------------------------
class PFAEv5Hybrid(nn.Module):
    """Hybrid Progressive Frequency-Aware Enhancement Block.

    Combines:
    - Lightweight multi-dilation residual freq-enhanced path (v4 lineage).
    - Optional *heavy* spectral stage (Original lineage) injected at chosen stage index.
    - Global pooled context + CoordConv positional enrichment.

    Args
    ----
    dim: int
        Input feature channels.
    in_dim: int
        Internal processing channels after first reduction.
    out_channels: int
        Output feature channels delivered to downstream neck/decoder.
    num_stages: int
        Number of dilation stages.
    min_channels: int
        Floor for reduced channel width.
    use_dct: bool
        Use DCT-like real transform instead of FFT complex.
    heavy_stage_cfg: Optional[dict]
        {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
        If enabled, inserts PFAEHeavyStage after specified stage.
    coord_with_r: bool
        Whether to include radial distance in CoordConv.
    bn2gn_ready: bool
        Placeholder flag; no behavior change, indicates block is safe for GN swap.
    """
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 4,
        min_channels: int = 16,
        use_dct: bool = False,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        coord_with_r: bool = False,
        bn2gn_ready: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stages = num_stages
        self.min_channels = min_channels
        self.use_dct = use_dct
        self.heavy_stage_cfg = heavy_stage_cfg
        self.coord_with_r = coord_with_r

        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        down_dim = max(in_dim // 2, min_channels)
        # adaptive head count for lightweight attn
        self.num_heads = 4 if down_dim <= 64 else 2
        head_dim = max(down_dim // self.num_heads, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(True),
        )

        self.cbam = CBAMBlock(down_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, 1, bias=False)

        # progressive multi-dilation conv stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(down_dim, down_dim, 3, dilation=3 + 2 * i, padding=3 + 2 * i, bias=False),
                nn.BatchNorm2d(down_dim),
                nn.ReLU(True),
            )
            for i in range(num_stages)
        ])

        # optional heavy spectral stage injection
        if heavy_stage_cfg is None:
            heavy_stage_cfg = {}
        self.heavy_enable = bool(heavy_stage_cfg.get("enable", False))
        self.heavy_stage_idx = int(heavy_stage_cfg.get("stage_idx", 0))
        heavy_heads = int(heavy_stage_cfg.get("num_heads", 8))
        heavy_temp  = float(heavy_stage_cfg.get("gamma_init", 1.0))
        if self.heavy_enable:
            self.heavy_stage = PFAEHeavyStage(down_dim, num_heads=heavy_heads, temperature=heavy_temp)
            self.heavy_gamma = nn.Parameter(torch.tensor(heavy_stage_cfg.get("gamma_init", 0.1), dtype=torch.float32))
        else:
            self.heavy_stage = None
            self.heavy_gamma = None

        # global pooling context
        self.conv6 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, 1, bias=False),
            nn.GroupNorm(1, down_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # fuse all stage outputs + global
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * down_dim, down_dim, 1, bias=False),
            nn.GroupNorm(1, down_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # final projection -> out_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(down_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

        self.coord = CoordConv(out_dim, out_dim, with_r=coord_with_r)

    # ---- internal helpers --------------------------------------------------
    def _freq_pair(self):
        if self.use_dct:
            return _safe_dct2, _safe_idct2
        else:
            return _safe_fft2, _safe_ifft2

    def _light_freq_enhance(self, feat: torch.Tensor) -> torch.Tensor:
        """Lightweight frequency attention path (v4 lineage)."""
        freq_fn, ifreq_fn = self._freq_pair()
        b, c, h, w = feat.shape

        fx = freq_fn(feat.float())  # complex (FFT) or real (DCT-proxy)
        if torch.is_complex(fx):
            q = fx; k = fx; v = fx
        else:
            q = fx; k = fx; v = fx

        # reshape to heads
        q = rearrange(q, 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads) if c % self.num_heads == 0 else rearrange(F.pad(q, (0,0,0,0,0,self.num_heads - (c % self.num_heads))), 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads)
        k = q
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(ifreq_fn(out_f)) if torch.is_complex(out_f) else ifreq_fn(out_f)

        # CBAM-gated local spectrum
        if torch.is_complex(fx):
            gated = self.cbam(fx.real) * fx
            out_f_l = torch.abs(ifreq_fn(gated))
        else:
            gated = self.cbam(fx)
            out_f_l = ifreq_fn(gated)

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if DEBUG:
            print(f"\n -------------- PFAEv5Hybrid.forward() --------------")
            print(f" Args: in_dim={self.in_dim}, out_dim={self.out_dim}, num_stages={self.num_stages}, min_channels={self.min_channels}, use_dct={self.use_dct}")
            print(f" Args: heavy_stage_cfg={self.heavy_stage_cfg}, coord_with_r={self.coord_with_r}")
            print(f" input x.shape: {x.shape}")

        x = self.down_conv(x)
        if DEBUG:
            print(f" down_conv x.shape: {x.shape}")

        x_reduced = self.conv1(x)
        if DEBUG:
            print(f" conv1 x_reduced.shape: {x_reduced.shape} = residual")

        outputs = [x_reduced]
        residual = x_reduced

        for i, stage in enumerate(self.stages):
            if DEBUG:
                print(f"  ------- stage {i+1}, input x.shape: {x.shape}")
            feat = stage(residual)
            if DEBUG:
                print(f"   stage feat.shape: {feat.shape}")
            # light freq enhance
            light = self._light_freq_enhance(feat)
            if DEBUG:
                print(f"   stage light.shape: {light.shape}")
            out = feat + light
            if DEBUG:
                print(f"   stage out.shape: {out.shape} = residual + light")

            # optional heavy stage injection
            if self.heavy_enable and i == self.heavy_stage_idx:
                heavy = self.heavy_stage(feat)
                out = out + self.heavy_gamma * heavy
                if DEBUG:
                    print(f"   stage heavy.shape: {heavy.shape} = gamma * (residual + light), self.heavy_gamma={self.heavy_gamma}")

            residual = x_reduced + out
            if DEBUG:
                print(f"   stage residual.shape: {residual.shape} = x_reduced + out")

            outputs.append(out)

        if DEBUG:
            print(f"   stage outputs: {len(outputs)} x {outputs[0].shape}")

        # global pooled context
        conv6 = F.adaptive_avg_pool2d(x_reduced, 1)
        if DEBUG:
            print(f"   global pool conv6.shape: {conv6.shape}")

        # fuse all stage outputs + global
        conv6 = self.conv6(conv6)
        if DEBUG:
            print(f"   fuse all stage outputs + global, conv6.shape: {conv6.shape} = residual")

        conv6 = F.interpolate(conv6, size=x.shape[2:], mode='bilinear', align_corners=False)
        if DEBUG:
            print(f"   upsample conv6.shape: {conv6.shape} = x.shape")
        outputs.append(conv6)

        if DEBUG:
            print(f"   outputs: {len(outputs)} x {outputs[0].shape}")

        fused = self.fuse(torch.cat(outputs, dim=1))
        if DEBUG:
            print(f"   fused.shape: {fused.shape} = self.fuse(torch.cat(outputs, dim=1))")
        fused = self.out_proj(fused)
        if DEBUG:
            print(f"   out_proj.shape: {fused.shape} = self.out_proj(fused)")

        fused = self.coord(fused)
        if DEBUG:
            print(f"   coord.shape: {fused.shape} = self.coord(fused)\n")
        return fused


# -----------------------------------------------------------------------------
# Heavy Spectral Stage (from Original PFAE) – Optional
# -----------------------------------------------------------------------------
class PFAEHeavyStageV2(nn.Module):
    """A single heavy spectral enhancement stage derived from the Original PFAE.

    Intended for *one* insertion in PFAEv5Hybrid (usually stage_idx=0 at coarsest scale).
    Includes dual-path frequency modeling: inter-frequency attention + CBAM-gated local spectrum.
    """
    def __init__(
            self, 
            channels: int, 
            num_heads: int = 8, 
            temperature: float = 1.0,
            use_dct: bool = False
        ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * temperature)
        self.use_dct = use_dct

        self.cbam = CBAMBlock(channels)
        self.project_out = nn.Conv2d(channels * 2, channels, 1, bias=False)

        if self.use_dct:
            self.freq_fn = _safe_dct2
            self.ifreq_fn = _safe_idct2
        else:
            self.freq_fn = _safe_fft2
            self.ifreq_fn = _safe_ifft2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            print(f"\n -------------- PFAEHeavyStageV2.forward() --------------")
            print(f" Args: channels={self.channels}, num_heads={self.num_heads}, use_dct={self.use_dct}")
            print(f" input x.shape: {x.shape}")
            if self.use_dct:
                print(f" [!!!!] Using DCT-like real transform.")
            else:
                print(f" [!!!!] Using FFT complex transform.")

        b, orig_c, h, w = x.shape

        # 1) Compute a single freq-domain tensor and pad it
        fx = self.freq_fn(x.float())  # complex tensor, shape [b, orig_c, h, w]
        if DEBUG:
            print(f" fx.shape: {fx.shape} = self.freq_fn(x.float())")

        c = orig_c
        if c % self.num_heads != 0:
            # pad channels up to next multiple of num_heads
            new_c = ((c + self.num_heads - 1) // self.num_heads) * self.num_heads
            pad   = new_c - c
            fx    = F.pad(fx, (0,0, 0,0, 0,pad))  # pad on C dim
            c     = new_c
        
        if DEBUG:
            print(f"  pad fx from {orig_c} to {c} channels")
            print(f" fx.shape: {fx.shape} = F.pad(fx, (0,0,0,0,0,pad))")

        assert c % self.num_heads == 0
        cph = c // self.num_heads

        if DEBUG:
            print(f"  2) Inter‐frequency attention ")
            print(f"     fx.shape: {fx.shape}, hds={self.num_heads}, cph={cph}")

        # 2) Inter‐frequency attention
        q = rearrange(fx, 'b (hds cph) H W -> b hds cph (H W)', hds=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = q; v = q
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(self.ifreq_fn(out_f))

        if DEBUG:
            print(f"    --> out_f.shape: {out_f.shape}")
            print(f"  3) Local‐gated frequency (use the *same* padded fx)")
            print(f"     fx.shape: {fx.shape}")

        # 3) Local‐gated frequency (use the *same* padded fx)
        fx_real = fx.real           # [b, c, h, w]
        fx_gated = self.cbam(fx_real) * fx
        out_f_l = torch.abs(self.ifreq_fn(fx_gated))

        if DEBUG:
            print(f"    fx_gated.shape: {fx_gated.shape}")
            print(f"    --> out_f_l.shape: {out_f_l.shape}")

        # 4) Crop both branches back to original channels
        out_f   = out_f[:, :orig_c, :, :]
        out_f_l = out_f_l[:, :orig_c, :, :]

        if DEBUG:
            print(f"  4) Fuse and project")
            print(f"    out_f.shape: {out_f.shape}, out_f_l.shape: {out_f_l.shape}")

        # 5) Fuse and project
        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        if DEBUG:
            print(f"    --> out.shape: {out.shape}\n")

        return out


# -----------------------------------------------------------------------------
# PFAEv6Hybrid
# -----------------------------------------------------------------------------
class PFAEv6Hybrid(nn.Module):
    """Hybrid Progressive Frequency-Aware Enhancement Block.

    Combines:
    - Lightweight multi-dilation residual freq-enhanced path (v4 lineage).
    - Optional *heavy* spectral stage (Original lineage) injected at chosen stage index.
    - Global pooled context + CoordConv positional enrichment.

    Args
    ----
    dim: int
        Input feature channels.
    in_dim: int
        Internal processing channels after first reduction.
    out_channels: int
        Output feature channels delivered to downstream neck/decoder.
    num_stages: int
        Number of dilation stages.
    min_channels: int
        Floor for reduced channel width.
    use_dct: bool
        Use DCT-like real transform instead of FFT complex.
    heavy_stage_cfg: Optional[dict]
        {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
        If enabled, inserts PFAEHeavyStage after specified stage.
    coord_with_r: bool
        Whether to include radial distance in CoordConv.
    bn2gn_ready: bool
        Placeholder flag; no behavior change, indicates block is safe for GN swap.
    """
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 4,
        min_channels: int = 16,
        use_dct: bool = False,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        coord_with_r: bool = False,
        bn2gn_ready: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stages = num_stages
        self.min_channels = min_channels
        self.use_dct = use_dct
        self.heavy_stage_cfg = heavy_stage_cfg
        self.coord_with_r = coord_with_r

        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        down_dim = max(in_dim // 2, min_channels)
        # adaptive head count for lightweight attn
        self.num_heads = 4 if down_dim <= 64 else 2
        head_dim = max(down_dim // self.num_heads, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, 1, bias=False),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(True),
        )

        self.cbam = CBAMBlock(down_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, 1, bias=False)

        # progressive multi-dilation conv stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(down_dim, down_dim, 3, dilation=3 + 2 * i, padding=3 + 2 * i, bias=False),
                nn.BatchNorm2d(down_dim),
                nn.ReLU(True),
            )
            for i in range(num_stages)
        ])

        # optional heavy spectral stage injection
        if heavy_stage_cfg is None:
            heavy_stage_cfg = {}
        self.heavy_enable = bool(heavy_stage_cfg.get("enable", False))
        self.heavy_stage_idx = int(heavy_stage_cfg.get("stage_idx", 0))
        heavy_heads = int(heavy_stage_cfg.get("num_heads", 8))
        heavy_temp  = float(heavy_stage_cfg.get("gamma_init", 1.0))
        if self.heavy_enable:
            self.heavy_stage = PFAEHeavyStageV2(down_dim, num_heads=heavy_heads, temperature=heavy_temp, use_dct=use_dct)
            self.heavy_gamma = nn.Parameter(torch.tensor(heavy_stage_cfg.get("gamma_init", 0.1), dtype=torch.float32))
        else:
            self.heavy_stage = None
            self.heavy_gamma = None

        # global pooling context
        self.conv6 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, 1, bias=False),
            nn.GroupNorm(1, down_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # fuse all stage outputs + global
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * down_dim, down_dim, 1, bias=False),
            nn.GroupNorm(1, down_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # final projection -> out_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(down_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

        self.coord = CoordConv(out_dim, out_dim, with_r=coord_with_r)

    # ---- internal helpers --------------------------------------------------
    def _freq_pair(self):
        if self.use_dct:
            return _safe_dct2, _safe_idct2
        else:
            return _safe_fft2, _safe_ifft2

    def _light_freq_enhance(self, feat: torch.Tensor) -> torch.Tensor:
        """Lightweight frequency attention path (v4 lineage)."""
        freq_fn, ifreq_fn = self._freq_pair()
        b, c, h, w = feat.shape

        fx = freq_fn(feat.float())  # complex (FFT) or real (DCT-proxy)
        if torch.is_complex(fx):
            q = fx; k = fx; v = fx
        else:
            q = fx; k = fx; v = fx

        # reshape to heads
        q = rearrange(q, 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads) if c % self.num_heads == 0 else rearrange(F.pad(q, (0,0,0,0,0,self.num_heads - (c % self.num_heads))), 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads)
        k = q
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(ifreq_fn(out_f)) if torch.is_complex(out_f) else ifreq_fn(out_f)

        # CBAM-gated local spectrum
        if torch.is_complex(fx):
            gated = self.cbam(fx.real) * fx
            out_f_l = torch.abs(ifreq_fn(gated))
        else:
            gated = self.cbam(fx)
            out_f_l = ifreq_fn(gated)

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if DEBUG:
            print(f"\n -------------- PFAEv6Hybrid.forward() --------------")
            print(f" Args: in_dim={self.in_dim}, out_dim={self.out_dim}, num_stages={self.num_stages}, min_channels={self.min_channels}, use_dct={self.use_dct}")
            print(f" Args: heavy_stage_cfg={self.heavy_stage_cfg}, coord_with_r={self.coord_with_r}")
            print(f" input x.shape: {x.shape}")

        x = self.down_conv(x)
        if DEBUG:
            print(f" down_conv x.shape: {x.shape}")

        x_reduced = self.conv1(x)
        if DEBUG:
            print(f" conv1 x_reduced.shape: {x_reduced.shape} = residual")

        outputs = [x_reduced]
        residual = x_reduced

        for i, stage in enumerate(self.stages):
            if DEBUG:
                print(f"  ------- stage {i+1}, input x.shape: {x.shape}")
            feat = stage(residual)
            if DEBUG:
                print(f"   stage feat.shape: {feat.shape}")
            # light freq enhance
            light = self._light_freq_enhance(feat)
            if DEBUG:
                print(f"   stage light.shape: {light.shape}")
            out = feat + light
            if DEBUG:
                print(f"   stage out.shape: {out.shape} = residual + light")

            # optional heavy stage injection
            if self.heavy_enable and i == self.heavy_stage_idx:
                heavy = self.heavy_stage(feat)
                out = out + self.heavy_gamma * heavy
                if DEBUG:
                    print(f"   stage heavy.shape: {heavy.shape} = gamma * (residual + light), self.heavy_gamma={self.heavy_gamma}")

            residual = x_reduced + out
            if DEBUG:
                print(f"   stage residual.shape: {residual.shape} = x_reduced + out")

            outputs.append(out)

        if DEBUG:
            print(f"   stage outputs: {len(outputs)} x {outputs[0].shape}")

        # global pooled context
        conv6 = F.adaptive_avg_pool2d(x_reduced, 1)
        if DEBUG:
            print(f"   global pool conv6.shape: {conv6.shape}")

        # fuse all stage outputs + global
        conv6 = self.conv6(conv6)
        if DEBUG:
            print(f"   fuse all stage outputs + global, conv6.shape: {conv6.shape} = residual")

        conv6 = F.interpolate(conv6, size=x.shape[2:], mode='bilinear', align_corners=False)
        if DEBUG:
            print(f"   upsample conv6.shape: {conv6.shape} = x.shape")
        outputs.append(conv6)

        if DEBUG:
            print(f"   outputs: {len(outputs)} x {outputs[0].shape}")

        fused = self.fuse(torch.cat(outputs, dim=1))
        if DEBUG:
            print(f"   fused.shape: {fused.shape} = self.fuse(torch.cat(outputs, dim=1))")
        fused = self.out_proj(fused)
        if DEBUG:
            print(f"   out_proj.shape: {fused.shape} = self.out_proj(fused)")

        fused = self.coord(fused)
        if DEBUG:
            print(f"   coord.shape: {fused.shape} = self.coord(fused)\n")
        return fused


# -----------------------------------------------------------------------------
# PFAEv7Hybrid， 从PFAEv6Hybrid的各个stage串联改为parallel，实现多尺度特征提取和融合
# -----------------------------------------------------------------------------
class PFAEv7Hybrid(nn.Module):
    """Hybrid Progressive Frequency-Aware Enhancement Block.

    Combines:
    - Lightweight multi-dilation residual freq-enhanced path (v4 lineage).
    - Optional *heavy* spectral stage (Original lineage) injected at chosen stage index.
    - Global pooled context + CoordConv positional enrichment.

    Args
    ----
    dim: int
        Input feature channels.
    in_dim: int
        Internal processing channels after first reduction.
    out_channels: int
        Output feature channels delivered to downstream neck/decoder.
    num_stages: int
        Number of dilation stages.
    min_channels: int
        Floor for reduced channel width.
    use_dct: bool
        Use DCT-like real transform instead of FFT complex.
    heavy_stage_cfg: Optional[dict]
        {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
        If enabled, inserts PFAEHeavyStage after specified stage.
    coord_with_r: bool
        Whether to include radial distance in CoordConv.
    bn2gn_ready: bool
        Placeholder flag; no behavior change, indicates block is safe for GN swap.
    """
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 4,
        min_channels: int = 16,
        use_dct: bool = False,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        coord_with_r: bool = False,
        bn2gn_ready: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stages = num_stages
        self.min_channels = min_channels
        self.use_dct = use_dct
        self.heavy_stage_cfg = heavy_stage_cfg
        self.coord_with_r = coord_with_r

        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        # down_dim = max(in_dim // 2, min_channels)
        # # adaptive head count for lightweight attn
        self.num_heads = 4 if dim <= 64 else 2
        # head_dim = max(down_dim // self.num_heads, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        self.cbam = CBAMBlock(in_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(in_dim * 2, in_dim, 1, bias=False)

        # progressive multi-dilation conv stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, dilation=3 + 2 * i, padding=3 + 2 * i, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(True),
            )
            for i in range(num_stages)
        ])

        # global pooling context
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.GroupNorm(1, in_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # fuse all stage outputs + global
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # final projection -> out_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

        self.coord = CoordConv(out_dim, out_dim, with_r=coord_with_r)

    # ---- internal helpers --------------------------------------------------
    def _freq_pair(self):
        if self.use_dct:
            return _safe_dct2, _safe_idct2
        else:
            return _safe_fft2, _safe_ifft2

    def _light_freq_enhance(self, feat: torch.Tensor) -> torch.Tensor:
        """Lightweight frequency attention path (v4 lineage)."""
        freq_fn, ifreq_fn = self._freq_pair()
        b, c, h, w = feat.shape

        fx = freq_fn(feat.float())  # complex (FFT) or real (DCT-proxy)
        if torch.is_complex(fx):
            q = fx; k = fx; v = fx
        else:
            q = fx; k = fx; v = fx

        # reshape to heads
        q = rearrange(q, 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads) if c % self.num_heads == 0 else rearrange(F.pad(q, (0,0,0,0,0,self.num_heads - (c % self.num_heads))), 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads)
        k = q
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(ifreq_fn(out_f)) if torch.is_complex(out_f) else ifreq_fn(out_f)

        # CBAM-gated local spectrum
        if torch.is_complex(fx):
            gated = self.cbam(fx.real) * fx
            out_f_l = torch.abs(ifreq_fn(gated))
        else:
            gated = self.cbam(fx)
            out_f_l = ifreq_fn(gated)

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if DEBUG:
            print(f"\n -------------- PFAEv7Hybrid.forward() --------------")
            print(f" Args: in_dim={self.in_dim}, out_dim={self.out_dim}, num_stages={self.num_stages}, min_channels={self.min_channels}, use_dct={self.use_dct}")
            print(f" Args: heavy_stage_cfg={self.heavy_stage_cfg}, coord_with_r={self.coord_with_r}")
            print(f" input x.shape: {x.shape}")
        
        # Stage input: strong backbone feature
        x_feat = self.down_conv(x)            # → 288通道，高表达特征，输入各 dilation stage

        if DEBUG:
            print(f" down_conv x_feat.shape: {x_feat.shape}")

        # Projection for residual + fusion path
        x_reduced = self.conv1(x_feat)        # → 144通道，用于 concat 与最终融合

        if DEBUG:
            print(f" conv1 x_reduced.shape: {x_reduced.shape} = residual")

        outputs = [x_reduced]    # init with reduced path (as base feature)

        for i, stage in enumerate(self.stages):
            if DEBUG:
                print(f"  ------- stage {i+1}, input x.shape: {x.shape}")
            feat = stage(x_feat)    # ✅ 重点修改：统一使用 down_conv 的输出
            if DEBUG:
                print(f"   stage feat.shape: {feat.shape}")
            # light freq enhance
            light = self._light_freq_enhance(feat)
            if DEBUG:
                print(f"   stage light.shape: {light.shape}")
            out = feat + light
            if DEBUG:
                print(f"  --> stage out.shape: {out.shape} = residual + light")

            outputs.append(out)

        # global pooled context
        conv6 = F.adaptive_avg_pool2d(x_reduced, 1)
        if DEBUG:
            print(f"   global pool conv6.shape: {conv6.shape}")

        # fuse all stage outputs + global
        conv6 = self.conv6(conv6)
        if DEBUG:
            print(f"   fuse all stage outputs + global, conv6.shape: {conv6.shape} = residual")

        conv6 = F.interpolate(conv6, size=x.shape[2:], mode='bilinear', align_corners=False)
        if DEBUG:
            print(f"   upsample conv6.shape: {conv6.shape} = x.shape")
        outputs.append(conv6)

        if DEBUG:
            print(f"   outputs: {len(outputs)} x {outputs[0].shape}")

        fused = self.fuse(torch.cat(outputs, dim=1))
        if DEBUG:
            print(f"   fused.shape: {fused.shape} = self.fuse(torch.cat(outputs, dim=1))")
        fused = self.out_proj(fused)
        if DEBUG:
            print(f"   out_proj.shape: {fused.shape} = self.out_proj(fused)")

        fused = self.coord(fused)
        if DEBUG:
            print(f"   coord.shape: {fused.shape} = self.coord(fused)\n")
        return fused



# -----------------------------------------------------------------------------
# PFAEv8Hybrid， 从PFAEv7Hybrid的将每个stage的output直接concat，改为Per-branch learnable weighting
# -----------------------------------------------------------------------------
class PFAEv8Hybrid(nn.Module):
    """Hybrid Progressive Frequency-Aware Enhancement Block.

    Combines:
    - Lightweight multi-dilation residual freq-enhanced path (v4 lineage).
    - Optional *heavy* spectral stage (Original lineage) injected at chosen stage index.
    - Global pooled context + CoordConv positional enrichment.

    Args
    ----
    dim: int
        Input feature channels.
    in_dim: int
        Internal processing channels after first reduction.
    out_channels: int
        Output feature channels delivered to downstream neck/decoder.
    num_stages: int
        Number of dilation stages.
    min_channels: int
        Floor for reduced channel width.
    use_dct: bool
        Use DCT-like real transform instead of FFT complex.
    heavy_stage_cfg: Optional[dict]
        {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
        If enabled, inserts PFAEHeavyStage after specified stage.
    coord_with_r: bool
        Whether to include radial distance in CoordConv.
    bn2gn_ready: bool
        Placeholder flag; no behavior change, indicates block is safe for GN swap.
    """
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 4,
        min_channels: int = 16,
        use_dct: bool = False,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        use_coord_gate: bool = True,
        coord_with_r: bool = True,
        bn2gn_ready: bool = True,
        use_weighted_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stages = num_stages
        self.min_channels = min_channels
        self.use_dct = use_dct
        self.heavy_stage_cfg = heavy_stage_cfg
        self.coord_with_r = coord_with_r

        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        # down_dim = max(in_dim // 2, min_channels)
        # # adaptive head count for lightweight attn
        self.num_heads = 4 if dim <= 64 else 2
        # head_dim = max(down_dim // self.num_heads, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        self.cbam = CBAMBlock(in_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(in_dim * 2, in_dim, 1, bias=False)

        # progressive multi-dilation conv stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, dilation=3 + 2 * i, padding=3 + 2 * i, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(True),
            )
            for i in range(num_stages)
        ])

        # learnable fusion weights for each branch
        self.use_weighted_fusion = use_weighted_fusion      # default True
        self.fusion_weights = nn.Parameter(torch.ones(self.num_stages + 2))  # x_reduced + stages + global
        self.fuse_linear = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, 1, bias=False),
            nn.GroupNorm(1, self.out_dim),
            nn.ReLU(True)
        )

        # global pooling context
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.GroupNorm(1, in_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # fuse all stage outputs + global
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # final projection -> out_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

        # coord gate for coordconv (optional)
        self.use_coord_gate = use_coord_gate    # default True
        self.coord_gate_conv = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False)
        self.coord = CoordConv(self.out_dim, self.out_dim, with_r=coord_with_r)

    # ---- internal helpers --------------------------------------------------
    def _freq_pair(self):
        if self.use_dct:
            return _safe_dct2, _safe_idct2
        else:
            return _safe_fft2, _safe_ifft2

    def _light_freq_enhance(self, feat: torch.Tensor) -> torch.Tensor:
        """Lightweight frequency attention path (v4 lineage)."""
        freq_fn, ifreq_fn = self._freq_pair()
        b, c, h, w = feat.shape

        fx = freq_fn(feat.float())  # complex (FFT) or real (DCT-proxy)
        if torch.is_complex(fx):
            q = fx; k = fx; v = fx
        else:
            q = fx; k = fx; v = fx

        # reshape to heads
        q = rearrange(q, 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads) if c % self.num_heads == 0 else rearrange(F.pad(q, (0,0,0,0,0,self.num_heads - (c % self.num_heads))), 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads)
        k = q
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(ifreq_fn(out_f)) if torch.is_complex(out_f) else ifreq_fn(out_f)

        # CBAM-gated local spectrum
        if torch.is_complex(fx):
            gated = self.cbam(fx.real) * fx
            out_f_l = torch.abs(ifreq_fn(gated))
        else:
            gated = self.cbam(fx)
            out_f_l = ifreq_fn(gated)

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if DEBUG:
            print(f"\n -------------- PFAEv8Hybrid.forward() --------------")
            print(f" Args: in_dim={self.in_dim}, out_dim={self.out_dim}, num_stages={self.num_stages}")
            print(f" Args: min_channels={self.min_channels}, use_dct={self.use_dct}")
            print(f" Args: heavy_stage_cfg={self.heavy_stage_cfg}, coord_with_r={self.coord_with_r}")
            print(f" input x.shape: {x.shape}")
        
        # Stage input: strong backbone feature
        x_feat = self.down_conv(x)            # → 288通道，高表达特征，输入各 dilation stage

        if DEBUG:
            print(f" down_conv x_feat.shape: {x_feat.shape}")

        # Projection for residual + fusion path
        x_reduced = self.conv1(x_feat)        # → 144通道，用于 concat 与最终融合

        if DEBUG:
            print(f" conv1 x_reduced.shape: {x_reduced.shape} = residual")

        outputs = [x_reduced]    # init with reduced path (as base feature)

        for i, stage in enumerate(self.stages):
            if DEBUG:
                print(f"  ------- stage {i+1}, input x.shape: {x.shape}")
            feat = stage(x_feat)    # ✅ 重点修改：统一使用 down_conv 的输出
            if DEBUG:
                print(f"   stage feat.shape: {feat.shape}")
            # light freq enhance
            light = self._light_freq_enhance(feat)
            if DEBUG:
                print(f"   stage light.shape: {light.shape}")
            out = feat + light
            if DEBUG:
                print(f"  --> stage out.shape: {out.shape} = residual + light")

            outputs.append(out)

        # global pooled context
        conv6 = F.adaptive_avg_pool2d(x_reduced, 1)
        if DEBUG:
            print(f"   global pool conv6.shape: {conv6.shape}")

        # fuse all stage outputs + global
        conv6 = self.conv6(conv6)
        if DEBUG:
            print(f"   fuse all stage outputs + global, conv6.shape: {conv6.shape} = residual")

        conv6 = F.interpolate(conv6, size=x.shape[2:], mode='bilinear', align_corners=False)
        if DEBUG:
            print(f"   upsample conv6.shape: {conv6.shape} = x.shape")
        outputs.append(conv6)

        if DEBUG:
            print(f"   outputs: {len(outputs)} x {outputs[0].shape}")
        
        # fuse all branches
        if self.use_weighted_fusion:
            # Per-branch learnable weighting, instead of concat
            # stack: [N_branch, B, C, H, W]
            stacked = torch.stack(outputs, dim=0)
            normed_weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1, 1)
            fused = (stacked * normed_weights).sum(0)  # [B,C,H,W]
            fused = self.fuse_linear(fused)
        else:
            fused = self.fuse(torch.cat(outputs, dim=1))

        if DEBUG:
            print(f" fused all branches --> fused.shape: {fused.shape}")

        fused = self.out_proj(fused)
        if DEBUG:
            print(f"   out_proj.shape: {fused.shape} = self.out_proj(fused)")

        if self.use_coord_gate:
            coord_feat = self.coord(fused)
            gate = torch.sigmoid(self.coord_gate_conv(fused))
            fused = fused + gate * coord_feat
        else:
            fused = self.coord(fused)

        if DEBUG:
            print(f"   coord.shape: {fused.shape} = self.coord(fused)\n")
        return fused


# -----------------------------------------------------------------------------
# PFAEv9Hybrid， 从PFAEv8Hybrid单纯并联的stage，改为每个stage的输入由前一stage的输出和全局池化的特征拼接/add而来
# -----------------------------------------------------------------------------
class PFAEv9Hybrid(nn.Module):
    """Hybrid Progressive Frequency-Aware Enhancement Block.

    Combines:
    - Lightweight multi-dilation residual freq-enhanced path (v4 lineage).
    - Optional *heavy* spectral stage (Original lineage) injected at chosen stage index.
    - Global pooled context + CoordConv positional enrichment.

    Args
    ----
    dim: int
        Input feature channels.
    in_dim: int
        Internal processing channels after first reduction.
    out_channels: int
        Output feature channels delivered to downstream neck/decoder.
    num_stages: int
        Number of dilation stages.
    min_channels: int
        Floor for reduced channel width.
    use_dct: bool
        Use DCT-like real transform instead of FFT complex.
    heavy_stage_cfg: Optional[dict]
        {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
        If enabled, inserts PFAEHeavyStage after specified stage.
    coord_with_r: bool
        Whether to include radial distance in CoordConv.
    bn2gn_ready: bool
        Placeholder flag; no behavior change, indicates block is safe for GN swap.
    """
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 4,
        min_channels: int = 16,
        use_dct: bool = False,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        use_coord_gate: bool = True,
        coord_with_r: bool = True,
        bn2gn_ready: bool = True,
        use_weighted_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_stages = num_stages
        self.min_channels = min_channels
        self.use_dct = use_dct
        self.heavy_stage_cfg = heavy_stage_cfg
        self.coord_with_r = coord_with_r

        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        self.num_heads = 4 if dim <= 64 else 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
        )

        self.cbam = CBAMBlock(in_dim)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(in_dim * 2, in_dim, 1, bias=False)

        # progressive multi-dilation conv stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, dilation=3 + 2 * i, padding=3 + 2 * i, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(True),
            )
            for i in range(num_stages)
        ])

        # learnable fusion weights for each branch
        self.use_weighted_fusion = use_weighted_fusion      # default True
        self.fusion_weights = nn.Parameter(torch.ones(self.num_stages + 2))  # x_reduced + stages + global
        self.fuse_linear = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, 1, bias=False),
            nn.GroupNorm(1, self.out_dim),
            nn.ReLU(True)
        )

        # global pooling context
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.GroupNorm(1, in_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # fuse all stage outputs + global
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim),  # ✅ 单组 GN，等效 LayerNorm2D
            nn.ReLU(True),
        )

        # final projection -> out_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

        # coord gate for coordconv (optional)
        self.use_coord_gate = use_coord_gate    # default True
        self.coord_gate_conv = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False)
        self.coord = CoordConv(self.out_dim, self.out_dim, with_r=coord_with_r)

    # ---- internal helpers --------------------------------------------------
    def _freq_pair(self):
        if self.use_dct:
            return _safe_dct2, _safe_idct2
        else:
            return _safe_fft2, _safe_ifft2

    def _light_freq_enhance(self, feat: torch.Tensor) -> torch.Tensor:
        """Lightweight frequency attention path (v4 lineage)."""
        freq_fn, ifreq_fn = self._freq_pair()
        b, c, h, w = feat.shape

        fx = freq_fn(feat.float())  # complex (FFT) or real (DCT-proxy)
        if torch.is_complex(fx):
            q = fx; k = fx; v = fx
        else:
            q = fx; k = fx; v = fx

        # reshape to heads
        q = rearrange(q, 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads) if c % self.num_heads == 0 else rearrange(F.pad(q, (0,0,0,0,0,self.num_heads - (c % self.num_heads))), 'b (hd cph) H W -> b hd cph (H W)', hd=self.num_heads)
        k = q
        v = q
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn / (attn.abs().amax(dim=-1, keepdim=True) + 1e-6)
        out_f = attn @ v
        out_f = rearrange(out_f, 'b hd cph (h w) -> b (hd cph) h w', h=h, w=w)
        out_f = torch.abs(ifreq_fn(out_f)) if torch.is_complex(out_f) else ifreq_fn(out_f)

        # CBAM-gated local spectrum
        if torch.is_complex(fx):
            gated = self.cbam(fx.real) * fx
            out_f_l = torch.abs(ifreq_fn(gated))
        else:
            gated = self.cbam(fx)
            out_f_l = ifreq_fn(gated)

        out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if DEBUG:
            print(f"\n -------------- PFAEv9Hybrid.forward() --------------")
            print(f" Args: in_dim={self.in_dim}, out_dim={self.out_dim}, num_stages={self.num_stages}")
            print(f" Args: min_channels={self.min_channels}, use_dct={self.use_dct}")
            print(f" Args: heavy_stage_cfg={self.heavy_stage_cfg}, coord_with_r={self.coord_with_r}")
            print(f" input x.shape: {x.shape}")
        
        # Stage input: strong backbone feature
        x_feat = self.down_conv(x)            # → 288通道，高表达特征，输入各 dilation stage

        if DEBUG:
            print(f" down_conv x_feat.shape: {x_feat.shape}")

        # Projection for residual + fusion path
        x_reduced = self.conv1(x_feat)        # → 144通道，用于 concat 与最终融合

        if DEBUG:
            print(f" conv1 x_reduced.shape: {x_reduced.shape} = residual")

        outputs = [x_reduced]    # init with reduced path (as base feature)

        for i, stage in enumerate(self.stages):
            if DEBUG:
                print(f"  ------- stage {i+1}, input x.shape: {x.shape}")
            if i == 0:
                feat = stage(x_feat)    # ✅ 重点修改：统一使用 down_conv 的输出
            else:
                feat = stage(x_feat+outputs[-1])             # 🔁 stage_i input = stage_{i-1}_out + global
            if DEBUG:
                print(f"   stage feat.shape: {feat.shape}")
            # light freq enhance
            light = self._light_freq_enhance(feat)    # residual + frequency-enhanced
            if DEBUG:
                print(f"   stage-{i+1}  light.shape: {light.shape}")
            feat = feat + light
            if DEBUG:
                print(f"  --> stage-{i+1}  feat.shape: {feat.shape} = feat + light")

            outputs.append(feat)

        # global pooled context
        global_ctx = F.adaptive_avg_pool2d(x_reduced, 1)
        if DEBUG:
            print(f"   global pool conv6.shape: {global_ctx.shape}")

        # fuse all stage outputs + global
        global_ctx = self.global_conv(global_ctx)
        if DEBUG:
            print(f"   fuse all stage outputs + global, conv6.shape: {global_ctx.shape} = residual")

        global_ctx = F.interpolate(global_ctx, size=x.shape[2:], mode='bilinear', align_corners=False)
        if DEBUG:
            print(f"   upsample global_ctx.shape: {global_ctx.shape} = x.shape")
        outputs.append(global_ctx)

        if DEBUG:
            print(f"   outputs: {len(outputs)} x {outputs[0].shape}")
        
        # fuse all branches
        if self.use_weighted_fusion:
            # Per-branch learnable weighting, instead of concat
            # stack: [N_branch, B, C, H, W]
            stacked = torch.stack(outputs, dim=0)
            normed_weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1, 1)
            fused = (stacked * normed_weights).sum(0)  # [B,C,H,W]
            fused = self.fuse_linear(fused)
        else:
            fused = self.fuse(torch.cat(outputs, dim=1))

        if DEBUG:
            print(f" fused all branches --> fused.shape: {fused.shape}")

        fused = self.out_proj(fused)
        if DEBUG:
            print(f"   out_proj.shape: {fused.shape} = self.out_proj(fused)")

        if self.use_coord_gate:
            coord_feat = self.coord(fused)
            gate = torch.sigmoid(self.coord_gate_conv(fused))
            fused = fused + gate * coord_feat
        else:
            fused = self.coord(fused)

        if DEBUG:
            print(f"   coord.shape: {fused.shape} = self.coord(fused)\n")
        return fused

# -----------------------------------------------------------------------------
# Parameter Counter
# -----------------------------------------------------------------------------

def pfae_count_params(module: nn.Module) -> int:
    """Return total trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# FLOP Counter (coarse, convolution + linear + attention matmul estimates)
# -----------------------------------------------------------------------------

def _conv2d_flops(cin, cout, k, h, w, groups=1):
    # MACs = h*w*cout*(cin/groups)*k*k ; FLOPs = 2*MACs (mul+add)
    macs = h * w * cout * (cin // groups) * (k * k)
    return 2 * macs


def _bn_flops(c, h, w):
    # scale + shift ~2 ops per element
    return 2 * c * h * w


def _relu_flops(c, h, w):
    return c * h * w


def _attn_flops(num_heads, ch_per_head, hw):
    # qk^T: (B*H, ch, hw)@(B*H,hw,ch) -> (B*H,ch,ch) unrealistic; we use q:(B,H,ch,HW) -> qk^T -> (B,H,HW,HW)
    # Actually in code we use q:(B,H,c,HW); cost dominated by HW*HW*c
    return num_heads * (hw * hw * ch_per_head * 2)  # mul+add


def pfae_count_flops(module: PFAEv5Hybrid, input_size: Tuple[int, int, int, int]) -> int:
    """Rough FLOP estimate for a forward pass.

    input_size: (B,C,H,W)
    NOTE: Ignores pooling / simple elementwise adds; focuses on conv + attn.
    """
    B, C, H, W = input_size
    flops = 0

    down_conv = module.down_conv[0]
    flops += _conv2d_flops(C, down_conv.out_channels, down_conv.kernel_size[0], H, W)
    flops += _bn_flops(down_conv.out_channels, H, W)
    flops += _relu_flops(down_conv.out_channels, H, W)

    conv1 = module.conv1[0]
    flops += _conv2d_flops(down_conv.out_channels, conv1.out_channels, 1, H, W)
    flops += _bn_flops(conv1.out_channels, H, W)
    flops += _relu_flops(conv1.out_channels, H, W)

    cur_c = conv1.out_channels
    for i, stage in enumerate(module.stages):
        dil = stage[0].dilation[0]
        flops += _conv2d_flops(cur_c, cur_c, 3, H, W)  # dilation doesn't change flops count
        flops += _bn_flops(cur_c, H, W)
        flops += _relu_flops(cur_c, H, W)

        # freq attention cost (light)
        hw = H * W
        ch_per_head = max(cur_c // module.num_heads, 1)
        flops += _attn_flops(module.num_heads, ch_per_head, hw)

        # heavy stage if active here
        if module.heavy_enable and i == module.heavy_stage_idx:
            ch_per_head_h = max(cur_c // module.heavy_stage.num_heads, 1)
            flops += _attn_flops(module.heavy_stage.num_heads, ch_per_head_h, hw)
            # extra project_out conv
            flops += _conv2d_flops(cur_c * 2, cur_c, 1, H, W)

    # conv6 global (1x1 on pooled → negligible; approximate upsample cost as copy)
    flops += _conv2d_flops(cur_c, cur_c, 1, 1, 1)
    flops += _bn_flops(cur_c, 1, 1)
    flops += _relu_flops(cur_c, 1, 1)

    # fuse conv
    fuse_in = (2 + module.num_stages) * cur_c
    flops += _conv2d_flops(fuse_in, cur_c, 1, H, W)
    flops += _bn_flops(cur_c, H, W)
    flops += _relu_flops(cur_c, H, W)

    # out proj
    outc = module.out_proj[0].out_channels
    flops += _conv2d_flops(cur_c, outc, 1, H, W)
    flops += _bn_flops(outc, H, W)
    flops += _relu_flops(outc, H, W)

    # coordconv (adds 2 or 3 coords)
    extra = 3 if module.coord.with_r else 2
    flops += _conv2d_flops(outc + extra, outc, 1, H, W)

    return flops * B


# -----------------------------------------------------------------------------
# BN→GN Conversion Utility
# -----------------------------------------------------------------------------

def convert_bn_to_gn(module: nn.Module, num_groups: int = 8, eps: float = 1e-5, affine: bool = True) -> nn.Module:
    """In-place recursive BatchNorm2d→GroupNorm conversion.

    Rules:
    - If C < num_groups, use GroupNorm(num_groups=1) (LayerNorm across channel).
    - Copy weight/bias if present.
    - Momentum/stat buffers are dropped; GN is stateless.
    - Returns the modified module (for chaining).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = num_groups if c >= num_groups else 1
            gn = nn.GroupNorm(g, c, eps=eps, affine=affine).to(next(child.parameters()).device)
            # load affine if available
            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight.data)
                    gn.bias.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups=num_groups, eps=eps, affine=affine)
    return module


# -----------------------------------------------------------------------------
# Comparative Counting Helper
# -----------------------------------------------------------------------------
def print_model_results(results):
    print("Model Complexity Comparison".center(60, "-"))
    print(f"{'Model':<15} | {'Params (K)':>12} | {'FLOPs (G)':>12}")
    print("-" * 60)
    print(f"{'Original PFAE':<15} | {results['original_params'] / 1e3:12.1f} | {results['original_flops'] / 1e9:12.2f}")
    print(f"{'PFAEv4':<15} | {results['v4_params'] / 1e3:12.1f} | {results['v4_flops'] / 1e9:12.2f}")
    print(f"{'PFAEv5 (Hybrid)':<15} | {results['v5_params'] / 1e3:12.1f} | {results['v5_flops'] / 1e9:12.2f}")
    print("-" * 60)

# -----------------------------------------------------------------------------
# 2. Utility blocks (CBAM, CoordConv, Wavelet stub, MS‑GlobalEncoder)
# -----------------------------------------------------------------------------

class ChannelGate(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(c // r, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, c, 1))
    def forward(self, x):
        w = self.mlp(F.adaptive_avg_pool2d(x, 1))
        return torch.sigmoid(w) * x


class WaveletTransformBlock(nn.Module):
    """Haar‑wavelet decomposition stub (keeps LL+LH+HL+HH)."""
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c*4, c, 1)  # fuse
    def forward(self, x):
        # naive 2×2 avg / diff – placeholder for true Haar
        LL = F.avg_pool2d(x, 2)
        LH = F.avg_pool2d(x[:,:,:-1:2,:],2) - LL
        HL = F.avg_pool2d(x[:,:,:,:-1:2],2) - LL
        HH = x[:, :, ::2, ::2] - LL
        up = F.interpolate(torch.cat([LL,LH,HL,HH],1), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(up)

class MultiScaleGlobalEncoder(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_c, out_c,1,bias=False) for _ in range(3)])
    def forward(self, x):
        h,w = x.shape[-2:]
        feats = [self.convs[0](x)]
        for i,ratio in enumerate([2,4]):
            f = F.avg_pool2d(x, ratio)
            f = F.interpolate(f, size=(h,w), mode='bilinear', align_corners=False)
            feats.append(self.convs[i+1](f))
        return torch.cat(feats,1)

class MultiScaleGlobalEncoderV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        inter_channels = 64
        # inter_channels = out_channels // 3  # 每个尺度通道数相等分配
        self.inter_channels = inter_channels

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(inter_channels*3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if DEBUG:
            print(f"\n ---- MultiScaleGlobalEncoderV3, forward ---- ")
            print(f" Args: in_channels = {self.in_channels}, out_channels = {self.out_channels}, inter_channels = {self.inter_channels}")
            print(f" input x.shape: {x.shape}")

        h, w = x.shape[2:]

        x1 = self.scale1(x)                          # 原图分支
        x2 = self.scale2(F.avg_pool2d(x, 2))         # 1/2 分支
        x3 = self.scale3(F.avg_pool2d(x, 4))         # 1/4 分支

        if DEBUG:
            print(f" level-1, x1.shape: {x1.shape}")
            print(f" level-2, x2.shape: {x2.shape}")
            print(f" level-3, x3.shape: {x3.shape}")

        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=False)

        if DEBUG:
            print(f" level-2, upsample x2.shape: {x2.shape}")
            print(f" level-3, upsample x3.shape: {x3.shape}")

        x_all = torch.cat([x1, x2, x3], dim=1)       # 通道拼接
        if DEBUG:
            print(f" x_all.shape: {x_all.shape}")
        x_all = self.fuse(x_all)                      # 输出统一通道数
        if DEBUG:
            print(f" output x_all.shape: {x_all.shape}\n")
        return x_all

# -------------------- Haar Wavelet Transform --------------------
class HaarWaveletPooling2D(nn.Module):
    def __init__(self):
        super().__init__()
        a = 1 / 2 ** 0.5
        self.register_buffer('haar_kernels', torch.tensor([
            [[a, a], [a, a]],     # LL
            [[a, a], [-a, -a]],   # LH
            [[a, -a], [a, -a]],   # HL
            [[a, -a], [-a, a]]    # HH
        ]).unsqueeze(1))  # shape = [4, 1, 2, 2]

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("Input H and W must be divisible by 2")

        device = x.device
        filters = self.haar_kernels.to(device)  # [4, 1, 2, 2]
        outputs = []

        for i in range(4):  # 对 LL, LH, HL, HH 四个分支分别处理
            kernel = filters[i].repeat(C, 1, 1, 1)  # [C, 1, 2, 2]
            out = F.conv2d(x, kernel, stride=2, padding=0, groups=C)  # [B, C, H/2, W/2]
            outputs.append(out)

        return outputs  # ll, lh, hl, hh

class WaveletTransformBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pool = HaarWaveletPooling2D()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if DEBUG:
            print(f"\n ---- WaveletTransformBlockV3, forward ---- ")
            print(f" Args: in_channels = {self.in_channels}, out_channels = {self.out_channels}")
            print(f" input x.shape: {x.shape}")
        ll, lh, hl, hh = self.pool(x)
        if DEBUG:
            print(f" ll.shape: {ll.shape}, lh.shape: {lh.shape}, hl.shape: {hl.shape}, hh.shape: {hh.shape}")

        feat = torch.cat([ll, lh, hl, hh], dim=1)
        if DEBUG:
            print(f" feat.shape: {feat.shape}")
        feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear')
        if DEBUG:
            print(f" feat.shape: {feat.shape}")
        feat = self.conv(feat)
        if DEBUG:
            print(f" feat.shape: {feat.shape}\n")
        return feat

# -------------------- PositionEnhancedFusion --------------------
class PositionEnhancedFusion(nn.Module):
    def __init__(self, in_channels, use_coord=True):
        super().__init__()
        self.use_coord = use_coord
        self.coord_conv = CoordConv(in_channels, in_channels) if use_coord else None
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.use_coord:
            pos_feat = self.coord_conv(x)
            x = x + pos_feat  # 残差式位置增强
        return self.fuse(x)

# -----------------------------------------------------------------------------
# 3. PFAEv2 (use_dct switch + gated attention)
# -----------------------------------------------------------------------------

class PFAEv2(nn.Module):
    def __init__(self, dim:int, in_dim:int, out_dim:int, 
                 num_stages:int=3, min_channels:int=16, 
                 use_dct:bool=False, bn2gn_ready: bool = True, **kwargs):
        super().__init__()
        self.use_dct = use_dct
        self.down = nn.Conv2d(dim, in_dim,3,1,1,bias=False)
        self.red  = nn.Conv2d(in_dim, max(in_dim//2,min_channels),1,bias=False)
        self.gate = ChannelGate(max(in_dim//2,min_channels))
        self.stages = nn.ModuleList([
            nn.Conv2d(max(in_dim//2,min_channels), max(in_dim//2,min_channels),3,1,1,bias=False) for _ in range(num_stages)
        ])
        self.proj = nn.Conv2d(max(in_dim//2,min_channels), out_dim,1,bias=False)
    def _freq_pair(self):
        return (_safe_dct2, _safe_idct2) if self.use_dct else (_safe_fft2, _safe_ifft2)
    def forward(self,x):
        x = self.red(self.down(x))
        freq, ifreq = self._freq_pair()
        f = freq(x)
        if torch.is_complex(f):
            amp = f.abs()
        else:
            amp = f
        f_enh = self.gate(amp)
        x = x + ifreq(f_enh)
        for conv in self.stages:
            x = x + conv(x)
        return self.proj(x)

class PFAEv4(nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        out_dim: int,
        num_stages: int = 3,
        min_c: int = 16,
        use_dct: bool = False,
        bn2gn_ready: bool = True,
        **kwargs,
    ):
        """
        PFAEv4:
          dim      - 输入通道数
          in_dim   - 降采样后通道数
          out_dim  - 输出通道数
          num_stages - 频域阶段数
          min_c    - down_dim 最小值
          use_dct  - True 切换到 DCT 模式，否则 FFT
          bn2gn_ready - 保留所有 GN 层（默认 True）
        """
        super().__init__()
        # 1. 下采样
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_dim),
            nn.ReLU(True),
        )
        # 2. reduction → down_dim
        down_dim = max(in_dim // 2, min_c)
        self.down_dim = down_dim
        self.num_stages = num_stages
        self.use_dct = use_dct
        # attention head
        self.num_heads = 4 if down_dim <= 64 else 2
        # 1×1 conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, down_dim),
            nn.ReLU(True),
        )
        # CBAM 通道+空间 注意力
        self.cbam = CBAMBlock(down_dim)
        # 频域温度参数
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # 输出投影：双路径 concat → down_dim
        self.project_out = nn.Conv2d(down_dim * 2, down_dim, kernel_size=1, bias=False)

        # 3. dilated stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(down_dim, down_dim,
                          kernel_size=3,
                          dilation=3 + 2*i,
                          padding=3 + 2*i,
                          bias=False),
                nn.GroupNorm(1, down_dim),
                nn.ReLU(True),
            ) for i in range(num_stages)
        ])

        # 4. 全局 context fusion
        self.conv6 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, down_dim),
            nn.ReLU(True),
        )

        # 5. 多阶段融合
        self.fuse = nn.Sequential(
            nn.Conv2d((2 + num_stages) * down_dim, down_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, down_dim),
            nn.ReLU(True),
        )
        # 最终输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(down_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.ReLU(True),
        )
        # 坐标卷积位置增强
        self.coord = CoordConv(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 下采样
        x = self.down_conv(x)
        # reduction
        x_reduced = self.conv1(x)
        outputs = [x_reduced]
        residual = x_reduced

        # 逐阶段频域注意力
        for stage in self.stages:
            feat = stage(residual)                # [B, C, H, W]
            b, c, h, w = feat.shape

            # 选择频域变换
            freq_fn = _safe_dct2 if self.use_dct else _safe_fft2
            ifreq_fn = _safe_idct2 if self.use_dct else _safe_ifft2

            # QKV 频域
            Ff = feat.float()  # cast to float32 for stability
            q = freq_fn(Ff); k = freq_fn(Ff); v = freq_fn(Ff)
            # reshape为多头
            q, k, v = [
                rearrange(t, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                for t in (q, k, v)
            ]
            # 归一化 + 注意力矩阵
            q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2,-1)) * self.temperature
            attn = custom_complex_normalization(attn, dim=-1)
            # 频域交互
            out_f = torch.abs(ifreq_fn(attn @ v))
            out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w',
                              head=self.num_heads, h=h, w=w)

            # 本地频率加权分支
            gated = self.cbam(feat)
            out_f_l = torch.abs(ifreq_fn(gated * freq_fn(Ff)))

            # 融合两路
            out = self.project_out(torch.cat([out_f, out_f_l], dim=1))
            residual = x_reduced + out + feat
            outputs.append(out + feat)

        # 全局 context
        conv6 = self.conv6(F.adaptive_avg_pool2d(x_reduced, 1))
        conv6 = F.interpolate(conv6, size=(h,w), mode='bilinear', align_corners=False)
        outputs.append(conv6)

        # fuse & final proj
        fused = self.fuse(torch.cat(outputs, dim=1))
        fused = self.out_proj(fused)

        # 坐标增强
        return self.coord(fused)

# -----------------------------------------------------------------------------
# 4. Fusion necks
# -----------------------------------------------------------------------------

class PFAEGlobalFusionNeckV2(nn.Module):
    def __init__(
        self,
        img_c: int,
        sam_c: int,
        uni_c: int,
        out_c: int = 256,
        pfae_cls: Type[nn.Module] = PFAEv2,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        use_dct: bool = True,
    ):
        super().__init__()
        # project channels
        self.img_proj = nn.Conv2d(img_c, out_c//4, 1)
        self.sam_proj = nn.Conv2d(sam_c, out_c//4, 1)
        self.uni_proj = nn.Conv2d(uni_c, out_c//4, 1)
        # multi-scale global context + wavelet
        self.ms_enc = MultiScaleGlobalEncoder(out_c//4, out_c//4)
        self.wave  = WaveletTransformBlock(out_c//4)
        # prepare PFAE injection
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=out_c*2, in_dim=out_c*2, out_dim=out_c, num_stages=3, min_c=16, use_dct=use_dct)
        self.pfae = pfae_cls(**pfae_kwargs)
        # position refine
        self.coord = CoordConv(out_c, out_c, with_r=True)

    def forward(self, img: torch.Tensor, img_emb: torch.Tensor, ex_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.img_proj(img), self.sam_proj(img_emb), self.uni_proj(ex_emb)], dim=1)
        x = torch.cat([x, self.ms_enc(x), self.wave(x)], dim=1)
        x = self.pfae(x)
        return self.coord(x)

class PFAEGlobalFusionNeckV3(nn.Module):
    def __init__(self,
                 sam_channels=1024,
                 uni_channels=768,
                 img_channels=3,
                 fusion_channels=256,
                 use_position=True):
        super().__init__()
        self.sam_proj = nn.Conv2d(sam_channels, fusion_channels, 1)
        self.uni_proj = nn.Conv2d(uni_channels, fusion_channels, 1)
        self.global_multi_scale_encoder = MultiScaleGlobalEncoderV3(img_channels, fusion_channels)
        self.wavelet_block = WaveletTransformBlockV3(img_channels, fusion_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels * 3, fusion_channels, 1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )
        self.position_refine = PositionEnhancedFusion(fusion_channels, use_coord=use_position)

    def forward(self, img, img_emb, ex_embed):
        sam_feat = self.sam_proj(img_emb)
        uni_feat = self.uni_proj(ex_embed)
        fusion_feat = sam_feat + uni_feat

        global_avg = self.global_multi_scale_encoder(img)
        global_wave = self.wavelet_block(img)

        combined = torch.cat([fusion_feat, global_avg, global_wave], dim=1)
        fused = self.fusion_conv(combined)
        refined = self.position_refine(fused)
        return refined  # [B, fusion_channels, H, W]

class PFAEGlobalFusionNeckV4(nn.Module):
    def __init__(self,
        img_c, sam_c, uni_c, fusion_c=256,
        use_dct=True, use_coord=True):
        super().__init__()
        # 1. 通道统一投影
        self.img_proj = nn.Conv2d(img_c, fusion_c//3, 1)
        self.sam_proj = nn.Conv2d(sam_c, fusion_c//3, 1)
        self.uni_proj = nn.Conv2d(uni_c, fusion_c//3, 1)
        # 2. 多尺度上下文（v3 架构）
        self.ms_enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fusion_c//3, fusion_c//3,3,padding=1),
                          nn.GroupNorm(8, fusion_c//3), nn.ReLU(inplace=True))
            for _ in range(3)
        ])
        # 3. 小波子带（v3 真 Haar）
        self.wave = WaveletTransformBlockV3(img_channels=img_c, out_channels=fusion_c//3)
        # 4. PFAEv4 注入
        self.pfae = PFAEv4(
            dim=fusion_c, in_dim=fusion_c,
            out_channels=fusion_c, num_stages=3,
            min_channels=16, use_dct=use_dct
        )
        # 5. 位置精炼
        self.pos = PositionEnhancedFusion(fusion_c, use_coord=use_coord)

    def forward(self, img, sam_feat, uni_feat):
        # 拼接投影
        x0 = torch.cat([
            self.img_proj(img),
            self.sam_proj(sam_feat),
            self.uni_proj(uni_feat)
        ], dim=1)  # ➔ [B, fusion_c, H, W]

        # 多尺度上下文
        h,w = x0.shape[-2:]
        mss = []
        for i,enc in enumerate(self.ms_enc):
            ratio = 1 if i==0 else (2 if i==1 else 4)
            fi = x0 if ratio==1 else F.avg_pool2d(x0, ratio)
            fi = enc(fi)
            fi = F.interpolate(fi, (h,w), 'bilinear', False)
            mss.append(fi)
        ms = torch.cat(mss,1)  # [B, fusion_c, H, W]

        # 小波子带
        wv = self.wave(img)   # [B, fusion_c//3, H, W]

        # PFAE 注入 + 位置精炼
        x = torch.cat([x0, ms, wv],1)      # [B, fusion_c* (1+1+1/3)=…]
        x = self.pfae(x)                   # [B, fusion_c, H, W]
        return self.pos(x)                 # [B, fusion_c, H, W]

class PFAEGlobalFusionNeckV5(nn.Module):
    def __init__(
        self,
        img_c: int,
        sam_c: int,
        uni_c: int,
        fusion_c: int = 512,
        # ——— 可插拔 PFAE 变体 ———
        pfae_cls: Type[nn.Module] = PFAEv4,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
                # heavy_stage_cfg: Optional[dict]
                # {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
                # If enabled, inserts PFAEHeavyStage after specified stage.
        # -- sam & uni 的比例控制
        proj_ratios: tuple[float, float] = (0.5, 0.5),  # sam:uni
        ms_out_c: int = 16,
        wave_out_c: int = 16,
        # ——— 频域 / 位置增强开关 ———
        use_dct: bool = True,
        use_coord: bool = True,
        # ——— 可选：多尺度 / 小波 客制化 ———
        ms_enc_cls: Type[nn.Module] = MultiScaleGlobalEncoderV3,
        wavelet_cls: Type[nn.Module] = WaveletTransformBlockV3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert abs(sum(proj_ratios) - 1.0) < 1e-6, "proj_ratios must sum to 1"

        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios
        self.ms_out_c = ms_out_c
        self.wave_out_c = wave_out_c

        self.use_dct = use_dct
        self.use_coord = use_coord
        self.dropout_rate = dropout_rate

        # 1. Pixel-level encoder, 多尺度 & 小波 模块, (in_channels=3, out_channels=256)
        self.ms_enc = ms_enc_cls(img_c, ms_out_c)     # [B, ms_out_c, H, W]
        self.wave  = wavelet_cls(img_c, wave_out_c)     # [B, wave_out_c, H, W]

        # 2. 通道划分（确保总和为 fusion_c）
        s_ch = int(fusion_c * proj_ratios[0])
        u_ch = int(fusion_c * proj_ratios[1])
        self.ch = (s_ch, u_ch, ms_out_c, wave_out_c)

        # 2. 单独投影
        self.sam_proj = nn.Conv2d(sam_c, s_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, u_ch, 1, bias=False)
        self.ms_proj = nn.Conv2d(ms_out_c, ms_out_c, 1, bias=False)
        self.wave_proj = nn.Conv2d(wave_out_c, wave_out_c, 1, bias=False)

        # 在 concat 之后再映射回 fusion_c
        x0_dim = s_ch + u_ch + ms_out_c + wave_out_c
        self.fusion_proj = nn.Conv2d(x0_dim, fusion_c, 1, bias=False)

        pfae_dim = fusion_c + ms_out_c + wave_out_c

        # 3. PFAE 变体，上下文建模模块
        if pfae_kwargs is None:
            pfae_kwargs = dict(
                dim=pfae_dim,
                in_dim=pfae_dim,
                out_dim=fusion_c,
                num_stages=3,
                min_channels=16,
                use_dct=use_dct,
                heavy_stage_cfg=heavy_stage_cfg,
            )
        else:
            pfae_kwargs['heavy_stage_cfg'] = heavy_stage_cfg
        self.pfae = pfae_cls(**pfae_kwargs)

        # 5. 位置增强
        self.pos = PositionEnhancedFusion(fusion_c, use_coord=use_coord)

    def forward(self, 
                img: torch.Tensor, 
                sam_feat: torch.Tensor, 
                uni_feat: torch.Tensor
               ) -> torch.Tensor:
        """
        Args:
          img:      [B, 3, H, W]
          sam_feat: [B, sam_c, H, W]
          uni_feat: [B, uni_c, H, W]
        Returns:
          [B, fusion_c, H, W]
        """
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckV5.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), ms_out_c: {self.ms_out_c}, wave_out_c: {self.wave_out_c}")
            print(f" Args: (s_ch, u_ch, m_ch, w_ch) = {self.ch}")
            print(f" ---- input img.shape: {img.shape}")
            print(f" ---- input sam_feat.shape: {sam_feat.shape}")
            print(f" ---- input uni_feat.shape: {uni_feat.shape}")

        B, _, Hf, Wf = sam_feat.shape

        # 1. 提取低阶特征（全分辨率）, 多尺度 & 小波
        ms_full = self.ms_enc(img)    # [B, ms_out_c, H_img, W_img]
        wv_full = self.wave(img)      # [B, wave_out_c, H_img, W_img]

        if DEBUG:
            print(f" encoder img -> ms_full.shape: {ms_full.shape}")
            print(f" encoder img -> wv_full.shape: {wv_full.shape}")

        # 2. 下采样到与 sam/uni 相同空间分辨率
        ms = F.interpolate(ms_full, size=(Hf, Wf), mode='bilinear', align_corners=False)
        wv = F.interpolate(wv_full, size=(Hf, Wf), mode='bilinear', align_corners=False)

        if DEBUG:
            print(f" F.interpolate(ms_full, size=(Hf, Wf)) -> ms.shape: {ms.shape}")
            print(f" F.interpolate(wv_full, size=(Hf, Wf)) -> wv.shape: {wv.shape}")

        # 3. 通道投影, 各自 1×1 投影 
        p_sam = self.sam_proj(sam_feat)   # [B, s_ch, Hf, Wf]
        p_uni = self.uni_proj(uni_feat)   # [B, u_ch, Hf, Wf]
        p_ms  = self.ms_proj(ms)          # [B, ms_out_c, Hf, Wf]
        p_wv  = self.wave_proj(wv)        # [B, wave_out_c, Hf, Wf]

        if DEBUG:
            print(f" proj_dropout ---> p_sam.shape: {p_sam.shape}")
            print(f" proj_dropout ---> p_uni.shape: {p_uni.shape}")
            print(f" proj_dropout ---> p_ms.shape: {p_ms.shape}")
            print(f" proj_dropout ---> p_wv.shape: {p_wv.shape}")
        
        # ——— 初步融合 ———
        # 4. 初步融合
        x0 = torch.cat([p_sam, p_uni, p_ms, p_wv], dim=1)  # [B, x0_dim, Hf, Wf]
        if DEBUG:
            print(" torch.cat([p_sam, p_uni, p_ms, p_wv]) ---> x0.shape: ", x0.shape)
        x0 = self.fusion_proj(x0)
        if DEBUG:
            print(" fusion_proj ---> x0.shape: ", x0.shape)

        # 5. PFAE 上下文融合
        # 使用下采样后的 ms, wv 保持空间一致
        # 拼接 x0 + 原 ms + 原 wv
        x_cat  = torch.cat([x0, ms, wv], dim=1)               # [B, fusion_c*3, H, W]
        if DEBUG:
            print(" torch.cat([x0, ms, wv]) ---> x_cat.shape: ", x_cat.shape)
        x_pfae = self.pfae(x_cat)                             # [B, fusion_c, H, W]
        if DEBUG:
            print(" pfae ---> x_pfae.shape: ", x_pfae.shape)

        # 6. 残差 + 位置精炼 ———
        x_out = x0 + x_pfae
        if DEBUG:
            print(" residual ---> x_out.shape: ", x_out.shape)
        x_out = self.pos(x_out)
        if DEBUG:
            print(f" pos ---> x_out.shape: {x_out.shape}\n")
        return x_out

class PFAEGlobalFusionNeckV6(nn.Module):
    """
    Hi-Lo Pixel Fusion + PFAE  (sam, uni, ms, wave)
    - Lo-Res:   16×16   → 用于与 sam/uni 融合
    - Hi-Res: 128×128   → 细节残差注入
    """
    def __init__(
        self,
        img_c: int,
        sam_c: int,
        uni_c: int,
        fusion_c: int = 256,
        proj_ratios: tuple[float, float] = (0.5, 0.5),   # sam:uni
        ms_hi_c: int = 8,   wave_hi_c: int = 8,          # hi-res 通道
        ms_lo_c: int = 16,  wave_lo_c: int = 16,         # lo-res 通道
        ms_enc_cls: type[nn.Module] = MultiScaleGlobalEncoderV3,
        wavelet_cls: type[nn.Module] = WaveletTransformBlockV3,
        pfae_cls: type[nn.Module] = PFAEv4,
        pfae_kwargs: dict | None = None,
        use_coord: bool = True,
        use_dct: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert abs(sum(proj_ratios)-1) < 1e-6
        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios
        self.ms_hi_c = ms_hi_c
        self.wave_hi_c = wave_hi_c
        self.ms_lo_c = ms_lo_c
        self.wave_lo_c = wave_lo_c
        self.use_coord = use_coord
        self.use_dct = use_dct

        # -------------------------------- Hi-Res 分支 -------------------------------- #
        self.ms_hi  = nn.Sequential(
            ms_enc_cls(img_c, ms_hi_c),              # [B, ms_hi_c, H, W]
            nn.Conv2d(ms_hi_c, ms_hi_c, 1, bias=False)
        )
        self.wv_hi  = nn.Sequential(
            wavelet_cls(img_c, wave_hi_c),
            nn.Conv2d(wave_hi_c, wave_hi_c, 1, bias=False)
        )

        # -------------------------------- Lo-Res 分支 -------------------------------- #
        self.ms_lo_c, self.wv_lo_c = ms_lo_c, wave_lo_c
        self.ms_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.wv_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.ms_lo_proj  = nn.Conv2d(ms_lo_c, ms_lo_c, 1, bias=False)
        self.wv_lo_proj  = nn.Conv2d(wave_lo_c, wave_lo_c, 1, bias=False)

        # sam / uni 通道分配
        s_ch = int(fusion_c * proj_ratios[0])
        u_ch = fusion_c - s_ch - ms_lo_c - wave_lo_c
        assert u_ch > 0

        self.sam_proj = nn.Conv2d(sam_c, s_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, u_ch, 1, bias=False)

        # 融合到 fusion_c
        self.fusion_proj = nn.Conv2d(s_ch+u_ch+ms_lo_c+wave_lo_c, fusion_c, 1, bias=False)

        # PFAE
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=fusion_c*3, in_dim=fusion_c*3,
                               out_dim=fusion_c, num_stages=3, min_channels=16, use_dct=use_dct)
        self.pfae = pfae_cls(**pfae_kwargs)

        # 位置增强 + 门控
        self.pos = PositionEnhancedFusion(fusion_c, use_coord=use_coord)
        self.gate = nn.Sequential(
            nn.Conv2d(ms_hi_c + wave_hi_c, fusion_c, 1),
            nn.Sigmoid()
        )

    # ------------------------------------------------------------------------- #
    def forward(self, img, sam_feat, uni_feat):
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckV6.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), ms_hi_c: {self.ms_hi_c}, wave_hi_c: {self.wave_hi_c}")
            print(f" Args: ms_lo_c: {self.ms_lo_c}, wave_lo_c: {self.wave_lo_c}")
            print(f" ---- input img.shape: {img.shape}")
            print(f" ---- input sam_feat.shape: {sam_feat.shape}")
            print(f" ---- input uni_feat.shape: {uni_feat.shape}")

        B, _, H_img, W_img = img.shape
        _, _, Hf, Wf = sam_feat.shape     # 16×16

        # Hi-Res
        ms_hi = self.ms_hi(img)                          # [B, ms_hi_c, 128,128]
        wv_hi = self.wv_hi(img)                          # [B, wave_hi_c,128,128]
        if DEBUG:
            print(f" (Hi-Res) encoder img -> ms_hi.shape: {ms_hi.shape}")
            print(f" (Hi-Res) encoder img -> wv_hi.shape: {wv_hi.shape}")

        # Lo-Res
        ms_lo = self.ms_pool(ms_hi)                      # down → 16×16
        wv_lo = self.wv_pool(wv_hi)
        if DEBUG:
            print(f" (Lo-Res) F.adaptive_avg_pool2d(ms_hi) -> ms_lo.shape: {ms_lo.shape}")
            print(f" (Lo-Res) F.adaptive_avg_pool2d(wv_hi) -> wv_lo.shape: {wv_lo.shape}")

        print(f"\n --- (Lo-Res) self.ms_lo_proj = {self.ms_lo_proj}")
        print(f"\n ---  (Lo-Res) self.wv_lo_proj = {self.wv_lo_proj}")

        ms_lo = self.ms_lo_proj(ms_lo)                   # [B, ms_lo_c, 16,16]
        wv_lo = self.wv_lo_proj(wv_lo)
        if DEBUG:
            print(f" (Lo-Res) ms_lo_proj -> ms_lo.shape: {ms_lo.shape}")
            print(f" (Lo-Res) wv_lo_proj -> wv_lo.shape: {wv_lo.shape}")

        # sam / uni 投影
        p_sam = self.sam_proj(sam_feat)                  # [B, s_ch,16,16]
        p_uni = self.uni_proj(uni_feat)                  # [B, u_ch,16,16]
        if DEBUG:
            print(f" (Lo-Res) sam_proj -> p_sam.shape: {p_sam.shape}")
            print(f" (Lo-Res) uni_proj -> p_uni.shape: {p_uni.shape}")

        # 拼接 → fusion_c
        x0 = torch.cat([p_sam, p_uni, ms_lo, wv_lo], dim=1)
        if DEBUG:
            print(f" (Lo-Res) torch.cat([p_sam, p_uni, ms_lo, wv_lo]) -> x0.shape: {x0.shape}")

        # fusion_proj
        x0 = self.fusion_proj(x0)                        # [B, fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) torch.cat([p_sam, p_uni, ms_lo, wv_lo]) -> x0.shape: {x0.shape}")

        # PFAE (x0 + Lo-Res)
        x_cat = torch.cat([x0, ms_lo, wv_lo], dim=1)     # [B, fusion_c*3,16,16]
        if DEBUG:
            print(f" (Lo-Res) torch.cat([x0, ms_lo, wv_lo]) -> x_cat.shape: {x_cat.shape}")
        x_pfae = self.pfae(x_cat)                        # [B, fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) pfae -> x_pfae.shape: {x_pfae.shape}")

        # Hi-Res 残差注入
        hi_feat = torch.cat([ms_hi, wv_hi], dim=1)       # [B, *,128,128]
        if DEBUG:
            print(f" (Hi-Res) torch.cat([ms_hi, wv_hi]) -> hi_feat.shape: {hi_feat.shape}")
        hi_feat = F.interpolate(hi_feat, size=(Hf, Wf), mode='bilinear')
        if DEBUG:
            print(f" (Hi-Res) F.interpolate(hi_feat, size=(Hf, Wf), mode='bilinear') -> hi_feat.shape: {hi_feat.shape}")
        gate = self.gate(hi_feat)                        # [B, fusion_c,16,16]
        if DEBUG:
            print(f" (Hi-Res) gate -> gate.shape: {gate.shape}")

        x_out = self.pos(x0 + x_pfae + gate * hi_feat)   # 融合+位置精炼
        if DEBUG:
            print(f" (Hi-Res) pos -> x_out.shape: {x_out.shape}\n")
        return x_out

class PFAEGlobalFusionNeckV7(nn.Module):
    """
    Hi-Lo Fusion V7:
      • Hi-Res: 原 img → ms_hi / wv_hi (128×128)
      • Lo-Res: img 下采样→ms_lo_enc / wv_lo_enc (16×16)
    """
    def __init__(
        self,
        img_c: int,
        sam_c: int,
        uni_c: int,
        fusion_c: int = 256,
        proj_ratios: tuple[float,float] = (0.5,0.5),
        ms_hi_c: int = 8,  wave_hi_c: int = 8,
        ms_lo_c: int = 16, wave_lo_c: int = 16,
        ms_enc_cls: type[nn.Module] = MultiScaleGlobalEncoderV3,
        wavelet_cls: type[nn.Module] = WaveletTransformBlockV3,
        pfae_cls: type[nn.Module] = PFAEv4,
        pfae_kwargs: dict | None = None,
        use_coord: bool = True,
        use_dct: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert abs(sum(proj_ratios)-1) < 1e-6
        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios
        self.use_coord = use_coord
        self.use_dct = use_dct

        # 分支通道配置
        self.ms_hi_c, self.wv_hi_c = ms_hi_c, wave_hi_c
        self.ms_lo_c, self.wv_lo_c = ms_lo_c, wave_lo_c

        # — Hi-Res 分支（全分辨率 MS & Wave） —
        self.ms_hi = nn.Sequential(
            ms_enc_cls(img_c, ms_hi_c),
            nn.Conv2d(ms_hi_c, ms_hi_c, 1, bias=False)
        )
        self.wv_hi = nn.Sequential(
            wavelet_cls(img_c, wave_hi_c),
            nn.Conv2d(wave_hi_c, wave_hi_c, 1, bias=False)
        )

        # — Lo-Res 分支：先下采样再编码 —
        self.ms_lo_enc = ms_enc_cls(img_c, ms_lo_c)
        self.wv_lo_enc = wavelet_cls(img_c, wave_lo_c)

        # SAM / UNI 投影
        s_ch = int(fusion_c * proj_ratios[0])
        u_ch = fusion_c - s_ch - ms_lo_c - wave_lo_c
        assert u_ch > 0, "proj_ratios 不合法"

        self.sam_proj = nn.Conv2d(sam_c, s_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, u_ch, 1, bias=False)

        # 拼接后再映射到 fusion_c
        self.fusion_proj = nn.Conv2d(
            s_ch + u_ch + ms_lo_c + wave_lo_c,
            fusion_c, 1, bias=False
        )

        # PFAE 上下文融合
        if pfae_kwargs is None:
            pfae_kwargs = dict(
                dim=fusion_c*3, in_dim=fusion_c*3,
                out_dim=fusion_c, num_stages=3,
                min_channels=16, use_dct=use_dct
            )
        self.pfae = pfae_cls(**pfae_kwargs)

        # 位置精炼 + hi_feat 门控
        self.pos  = PositionEnhancedFusion(fusion_c, use_coord=use_coord)
        self.gate = nn.Sequential(
            nn.Conv2d(ms_hi_c+wave_hi_c, fusion_c, 1),
            nn.Sigmoid()
        )

    def forward(self, img, sam_feat, uni_feat):
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckV7.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), ms_hi_c: {self.ms_hi_c}, wave_hi_c: {self.wave_hi_c}")
            print(f" Args: ms_lo_c: {self.ms_lo_c}, wave_lo_c: {self.wave_lo_c}")
            print(f" ---- input img.shape: {img.shape}")
            print(f" ---- input sam_feat.shape: {sam_feat.shape}")
            print(f" ---- input uni_feat.shape: {uni_feat.shape}")

        B,_,H_img,W_img = img.shape
        _,_,Hf,Wf = sam_feat.shape   # e.g. 16×16

        # — Hi-Res 分支 —
        ms_hi = self.ms_hi(img)       # [B, ms_hi_c,128,128]
        wv_hi = self.wv_hi(img)       # [B, wv_hi_c,128,128]
        if DEBUG:
            print(f" (Hi-Res) encoder img -> ms_hi.shape: {ms_hi.shape}")
            print(f" (Hi-Res) encoder img -> wv_hi.shape: {wv_hi.shape}")

        # — Lo-Res 分支：先下采样图像，再编码 —
        img_lo = F.adaptive_avg_pool2d(img, (Hf,Wf))  # [B,3,16,16]
        if DEBUG:
            print(f" (Lo-Res) F.adaptive_avg_pool2d(img) -> img_lo.shape: {img_lo.shape}")

        ms_lo = self.ms_lo_enc(img_lo)                # [B,ms_lo_c,16,16]
        wv_lo = self.wv_lo_enc(img_lo)                # [B,wv_lo_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) ms_lo_enc -> ms_lo.shape: {ms_lo.shape}")
            print(f" (Lo-Res) wv_lo_enc -> wv_lo.shape: {wv_lo.shape}")


        # — SAM/UNI 投影 &
        p_sam = self.sam_proj(sam_feat)               # [B,s_ch,16,16]
        p_uni = self.uni_proj(uni_feat)               # [B,u_ch,16,16]
        if DEBUG:
            print(f" (Lo-Res) sam_proj -> p_sam.shape: {p_sam.shape}")
            print(f" (Lo-Res) uni_proj -> p_uni.shape: {p_uni.shape}")

        # — 初步融合 —
        x0 = torch.cat([p_sam, p_uni, ms_lo, wv_lo], dim=1)
        if DEBUG:
            print(f" (Lo-Res) torch.cat([p_sam, p_uni, ms_lo, wv_lo]) -> x0.shape: {x0.shape}")
        x0 = self.fusion_proj(x0)                     # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) torch.cat([p_sam, p_uni, ms_lo, wv_lo]) -> x0.shape: {x0.shape}")

        # — PFAE 上下文增强 —
        x_cat  = torch.cat([x0, ms_lo, wv_lo], dim=1)  # [B,fusion_c*3,16,16]
        if DEBUG:
            print(f" (Lo-Res) torch.cat([x0, ms_lo, wv_lo]) -> x_cat.shape: {x_cat.shape}")
        x_pfae = self.pfae(x_cat)                      # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) pfae(x_cat) -> x_pfae.shape: {x_pfae.shape}")

        # — Hi-Res 细节注入 —
        hi_feat = torch.cat([ms_hi, wv_hi], dim=1)     # [B, ms_hi_c+wv_hi_c,128,128]
        if DEBUG:
            print(f" (Hi-Res) torch.cat([ms_hi, wv_hi]) -> hi_feat.shape: {hi_feat.shape}")
            
        hi_feat = F.interpolate(hi_feat, (Hf,Wf),
                                mode='bilinear',
                                align_corners=False)
        if DEBUG:
            print(f" (Hi-Res) F.interpolate(hi_feat) -> hi_feat.shape: {hi_feat.shape}")
        gate    = self.gate(hi_feat)                   # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Hi-Res) gate -> gate.shape: {gate.shape}")

        # — 融合 + 位置精炼 + 门控 —
        x_out = self.pos(x0 + x_pfae + gate * hi_feat)  # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Hi-Res) pos -> x_out.shape: {x_out.shape}\n")
        return x_out

class PFAEGlobalFusionNeckForNucleiV2(nn.Module):
    """
    Hi-Res (128×128) + Mid-Res (32×32) Pixel Encoder  →  SAM / UNI / MS / WAVE 融合

    Pipeline
    --------
    img ─► MS_hi / WV_hi ─┬─► 32×32 池化 → sam/uni 拼接 → 1×1 (fusion_proj) ─┐
                           └─► 128×128 gate (细节)                       │
    sam_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┤
    uni_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┘
    """
    def __init__(
        self,
        img_c      : int,
        sam_c      : int,
        uni_c      : int,
        fusion_c   : int = 256,
        proj_ratios: Tuple[float,float] = (0.4, 0.4),  # sam : uni
        hi_c       : int = 8,   # channels of MS/WAVE in 128²
        mid_c      : int = 16,  # channels after 32² pooling
        mid2_enable: bool = False,  # optional 64×64 context
        use_dct    : bool = True,
        use_coord  : bool = True,
        pfae_cls   : Type[nn.Module] = PFAEv4,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
                # heavy_stage_cfg: Optional[dict]
                # {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
                # If enabled, inserts PFAEHeavyStage after specified stage.
        ms_enc_cls: Type[nn.Module] = MultiScaleGlobalEncoderV3,
        wavelet_cls: Type[nn.Module] = WaveletTransformBlockV3,
        **kwargs
    ):
        super().__init__()
        # assert abs(sum(proj_ratios)-1) < 1e-6, "proj_ratios must sum to 1"
        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios

        self.hi_c = hi_c
        self.mid_c = mid_c
        self.mid2_enable = mid2_enable

        self.use_dct = use_dct
        self.use_coord = use_coord
        self.heavy_stage_cfg = heavy_stage_cfg

        self.ms_enc_cls = ms_enc_cls
        self.wavelet_cls = wavelet_cls

        # ── Hi-Res encoders (128²) ────────────────────────────────────── #
        self.ms_hi = nn.Sequential(
            ms_enc_cls(img_c, hi_c),
            nn.Conv2d(hi_c, hi_c, 1, bias=False)
        )
        self.wv_hi = nn.Sequential(
            wavelet_cls(img_c, hi_c),
            nn.Conv2d(hi_c, hi_c, 1, bias=False)
        )

        # ── Mid-Res pooling (32²) ────────────────────────────────────── #
        self.pool_32 = nn.AdaptiveAvgPool2d((32,32))      # 池化到 32×32
        self.ms_mid_proj  = nn.Conv2d(hi_c, mid_c, 1, bias=False)
        self.wv_mid_proj  = nn.Conv2d(hi_c, mid_c, 1, bias=False)

        # ── 可选第二中分辨率 64² ─────────────────────────────────────── #
        self.mid2_enable = mid2_enable
        if mid2_enable:
            self.pool_64   = nn.AdaptiveAvgPool2d((64,64))
            self.ms_mid2_proj = nn.Conv2d(hi_c, mid_c, 1, bias=False)
            self.wv_mid2_proj = nn.Conv2d(hi_c, mid_c, 1, bias=False)

        # ── SAM / UNI 投影 ─────────────────────────────────────────── #
        sam_ch = int(fusion_c * proj_ratios[0])
        uni_ch = int(fusion_c * proj_ratios[1])

        self.sam_proj = nn.Conv2d(sam_c, sam_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, uni_ch, 1, bias=False)

        # ── 融合映射到 fusion_c ─────────────────────────────────────── #
        fuse_in_channels = sam_ch + uni_ch + mid_c*2 + (mid_c*2 if mid2_enable else 0)
        self.fusion_proj = nn.Conv2d(fuse_in_channels, fusion_c, 1, bias=False)

        # ── PFAE (x0 + mid_res) ─────────────────────────────────────── #
        pfae_in = fusion_c + mid_c*2 + (mid_c*2 if mid2_enable else 0)
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=pfae_in, in_dim=pfae_in, out_dim=fusion_c,
                               num_stages=3, min_channels=16, 
                               use_dct=use_dct, 
                               heavy_stage_cfg=heavy_stage_cfg)
        else:
            pfae_kwargs["heavy_stage_cfg"] = heavy_stage_cfg
        self.pfae = pfae_cls(**pfae_kwargs)

        # ── Hi-Res gate & positional refine ─────────────────────────── #
        self.hi_gate_value = nn.Conv2d(hi_c*2, fusion_c*2, 1, bias=False)
        self.pos  = PositionEnhancedFusion(fusion_c, use_coord=use_coord)

    # ----------------------------------------------------------------- #
    def forward(self, img, sam_feat, uni_feat):
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckForNucleiV2.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), hi_c: {self.hi_c}, mid_c: {self.mid_c}")
            print(f" Args: mid2_enable: {self.mid2_enable}, use_dct: {self.use_dct}, use_coord: {self.use_coord}")
            # print(f" Args: heavy_stage_cfg: {self.heavy_stage_cfg}")
            print(f" input: img.shape: {img.shape}")
            print(f" input: sam_feat.shape: {sam_feat.shape}")
            print(f" input: uni_feat.shape: {uni_feat.shape}")
            
        B,_,H_img,W_img = img.shape             # 128×128
        _,_,Hf,Wf = sam_feat.shape              # 16×16

        # Hi-Res 128²
        ms_hi = self.ms_hi(img)                 # [B, hi_c,128,128]
        wv_hi = self.wv_hi(img)                 # idem
        if DEBUG:
            print(f" (Hi-Res) ms_hi_enc -> ms_hi.shape: {ms_hi.shape}")
            print(f" (Hi-Res) wv_hi_enc -> wv_hi.shape: {wv_hi.shape}")

        # Mid-Res 32²
        ms_mid = self.ms_mid_proj(self.pool_32(ms_hi))     # [B,mid_c,32,32]
        wv_mid = self.wv_mid_proj(self.pool_32(wv_hi))
        if DEBUG:
            print(f" (Mid-Res) ms_mid_proj -> ms_mid.shape: {ms_mid.shape}")
            print(f" (Mid-Res) wv_mid_proj -> wv_mid.shape: {wv_mid.shape}")
            
        # 再下采样到 16² 以与 sam/uni 对齐
        ms_mid16 = F.adaptive_avg_pool2d(ms_mid, (Hf,Wf))
        wv_mid16 = F.adaptive_avg_pool2d(wv_mid, (Hf,Wf))
        if DEBUG:
            print(f" (Mid-Res) F.adaptive_avg_pool2d(ms_mid) -> ms_mid16.shape: {ms_mid16.shape}")
            print(f" (Mid-Res) F.adaptive_avg_pool2d(wv_mid) -> wv_mid16.shape: {wv_mid16.shape}")

        # Second mid (64²) if enabled
        if self.mid2_enable:
            ms_mid2 = self.ms_mid2_proj(self.pool_64(ms_hi))   # 64²→proj
            wv_mid2 = self.wv_mid2_proj(self.pool_64(wv_hi))
            if DEBUG:
                print(f" (Mid-Res) ms_mid2_proj -> ms_mid2.shape: {ms_mid2.shape}")
                print(f" (Mid-Res) wv_mid2_proj -> wv_mid2.shape: {wv_mid2.shape}")

            ms_mid2_16 = F.adaptive_avg_pool2d(ms_mid2, (Hf,Wf))
            wv_mid2_16 = F.adaptive_avg_pool2d(wv_mid2, (Hf,Wf))
            if DEBUG:
                print(f" (Mid-Res) F.adaptive_avg_pool2d(ms_mid2) -> ms_mid2_16.shape: {ms_mid2_16.shape}")
                print(f" (Mid-Res) F.adaptive_avg_pool2d(wv_mid2) -> wv_mid2_16.shape: {wv_mid2_16.shape}")

        # SAM / UNI
        p_sam = self.sam_proj(sam_feat)
        p_uni = self.uni_proj(uni_feat)
        if DEBUG:
            print(f" (Lo-Res) sam_proj -> p_sam.shape: {p_sam.shape}")
            print(f" (Lo-Res) uni_proj -> p_uni.shape: {p_uni.shape}")

        # 拼接 & 投影到 fusion_c
        feats = [p_sam, p_uni, ms_mid16, wv_mid16]
        if self.mid2_enable:
            feats.extend([ms_mid2_16, wv_mid2_16])
        if DEBUG:
            print(f" (Lo-Res) feats len: {len(feats)}")
            for i, feat in enumerate(feats):
                print(f"     feats[{i}].shape: {feat.shape}")
        x0 = self.fusion_proj(torch.cat(feats, dim=1))     # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) fusion_proj -> x0.shape: {x0.shape}")

        # PFAE 处理 (带 32² / 64² 上下文)
        pfae_in = [x0, ms_mid16, wv_mid16]
        if self.mid2_enable:
            pfae_in += [ms_mid2_16, wv_mid2_16]
        if DEBUG:
            print(f" (Lo-Res) pfae_in len: {len(pfae_in)}")
            for i, feat in enumerate(pfae_in):
                print(f"     pfae_in[{i}].shape: {feat.shape}")
        x_pfae = self.pfae(torch.cat(pfae_in, dim=1))
        if DEBUG:
            print(f" (Lo-Res) pfae -> x_pfae.shape: {x_pfae.shape}")

        # Hi-Res gate注入
        hi_feat = torch.cat([ms_hi, wv_hi], dim=1)
        if DEBUG:
            print(f" (Hi-Res) torch.cat([ms_hi, wv_hi]) -> hi_feat.shape: {hi_feat.shape}")
        hi_feat = F.interpolate(hi_feat, size=(Hf,Wf), mode='bilinear', align_corners=False)
        if DEBUG:
            print(f" (Hi-Res) F.interpolate(hi_feat) -> hi_feat.shape: {hi_feat.shape}")
        
        if DEBUG:
            print(f"\n final output pos input: ")
            print(f"   input x0.shape: {x0.shape}")
            print(f"   input x_pfae.shape: {x_pfae.shape}")
            print(f"   input hi_feat.shape: {hi_feat.shape}")

        gv = self.hi_gate_value(hi_feat)
        gate, value = gv.chunk(2, dim=1)
        gated_hi = torch.sigmoid(gate) * value

        if DEBUG:
            print(f"   gate.shape: {gate.shape}")
            print(f"   value.shape: {value.shape}")
            print(f"   gated_hi.shape: {gated_hi.shape}")

        x_out = self.pos(x0 + x_pfae + gated_hi)

        if DEBUG:
            print(f" Final output: (Hi-Res) gate output -> x_out.shape: {x_out.shape}\n")
        return x_out


class PFAEGlobalFusionNeckForNucleiV2_woImages(nn.Module):
    """
    Hi-Res (128×128) + Mid-Res (32×32) Pixel Encoder  →  SAM / UNI / MS / WAVE 融合

    Pipeline
    --------
    img ─► MS_hi / WV_hi ─┬─► 32×32 池化 → sam/uni 拼接 → 1×1 (fusion_proj) ─┐
                           └─► 128×128 gate (细节)                       │
    sam_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┤
    uni_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┘
    """
    def __init__(
        self,
        img_c      : int,
        sam_c      : int,
        uni_c      : int,
        fusion_c   : int = 256,
        proj_ratios: Tuple[float,float] = (0.4, 0.4),  # sam : uni
        hi_c       : int = 8,   # channels of MS/WAVE in 128²
        mid_c      : int = 16,  # channels after 32² pooling
        mid2_enable: bool = False,  # optional 64×64 context
        use_dct    : bool = True,
        use_coord  : bool = True,
        pfae_cls   : Type[nn.Module] = PFAEv4,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
                # heavy_stage_cfg: Optional[dict]
                # {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
                # If enabled, inserts PFAEHeavyStage after specified stage.
        **kwargs
    ):
        super().__init__()
        # assert abs(sum(proj_ratios)-1) < 1e-6, "proj_ratios must sum to 1"
        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios

        self.hi_c = hi_c
        self.mid_c = mid_c
        self.mid2_enable = mid2_enable

        self.use_dct = use_dct
        self.use_coord = use_coord
        self.heavy_stage_cfg = heavy_stage_cfg
    
        # ── SAM / UNI 投影 ─────────────────────────────────────────── #
        sam_ch = int(fusion_c * proj_ratios[0])
        uni_ch = int(fusion_c * proj_ratios[1])

        self.sam_proj = nn.Conv2d(sam_c, sam_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, uni_ch, 1, bias=False)

        # ── 融合映射到 fusion_c ─────────────────────────────────────── #
        fuse_in_channels = sam_ch + uni_ch
        self.fusion_proj = nn.Conv2d(fuse_in_channels, fusion_c, 1, bias=False)

        # ── PFAE (x0 + mid_res) ─────────────────────────────────────── #
        pfae_in = fusion_c
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=pfae_in, in_dim=pfae_in, out_dim=fusion_c,
                               num_stages=3, min_channels=16, 
                               use_dct=use_dct, 
                               heavy_stage_cfg=heavy_stage_cfg)
        else:
            pfae_kwargs["heavy_stage_cfg"] = heavy_stage_cfg
        self.pfae = pfae_cls(**pfae_kwargs)

        # ── Hi-Res gate & positional refine ─────────────────────────── #
        self.hi_gate_value = nn.Conv2d(hi_c*2, fusion_c*2, 1, bias=False)
        self.pos  = PositionEnhancedFusion(fusion_c, use_coord=use_coord)

    # ----------------------------------------------------------------- #
    def forward(self, img, sam_feat, uni_feat):
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckForNucleiV2.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), hi_c: {self.hi_c}, mid_c: {self.mid_c}")
            print(f" Args: mid2_enable: {self.mid2_enable}, use_dct: {self.use_dct}, use_coord: {self.use_coord}")
            # print(f" Args: heavy_stage_cfg: {self.heavy_stage_cfg}")
            print(f" input: img.shape: {img.shape}")
            print(f" input: sam_feat.shape: {sam_feat.shape}")
            print(f" input: uni_feat.shape: {uni_feat.shape}")
            
        B,_,H_img,W_img = img.shape             # 128×128
        _,_,Hf,Wf = sam_feat.shape              # 16×16

        # SAM / UNI
        p_sam = self.sam_proj(sam_feat)
        p_uni = self.uni_proj(uni_feat)
        if DEBUG:
            print(f" (Lo-Res) sam_proj -> p_sam.shape: {p_sam.shape}")
            print(f" (Lo-Res) uni_proj -> p_uni.shape: {p_uni.shape}")

        # 拼接 & 投影到 fusion_c
        feats = [p_sam, p_uni]
        if DEBUG:
            print(f" (Lo-Res) feats len: {len(feats)}")
            for i, feat in enumerate(feats):
                print(f"     feats[{i}].shape: {feat.shape}")
        x0 = self.fusion_proj(torch.cat(feats, dim=1))     # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) fusion_proj -> x0.shape: {x0.shape}")

        # PFAE 处理 (带 32² / 64² 上下文)
        x_pfae = self.pfae(x0)
        if DEBUG:
            print(f" (Lo-Res) pfae -> x_pfae.shape: {x_pfae.shape}")

        if DEBUG:
            print(f"\n final output pos input: ")
            print(f"   input x0.shape: {x0.shape}")
            print(f"   input x_pfae.shape: {x_pfae.shape}")

        x_out = self.pos(x0 + x_pfae)

        if DEBUG:
            print(f" Final output: (Hi-Res) gate output -> x_out.shape: {x_out.shape}\n")
        return x_out

class PFAEGlobalFusionNeckForNucleiV3(nn.Module):
    """
    现在的V3和V2还是一模一样的。
    Hi-Res (128×128) + Mid-Res (32×32) Pixel Encoder  →  SAM / UNI / MS / WAVE 融合

    Pipeline
    --------
    img ─► MS_hi / WV_hi ─┬─► 32×32 池化 → sam/uni 拼接 → 1×1 (fusion_proj) ─┐
                           └─► 128×128 gate (细节)                       │
    sam_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┤
    uni_feat 16×16 ─► 1×1 ───────────────────────────────────────────────┘
    """
    def __init__(
        self,
        img_c      : int,
        sam_c      : int,
        uni_c      : int,
        fusion_c   : int = 256,
        proj_ratios: Tuple[float,float] = (0.4, 0.4),  # sam : uni
        hi_c       : int = 8,   # channels of MS/WAVE in 128²
        mid_c      : int = 16,  # channels after 32² pooling
        mid2_enable: bool = False,  # optional 64×64 context
        use_dct    : bool = True,
        use_coord  : bool = True,
        pfae_cls   : Type[nn.Module] = PFAEv5Hybrid,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
                # heavy_stage_cfg: Optional[dict]
                # {"enable": bool, "stage_idx": int, "gamma_init": float, "num_heads": int}
                # If enabled, inserts PFAEHeavyStage after specified stage.
        ms_enc_cls: Type[nn.Module] = MultiScaleGlobalEncoderV3,
        wavelet_cls: Type[nn.Module] = WaveletTransformBlockV3,
        **kwargs
    ):
        super().__init__()
        # assert abs(sum(proj_ratios)-1) < 1e-6, "proj_ratios must sum to 1"
        self.img_c = img_c
        self.sam_c = sam_c
        self.uni_c = uni_c
        self.fusion_c = fusion_c
        self.proj_ratios = proj_ratios

        self.hi_c = hi_c
        self.mid_c = mid_c
        self.mid2_enable = mid2_enable

        self.use_dct = use_dct
        self.use_coord = use_coord
        self.heavy_stage_cfg = heavy_stage_cfg
        self.ms_enc_cls = ms_enc_cls
        self.wavelet_cls = wavelet_cls

        # ── Hi-Res encoders (128²) ────────────────────────────────────── #
        self.ms_hi = nn.Sequential(
            ms_enc_cls(img_c, hi_c),
            nn.Conv2d(hi_c, hi_c, 1, bias=False)
        )
        self.wv_hi = nn.Sequential(
            wavelet_cls(img_c, hi_c),
            nn.Conv2d(hi_c, hi_c, 1, bias=False)
        )

        # ── Mid-Res pooling (32²) ────────────────────────────────────── #
        self.pool_32 = nn.AdaptiveAvgPool2d((32,32))      # 池化到 32×32
        self.ms_mid_proj  = nn.Conv2d(hi_c, mid_c, 1, bias=False)
        self.wv_mid_proj  = nn.Conv2d(hi_c, mid_c, 1, bias=False)

        # ── 可选第二中分辨率 64² ─────────────────────────────────────── #
        self.mid2_enable = mid2_enable
        if mid2_enable:
            self.pool_64   = nn.AdaptiveAvgPool2d((64,64))
            self.ms_mid2_proj = nn.Conv2d(hi_c, mid_c, 1, bias=False)
            self.wv_mid2_proj = nn.Conv2d(hi_c, mid_c, 1, bias=False)

        # ── SAM / UNI 投影 ─────────────────────────────────────────── #
        sam_ch = int(fusion_c * proj_ratios[0])
        uni_ch = int(fusion_c * proj_ratios[1])

        self.sam_proj = nn.Conv2d(sam_c, sam_ch, 1, bias=False)
        self.uni_proj = nn.Conv2d(uni_c, uni_ch, 1, bias=False)

        # ── 融合映射到 fusion_c ─────────────────────────────────────── #
        fuse_in_channels = sam_ch + uni_ch + mid_c*2 + (mid_c*2 if mid2_enable else 0)
        self.fusion_proj = nn.Conv2d(fuse_in_channels, fusion_c, 1, bias=False)

        # ── PFAE (x0 + mid_res) ─────────────────────────────────────── #
        pfae_in = fusion_c + mid_c*2 + (mid_c*2 if mid2_enable else 0)
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=pfae_in, in_dim=pfae_in, out_dim=fusion_c,
                               num_stages=3, min_channels=16, 
                               use_dct=use_dct, 
                               heavy_stage_cfg=heavy_stage_cfg)
        else:
            pfae_kwargs["heavy_stage_cfg"] = heavy_stage_cfg
        self.pfae = pfae_cls(**pfae_kwargs)

        # ── Hi-Res gate & positional refine ─────────────────────────── #
        self.hi_gate_value = nn.Conv2d(hi_c*2, fusion_c*2, 1, bias=False)

        self.pos  = PositionEnhancedFusion(fusion_c, use_coord=use_coord)

    # ----------------------------------------------------------------- #
    def forward(self, img, sam_feat, uni_feat):
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAEGlobalFusionNeckForNucleiV2.forward() -------")
            print(f" Args: img_c: {self.img_c}, sam_c: {self.sam_c}, uni_c: {self.uni_c}, fusion_c: {self.fusion_c}")
            print(f" Args: proj_ratios: {self.proj_ratios} = (sam:uni), hi_c: {self.hi_c}, mid_c: {self.mid_c}")
            print(f" Args: mid2_enable: {self.mid2_enable}, use_dct: {self.use_dct}, use_coord: {self.use_coord}")
            # print(f" Args: heavy_stage_cfg: {self.heavy_stage_cfg}")
            print(f" input: img.shape: {img.shape}")
            print(f" input: sam_feat.shape: {sam_feat.shape}")
            print(f" input: uni_feat.shape: {uni_feat.shape}")
            
        B,_,H_img,W_img = img.shape             # 128×128
        _,_,Hf,Wf = sam_feat.shape              # 16×16

        # Hi-Res 128²
        ms_hi = self.ms_hi(img)                 # [B, hi_c,128,128]
        wv_hi = self.wv_hi(img)                 # idem
        if DEBUG:
            print(f" (Hi-Res) ms_hi_enc -> ms_hi.shape: {ms_hi.shape}")
            print(f" (Hi-Res) wv_hi_enc -> wv_hi.shape: {wv_hi.shape}")

        # Mid-Res 32²
        ms_mid = self.ms_mid_proj(self.pool_32(ms_hi))     # [B,mid_c,32,32]
        wv_mid = self.wv_mid_proj(self.pool_32(wv_hi))
        if DEBUG:
            print(f" (Mid-Res) ms_mid_proj -> ms_mid.shape: {ms_mid.shape}")
            print(f" (Mid-Res) wv_mid_proj -> wv_mid.shape: {wv_mid.shape}")
            
        # 再下采样到 16² 以与 sam/uni 对齐
        ms_mid16 = F.adaptive_avg_pool2d(ms_mid, (Hf,Wf))
        wv_mid16 = F.adaptive_avg_pool2d(wv_mid, (Hf,Wf))
        if DEBUG:
            print(f" (Mid-Res) F.adaptive_avg_pool2d(ms_mid) -> ms_mid16.shape: {ms_mid16.shape}")
            print(f" (Mid-Res) F.adaptive_avg_pool2d(wv_mid) -> wv_mid16.shape: {wv_mid16.shape}")

        # Second mid (64²) if enabled
        if self.mid2_enable:
            ms_mid2 = self.ms_mid2_proj(self.pool_64(ms_hi))   # 64²→proj
            wv_mid2 = self.wv_mid2_proj(self.pool_64(wv_hi))
            if DEBUG:
                print(f" (Mid-Res) ms_mid2_proj -> ms_mid2.shape: {ms_mid2.shape}")
                print(f" (Mid-Res) wv_mid2_proj -> wv_mid2.shape: {wv_mid2.shape}")

            ms_mid2_16 = F.adaptive_avg_pool2d(ms_mid2, (Hf,Wf))
            wv_mid2_16 = F.adaptive_avg_pool2d(wv_mid2, (Hf,Wf))
            if DEBUG:
                print(f" (Mid-Res) F.adaptive_avg_pool2d(ms_mid2) -> ms_mid2_16.shape: {ms_mid2_16.shape}")
                print(f" (Mid-Res) F.adaptive_avg_pool2d(wv_mid2) -> wv_mid2_16.shape: {wv_mid2_16.shape}")

        # SAM / UNI
        p_sam = self.sam_proj(sam_feat)
        p_uni = self.uni_proj(uni_feat)
        if DEBUG:
            print(f" (Lo-Res) sam_proj -> p_sam.shape: {p_sam.shape}")
            print(f" (Lo-Res) uni_proj -> p_uni.shape: {p_uni.shape}")

        # 拼接 & 投影到 fusion_c
        feats = [p_sam, p_uni, ms_mid16, wv_mid16]
        if self.mid2_enable:
            feats.extend([ms_mid2_16, wv_mid2_16])
        if DEBUG:
            print(f" (Lo-Res) feats len: {len(feats)}")
            for i, feat in enumerate(feats):
                print(f"     feats[{i}].shape: {feat.shape}")
        x0 = self.fusion_proj(torch.cat(feats, dim=1))     # [B,fusion_c,16,16]
        if DEBUG:
            print(f" (Lo-Res) fusion_proj -> x0.shape: {x0.shape}")

        # PFAE 处理 (带 32² / 64² 上下文)
        pfae_in = [x0, ms_mid16, wv_mid16]
        if self.mid2_enable:
            pfae_in += [ms_mid2_16, wv_mid2_16]
        if DEBUG:
            print(f" (Lo-Res) pfae_in len: {len(pfae_in)}")
            for i, feat in enumerate(pfae_in):
                print(f"     pfae_in[{i}].shape: {feat.shape}")
        x_pfae = self.pfae(torch.cat(pfae_in, dim=1))
        if DEBUG:
            print(f" (Lo-Res) pfae -> x_pfae.shape: {x_pfae.shape}")

        # Hi-Res gate注入
        hi_feat = torch.cat([ms_hi, wv_hi], dim=1)
        if DEBUG:
            print(f" (Hi-Res) torch.cat([ms_hi, wv_hi]) -> hi_feat.shape: {hi_feat.shape}")
        hi_feat = F.interpolate(hi_feat, size=(Hf,Wf), mode='bilinear', align_corners=False)
        if DEBUG:
            print(f" (Hi-Res) F.interpolate(hi_feat) -> hi_feat.shape: {hi_feat.shape}")
        
        if DEBUG:
            print(f"\n final output pos input: ")
            print(f"   input x0.shape: {x0.shape}")
            print(f"   input x_pfae.shape: {x_pfae.shape}")
            print(f"   input hi_feat.shape: {hi_feat.shape}")

        gv = self.hi_gate_value(hi_feat)
        gate, value = gv.chunk(2, dim=1)
        gated_hi = torch.sigmoid(gate) * value

        if DEBUG:
            print(f"   gate.shape: {gate.shape}")
            print(f"   value.shape: {value.shape}")
            print(f"   gated_hi.shape: {gated_hi.shape}")

        x_out = self.pos(x0 + x_pfae + gated_hi)

        if DEBUG:
            print(f" Final output: (Hi-Res) gate output -> x_out.shape: {x_out.shape}\n")
        return x_out

class PFAESkipEnhanceNeck(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        pfae_cls: Type[nn.Module] = PFAEv2,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if pfae_kwargs is None:
            pfae_kwargs = dict(dim=in_c, in_dim=in_c, out_dim=out_c, num_stages=1, min_c=1, use_dct=False)
        self.enh = pfae_cls(**pfae_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enh(x)

class PFAESkipEnhanceNeckV5(nn.Module):
    """
    Lightweight skip-level enhancement neck using configurable PFAE module.

    Args:
        in_c (int): Input channel dimension.
        out_c (int): Output channel dimension.
        pfae_cls (Type[nn.Module]): Which PFAE variant to use.
        pfae_kwargs (dict): Parameters passed to the PFAE module.
        heavy_stage_cfg (dict): Parameters for the PFAE module in the heavy stage.
        use_proj (bool): Whether to project input channels before feeding into PFAE.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        pfae_cls: Type[nn.Module] = PFAEv2,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        use_proj: bool = False
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.pfae_cls = pfae_cls
        self.use_proj = use_proj

        if self.use_proj:
            self.in_proj = nn.Conv2d(in_c, out_c, kernel_size=1)

        # Setup default PFAE parameters if not provided
        if pfae_kwargs is None:
            pfae_kwargs = dict(
                dim=out_c,
                in_dim=out_c,
                out_dim=out_c,
                num_stages=1,
                min_c=max(4, out_c // 16),
                use_dct=False,
                heavy_stage_cfg=heavy_stage_cfg,
            )
        else:
            pfae_kwargs['heavy_stage_cfg'] = heavy_stage_cfg

        self.pfae_kwargs = pfae_kwargs
        self.enhancer = pfae_cls(**pfae_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            print(f"\n ------- network/sam_pfae_fusion_neck_modules.py, PFAESkipEnhanceNeckV5.forward() -------")
            print(f" Args: in_c: {self.in_c}, out_c: {self.out_c}, use_proj: {self.use_proj}")
            print(f" Args: pfae_cls: {self.pfae_cls}")
            print(f" Args: pfae_kwargs: {self.pfae_kwargs}")
            print(f" input: x.shape: {x.shape}")

        if self.use_proj:
            x = self.in_proj(x)
            if DEBUG:
                print(f" (Lo-Res) in_proj -> x.shape: {x.shape}")
        x = self.enhancer(x)
        if DEBUG:
            print(f" (Lo-Res) enhancer -> x.shape: {x.shape}\n")
        return x

class EdgeGuideConv(nn.Module):
    def __init__(self, channels: int, use_sobel: bool = True):
        super().__init__()
        self.channels = channels
        self.use_sobel = use_sobel

        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            print(f"\n ------- EdgeGuideConv.forward() -------")
            print(f" Args: channels: {self.channels}, use_sobel: {self.use_sobel}")
            print(f" input: x.shape: {x.shape}")

        if self.use_sobel:
            sobel_kernel = self._sobel_kernel(x.device)  # [C,1,3,3]
            sobel_x = F.conv2d(x, weight=sobel_kernel, padding=1, groups=self.channels)
            if DEBUG:
                print(f" (Lo-Res) use Sobel sobel_kernel.shape: {sobel_kernel.shape}")
                print(f" (Lo-Res) use Sobel sobel_x.shape: {sobel_x.shape}")
            edge = torch.abs(sobel_x)
        else:
            edge = torch.abs(x - F.avg_pool2d(x, 3, stride=1, padding=1))

        if DEBUG:
            print(f" (Lo-Res) Edge -> edge.shape: {edge.shape}")
        
        edge = self.edge_conv(edge)
        if DEBUG:
            print(f" (Lo-Res) output edge_conv -> edge.shape: {edge.shape}\n")
        return edge

    def _sobel_kernel(self, device):
        # [1, 1, 3, 3]
        k = torch.tensor([[[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]], dtype=torch.float32)
        k = k.view(1, 1, 3, 3)  # [1, 1, 3, 3]
        # Expand to [C, 1, 3, 3] for depthwise conv
        k = k.expand(self.channels, 1, 3, 3).contiguous()
        return k.to(device)


# -----------------------------------------------------------------------------
# PFAESkipEnhanceNeckV6, 与PFAESkipEnhanceNeckV5不同在于设置了Edge-Guided with EdgeGuideConv
# -----------------------------------------------------------------------------
class PFAESkipEnhanceNeckV6(nn.Module):
    """
    Lightweight skip-level enhancement neck using configurable PFAE module.

    Args:
        in_c (int): Input channel dimension.
        out_c (int): Output channel dimension.
        pfae_cls (Type[nn.Module]): Which PFAE variant to use.
        pfae_kwargs (dict): Parameters passed to the PFAE module.
        heavy_stage_cfg (dict): Parameters for the PFAE module in the heavy stage.
        use_proj (bool): Whether to project input channels before feeding into PFAE.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        pfae_cls: Type[nn.Module] = PFAEv2,
        pfae_kwargs: Optional[Dict[str, Any]] = None,
        heavy_stage_cfg: Optional[Dict[str, Any]] = None,
        use_proj: bool = True,
        use_edge: bool = True,
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.pfae_cls = pfae_cls
        self.use_proj = use_proj
        self.use_edge = use_edge

        if self.use_proj:
            self.in_proj = nn.Conv2d(in_c, out_c, kernel_size=1)

        if self.use_edge:
            self.edge_enhancer = EdgeGuideConv(out_c)

        # Setup default PFAE parameters if not provided
        if pfae_kwargs is None:
            pfae_kwargs = dict(
                dim=out_c,
                in_dim=out_c,
                out_dim=out_c,
                num_stages=1,
                min_c=max(4, out_c // 16),
                use_dct=False,
                heavy_stage_cfg=heavy_stage_cfg,
            )
        else:
            pfae_kwargs['heavy_stage_cfg'] = heavy_stage_cfg

        self.pfae_kwargs = pfae_kwargs
        self.enhancer = pfae_cls(**pfae_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            print(f"\n ------- PFAESkipEnhanceNeckV6.forward() -------")
            print(f" Args: in_c: {self.in_c}, out_c: {self.out_c}, use_proj: {self.use_proj}, use_edge: {self.use_edge}")
            print(f" Args: pfae_cls: {self.pfae_cls}")
            print(f" Args: pfae_kwargs: {self.pfae_kwargs}")
            print(f" input: x.shape: {x.shape}")

        if self.use_proj:
            x = self.in_proj(x)
            if DEBUG:
                print(f" (Lo-Res) in_proj -> x.shape: {x.shape}")

        if self.use_edge:
            edge_feat = self.edge_enhancer(x)
            if DEBUG:
                print(f" (Lo-Res) edge_enhancer edge_feat.shape: {edge_feat.shape}")
            x = x + edge_feat
            if DEBUG:
                print(f" (Lo-Res) edge_enhancer -> x.shape: {x.shape}")

        x = self.enhancer(x)
        if DEBUG:
            print(f" (Lo-Res) enhancer -> x.shape: {x.shape}\n")
        return x

# -----------------------------------------------------------------------------
# 5. SAMNuSCNetV232 (decoder stubs only shown)
# -----------------------------------------------------------------------------

class DummyDecoder(nn.Module):
    def __init__(self,in_c:int,skip_c:List[int],out_ch:int):
        super().__init__()
        self.conv = nn.Conv2d(in_c,sum(skip_c)+in_c,3,1,1)
        self.proj = nn.Conv2d(sum(skip_c)+in_c,out_ch,1)
    def forward(self,fusion,skips,bin_feats=None):
        x = torch.cat([fusion,*skips],1)
        x = self.proj(self.conv(x))
        return x, [], fusion  # placeholder

class SAMNuSCNetV232(nn.Module):
    def __init__(self,img_c:int=3,sam_c:int=256,uni_c:int=256,base_c:int=256):
        super().__init__()
        # --- Encoders (placeholders) ---
        self.sam_enc = nn.Conv2d(img_c,sam_c,3,1,1)
        self.uni_enc = nn.Conv2d(img_c,uni_c,3,1,1)
        # dummy extra skip feats
        self.skip1 = nn.Conv2d(img_c,16,3,1,1)
        self.skip2 = nn.Conv2d(img_c,64,3,1,1)
        # --- Fusion & skip necks ---
        self.fusion_neck = PFAEGlobalFusionNeckV2(img_c,sam_c,uni_c,out_c=base_c)
        self.skip_neck2  = PFAESkipEnhanceNeck(64,64,num_stages=3)
        self.skip_neck1  = PFAESkipEnhanceNeck(16,16,num_stages=1)
        # --- Decoders (stubs) ---
        self.binary_decoder  = DummyDecoder(base_c,[64,16],1)
        self.hv_decoder      = DummyDecoder(base_c,[64,16],2)
        self.type_decoder    = DummyDecoder(base_c,[64,16],7)

    def forward(self, img: torch.Tensor)->Dict[str,torch.Tensor]:
        sam_feat = self.sam_enc(img)
        uni_feat = self.uni_enc(img)
        skip2 = self.skip_neck2(self.skip2(img))
        skip1 = self.skip_neck1(self.skip1(img))
        fusion = self.fusion_neck(img,sam_feat,uni_feat)
        bin_map, _, bin_feats = self.binary_decoder(fusion,[skip2,skip1])
        hv_map,  _, _         = self.hv_decoder(fusion,[skip2,skip1],bin_feats)
        tp_map,  _, _         = self.type_decoder(fusion,[skip2,skip1],bin_feats)
        return {"bin":bin_map,"hv":hv_map,"tp":tp_map}

# -----------------------------------------------------------------------------
# 6. BN→GN converter stays unchanged from previous version
# -----------------------------------------------------------------------------

def convert_bn_to_gn(module:nn.Module,num_groups:int=8):
    for name,child in module.named_children():
        if isinstance(child,nn.BatchNorm2d):
            c=child.num_features; g=num_groups if c>=num_groups else 1
            gn=nn.GroupNorm(g,c)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module,name,gn)
        else:
            convert_bn_to_gn(child,num_groups)
    return module

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    img = torch.randn(2,3,256,256).cuda() if torch.cuda.is_available() else torch.randn(2,3,256,256)
    model = SAMNuSCNetV232().to(img.device)
    out = model(img)
    for k,v in out.items():
        print(k,v.shape)
