import sys
sys.path.append('/root/SAM2PATH-main')

import logging
from typing import Optional, Tuple, Dict, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# from network.FMNET_module import *
from network.SAM_NuSCNet_modules import BinaryDecoder, HVDecoder
from network.SAM_NuSCNet_modules import TypeDecoderV3 as TypeDecoder
from network.SAM_NuSCNet_utils import get_grid_from_patch_size, print_trainable_params

from network.sam_pfae_fusion_neck_modules import (
    PFAEGlobalFusionNeckV5,
    PFAESkipEnhanceNeckV5,
    PFAEv5Hybrid,
    MultiScaleGlobalEncoderV3,
    WaveletTransformBlockV3,
)

from segment_anything_local import sam_model_registry_adapter as sam_model_registry

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False
# DEBUG = True     # set to True to print debug messages

# 获取当前文件名
import os
file_name = os.path.basename(__file__)
if "debug" in file_name:
    DEBUG = True

if DEBUG:
    print(f"\n ------ Debug mode is {DEBUG}, setting on {file_name} ------ \n")

show_MEMORY = False
# show_MEMORY = True

class SAMNuSCNetV231(nn.Module):
    """
    Fusion network combining SAM image encoder and an auxiliary ViT-based encoder.

    - Requires extra_encoder to be provided.
    - Freezes both backbones, trains only their prompt_generator adapters.
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str = "",
        num_classes: int = 6,
        extra_encoder: Optional[nn.Module] = None,
        freeze_image_encoder: bool = True,
        input_HW: Tuple[int,int] = (128, 128),
        fusion_heads: int = 4,
        base_channels: int = 256,
        output_aux_tokens: bool = False,
        **kwargs
    ):
        super().__init__()
        if extra_encoder is None:
            raise ValueError("SAMNuSCNetV231 requires an extra_encoder instance, got None.")

        # Basic settings
        self.input_HW = input_HW
        self.num_classes = num_classes
        self.output_aux_tokens = output_aux_tokens

        # Patch & grid
        ph, pw = 8, 8
        gh, gw = get_grid_from_patch_size(input_HW, patch_size=(ph, pw))
        self.patch_size = (ph, pw)
        self.grid_size = (gh, gw)

        # Skip feature shapes
        self.ex_skip_shapes = [(gh*2, gw*2), (gh*4, gw*4), (gh*8, gw*8)]

        # Decoder dims
        self.decoder_feat_dims = [256, 128, 64, 32]
        self.skip_dims = [16, 256, 64, 16]

        # === 1. SAM image encoder adaptation ===
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint,
                                                  img_size=input_HW[0],
                                                  patch_size=ph)
        # self.sam.prompt_encoder = None  # 去掉 prompt encoder
        self.sam.mask_decoder = None  # 去掉 mask decoder
        self.sam.head = None  # 去掉 SAM head

        sam_pe = self.sam.image_encoder.patch_embed
        sam_out_ch = sam_pe.proj.out_channels
        print(f" SAM image encoder: {self.sam.image_encoder.__class__.__name__} with {sam_out_ch} output channels.")
        print(f" SAM image encoder patch_embed shape: {sam_pe.proj.weight.shape}") # torch.Size([1024, 3, 16, 16])

        # Resize patch embedding to (ph, pw)
        sam_pe.proj = nn.Conv2d(3, sam_out_ch, kernel_size=(ph, pw), stride=(ph, pw))
        self._interp_pos4d(self.sam.image_encoder, (gh, gw))

        # 冻结 / 解冻 SAM backbone
        for n,p in self.sam.image_encoder.named_parameters():
            p.requires_grad = ('prompt_generator' in n or 'patch_embed.proj' in n)

        # # # adjust patch embedding and pos_embed
        # sam_pe = self.sam.image_encoder.patch_embed
        # sam_out_ch = sam_pe.proj.out_channels
        # sam_pe.proj = nn.Conv2d(3, sam_out_ch, kernel_size=(ph,pw), stride=(ph,pw))
        # self._interp_pos4d(self.sam.image_encoder, (gh,gw))

        # === 2. Extra encoder adaptation ===
        self.extra_encoder = extra_encoder
        vit = extra_encoder.model
        # patch embedding
        ee_pe = vit.patch_embed
        ee_ch = ee_pe.proj.out_channels
        ee_pe.proj = nn.Conv2d(3, ee_ch, kernel_size=(ph,pw), stride=(ph,pw))
        # prompt embedding
        ee_prompt = extra_encoder.prompt_generator.prompt_generator
        ee_prompt_ch = ee_prompt.proj.out_channels
        ee_prompt.proj = nn.Conv2d(3, ee_prompt_ch, kernel_size=(ph,pw), stride=(ph,pw))
        # pos_embed
        if hasattr(vit, 'pos_embed'):
            if vit.pos_embed.ndim == 4:
                self._interp_pos4d(vit, (gh,gw))
            elif vit.pos_embed.ndim == 3:
                self._interp_pos3d(vit, (gh,gw))
            else:
                raise ValueError(f"Unknown pos_embed ndim: {vit.pos_embed.shape}")

        # === 3. Freeze backbones, train adapters ===
        if freeze_image_encoder:
            logger.info("Freezing SAM encoder backbone, training only adapters.")
            for name, param in self.sam.image_encoder.named_parameters():
                requires = 'patch_embed.proj' in name or 'prompt_generator' in name
                param.requires_grad = requires
        else:
            logger.info("SAM encoder fully trainable.")

        logger.info("Freezing extra encoder backbone, training only its adapters.")
        for name, param in self.extra_encoder.model.named_parameters():
            param.requires_grad = False
        for name, param in self.extra_encoder.prompt_generator.named_parameters():
            param.requires_grad = True

        # === 4. Fusion necks ===
        self.fusion_neck = PFAEGlobalFusionNeckV5(
            img_c=3, sam_c=sam_out_ch, uni_c=ee_ch, fusion_c=base_channels,
            pfae_cls=PFAEv5Hybrid,
            pfae_kwargs=None, # set to None to use default values
            proj_ratios=(0.4, 0.4, 0.1, 0.1), 
            use_dct=True, use_coord=True,
            ms_enc_cls=MultiScaleGlobalEncoderV3,
            wavelet_cls=WaveletTransformBlockV3,
            dropout_rate=0.0
        )

        self.skip_neck2 = PFAESkipEnhanceNeckV5(
            in_c=64, out_c=64, pfae_cls=PFAEv5Hybrid,
            pfae_kwargs=dict(dim=64, in_dim=64, out_dim=64, num_stages=2,
                             min_channels=8,use_dct=True)
        )
        self.skip_neck1 = PFAESkipEnhanceNeckV5(
            in_c=16, out_c=16, pfae_cls=PFAEv5Hybrid,
            pfae_kwargs=dict(dim=16, in_dim=16, out_dim=16, num_stages=2,
                             min_channels=8,use_dct=True)
        )

        # === 5. Decoders ===
        print(f"\n ------------ SAMNuSCNetV231")
        print(f" self.decoder_feat_dims = {self.decoder_feat_dims}")
        print(f" self.skip_dims = {self.skip_dims}")
        print(f" output_aux_tokens = {self.output_aux_tokens}")
        print(f" num_classes = {num_classes}")
        print(f" prompt_dims = {self.decoder_feat_dims}")

        self.binary_decoder = BinaryDecoder(
            feat_dims=self.decoder_feat_dims, 
            skip_dims=self.skip_dims,
            output_aux_tokens=self.output_aux_tokens
        )
        self.hv_decoder = HVDecoder(
            feat_dims=self.decoder_feat_dims, 
            skip_dims=self.skip_dims,
            output_aux_tokens=self.output_aux_tokens
        )
        self.type_decoder = TypeDecoder(
            feat_dims=self.decoder_feat_dims,
            skip_dims=self.skip_dims,
            num_classes=num_classes,
            prompt_dims=self.decoder_feat_dims, 
            output_aux_tokens=True,     # always output aux tokens in type decoder
        )

        # === 7. Init weights ===
        self._set_trainable_flags()

        # === 6. Parameter summary ===
        self._print_param_summary()

        # Log trainable parameters
        print_trainable_params(self)

    def _interp_pos4d(self, module: nn.Module, new_grid: Tuple[int,int]):
        """Interpolate 4D pos_embed channels."""
        old = module.pos_embed
        B,H,W,C = old.shape
        t = old.permute(0,3,1,2)
        nh,nw = new_grid
        t2 = F.interpolate(t, size=(nh,nw), mode='bilinear', align_corners=False)
        new = t2.permute(0,2,3,1).to(old.device, old.dtype)
        module.pos_embed = nn.Parameter(new)

    def _interp_pos3d(self, module: nn.Module, new_grid: Tuple[int,int]):
        """Interpolate 3D ViT-style pos_embed."""
        old = module.pos_embed  # [1, N+1, C]
        cls_tok, spatial = old[:,0:1,:], old[:,1:,:]
        N,C = spatial.shape[1], spatial.shape[2]
        gs = int(N**0.5)
        assert gs*gs==N, f"Cannot infer grid from N={N}"
        t = spatial.view(1,gs,gs,C).permute(0,3,1,2)
        nh,nw=new_grid
        t2 = F.interpolate(t, size=(nh,nw), mode='bilinear', align_corners=False)
        sp = t2.permute(0,2,3,1).view(1,nh*nw,C)
        new = torch.cat((cls_tok, sp), dim=1).to(old.device, old.dtype)
        module.pos_embed = nn.Parameter(new)
    
    def _print_param_summary(self):
        """Print parameter counts for each sub-module."""
        parts = {
            'SAM encoder backbone': self.sam.image_encoder.patch_embed,
            'SAM adapter': self.sam.image_encoder.prompt_generator,
            'Extra encoder backbone': self.extra_encoder.model,
            'Extra encoder adapter': self.extra_encoder.prompt_generator,
            'Fusion neck': self.fusion_neck,
            'Skip necks': nn.ModuleList([self.skip_neck1, self.skip_neck2]),
            'Binary decoder': self.binary_decoder,
            'HV decoder': self.hv_decoder,
            'Type decoder': self.type_decoder,
        }
        logger.info("Parameter summary (total | trainable):")
        for name, module in parts.items():
            total, trainable = self._count_params(module)
            logger.info(f"  {name}: {total:,d} | {trainable:,d}")
    
    @staticmethod
    def _count_params(module: nn.Module) -> Tuple[int, int]:
        """Return total and trainable parameter counts of a module."""
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _set_trainable_flags(self):
        # 1) 先全部冻结，避免参数引用冲突
        for p in self.parameters():
            p.requires_grad = False

        # 2) SAM -------------------------
        # patch_embed + adapter
        for n, p in self.sam.image_encoder.named_parameters():
            if 'patch_embed.proj' in n or 'prompt_generator' in n:
                p.requires_grad = True
        # prompt_encoder（可选）
        if self.sam.prompt_encoder is not None:
            for n, p in self.sam.prompt_encoder.named_parameters():
                p.requires_grad = True

        # 3) Extra encoder --------------
        for n, p in self.extra_encoder.prompt_generator.named_parameters():
            p.requires_grad = True     # 只开 adapter

        # 4) Fusion & Decoders ----------
        for n, p in self.fusion_neck.named_parameters():
            p.requires_grad = True
        for n, p in self.skip_neck1.named_parameters():
            p.requires_grad = True
        for n, p in self.skip_neck2.named_parameters():
            p.requires_grad = True
        for n, p in self.binary_decoder.named_parameters():
            p.requires_grad = True
        for n, p in self.hv_decoder.named_parameters():
            p.requires_grad = True
        for n, p in self.type_decoder.named_parameters():
            p.requires_grad = True

    def unfreeze_all_parameters(self):
        """
        释放所有参数，用于分阶段训练
        在前几个epoch冻结部分参数后，释放所有参数进行全参数训练
        """
        logger.info("Unfreezing all model parameters for full parameter training...")
        
        # 释放所有参数
        for p in self.parameters():
            p.requires_grad = True
            
        # 统计参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"All parameters unfrozen: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if DEBUG:
            print("\n\n ---------- SAMNuSCNetV231.forward() ---------- ")
            print(f" input images.shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
        B,C,H,W = images.shape
        # 1. SAM features
        with torch.no_grad():
            img_emb = self.sam.image_encoder(images, no_neck=True)
            if DEBUG:
                print(f"\n SAM image encoder output: {img_emb.shape}")
        
        img_emb = img_emb.permute(0,3,1,2).contiguous()
        if DEBUG:
            print(f" SAM image encoder output (permuted) --> {img_emb.shape}")

        # 2. Extra encoder features
        ex_embed, ex_skips = self.extra_encoder(images)
        if DEBUG:
            print(f"\n Extra encoder output: {ex_embed.shape}")
            print(f" Extra encoder skips: {len(ex_skips)}")
            for i,t in enumerate(ex_skips):
                print(f"  {i}: {t.shape}")

        # reshape skips
        for i,t in enumerate(ex_skips):
            b,c0,h0,w0 = t.shape
            oh,ow = self.ex_skip_shapes[i]
            oc = (c0*h0*w0)//(oh*ow)
            ex_skips[i] = t.view(b,oc,oh,ow)
        
        if DEBUG:
            print(f" Extra encoder skips (reshaped) --> {len(ex_skips)}")
            for i,t in enumerate(ex_skips):
                print(f"  {i}: {t.shape}")

        # enhance skips
        ex_skips[1] = self.skip_neck2(ex_skips[1])
        if DEBUG:
            print(f" enhanced ex_skips[1] --> {ex_skips[1].shape}")

        ex_skips[2] = self.skip_neck1(ex_skips[2])
        if DEBUG:
            print(f" enhanced ex_skips[2] --> {ex_skips[2].shape}")
        

        # reshape extra main feature
        if ex_embed.ndim != 4:
            # assume [B,N,C]
            B2,N,C2 = ex_embed.shape
            gs = int(N**0.5)
            ex_embed = ex_embed.view(B2,gs,gs,C2).permute(0,3,1,2)
            if DEBUG:
                print(f" Extra encoder (reshaped) from [{B2},{N},{C2}] --> {ex_embed.shape}")

        assert ex_embed.shape[-2:]==img_emb.shape[-2:], "Feature ex_embed map spatial mismatch with SAM image encoder img_emb"

        if DEBUG:
            print(f" Extra encoder output (reshaped) --> {ex_embed.shape}")

        # 3. Fusion
        if DEBUG:
            print(f"\n Fusion input 1: images.shape = {images.shape}, ")
            print(f" Fusion input 2: img_emb.shape = {img_emb.shape}, ")
            print(f" Fusion input 3: ex_embed.shape = {ex_embed.shape}, ")
        fusion_embed = self.fusion_neck(images, img_emb, ex_embed)
        if DEBUG:
            print(f" Fusion output fusion_embed.shape = {fusion_embed.shape}")

        # 4. Decoding
        bin_map, boundary, bin_feats, bin_aux = self.binary_decoder(fusion_embed, ex_skips)
        if DEBUG:
            print(f"\n Binary decoder output: bin_map.shape = {bin_map.shape}, boundary.shape = {boundary.shape}")
            if bin_aux is not None:
                print(f" Binary decoder aux_outs: {len(bin_aux)}")
                for i,aux in enumerate(bin_aux):
                    print(f"  {i}: {aux.shape}")

        hv_map, hv_aux = self.hv_decoder(fusion_embed, ex_skips, bin_feats)
        if DEBUG:
            print(f"\n HV decoder output: hv_map.shape = {hv_map.shape}")
            if hv_aux is not None:
                print(f" HV decoder aux_outs: {len(hv_aux)}")
                for i,aux in enumerate(hv_aux):
                    print(f"  {i}: {aux.shape}")

        tp_map, tp_aux = self.type_decoder(fusion_embed, ex_skips, bin_feats)
        if DEBUG:
            print(f"\n Type decoder output: tp_map.shape = {tp_map.shape}")
            if tp_aux is not None:
                print(f" Type decoder aux_outs: {len(tp_aux)}")
                for i,aux in enumerate(tp_aux):
                    print(f"  {i}: {aux.shape}")

        out = { 'bin':bin_map, 'boundary':boundary,
                'hv':hv_map, 'tp':tp_map, 'type_aux':tp_aux }
        if self.output_aux_tokens:
            out['bin_aux'] = bin_aux[:-1]
            out['hv_aux']  = hv_aux[:-1]
        return out

# -----------------------
# Test script
# -----------------------
if __name__ == "__main__":
    import torch
    from torchinfo import summary
    from network.get_uni_adapter import get_uni_adapter
    
    # Load extra encoder
    extra_checkpoint = "/root/aMI_DATASET/202307_MedAGI_SAMPath/uni/pytorch_model.bin"
    extra_encoder = get_uni_adapter(extra_checkpoint, neck=False)

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n ---- ✅ Using device: {device}")

    img_size = 128

    model = SAMNuSCNetV231(
        model_type="vit_b",
        checkpoint="/root/aMI_DATASET/202307_MedAGI_SAMPath/sam_vit_b_01ec64.pth",
        num_classes=6,
        extra_encoder=extra_encoder,
        freeze_image_encoder=True,
        input_HW=(img_size, img_size),
        mask_HW=(img_size, img_size),
        fusion_heads=4,
        base_channels=256,
        output_aux_tokens=True
    ).to(device)

    model_name = model.__class__.__name__
    # print(f"\n\n - Model name: {model_name}, Model architecture:")
    # print(model)
    # print(" ----------- end of Model architecture ---------- \n\n")

    # print("\n -------- Summary of trainable SAM model parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f" ✅ {name}")
    # print(" -------- end of trainable parameters ---------- \n")
    
    print("\n -------- Trainable parameters:")
    print_trainable_params(model)
    print(" -------- end of trainable parameters ---------- \n")


    print(f"\n\n - Model name: {model_name}, Model example:")
    # Create dummy input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    print(" -- Input Dummy shape:", dummy_images.shape)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images)
    # print("\n ------------- Forward pass completed.")

    # Print output shapes
    print(f" --- Model outputs.keys(): {outputs.keys()}")
    for k, v in outputs.items():
        if "aux_outs" in k:   # the aux_outs is a list
            print(f" ------ {k}: {len(v)}")
            for i, aux in enumerate(v):
                print(f"  {i}: {aux.shape}, dtype: {aux.dtype}")
        else:
            if isinstance(v, list):
                print(f" ------ {k}: {len(v)}")
                for i, item in enumerate(v):
                    print(f" --{k}[{i}]: {item.shape}, dtype: {item.dtype}")
            else:
                print(f" ------ {k}: v.shape: {v.shape}, dtype: {v.dtype}")
    
    # # Model summary
    print(f"\n\n - Model name: {model_name}, summary (depth=2):")
    # depth=1 shows only top‐level modules
    # summary(model, depth=2)
    model_summary = summary(model, input_size=(2, 3, img_size, img_size), depth=2)

    from thop import profile
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model_name = model.__class__.__name__
    print(f"\n\n - Model name: {model_name}, FLOPs analysis:")
    # # FLOPs 分析
    # flops = FlopCountAnalysis(model, dummy_images)
    # print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # 单位：Giga

    # # 参数量分析
    # print("\n\n - Model parameter count:")
    # print(parameter_count_table(model))

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_images, ))

    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")       # 单位换算为 Giga FLOPs
    print(f"Params: {params / 1e6:.2f} M parameters")

    from network.SAM_NuSCNet_utils import summarize_model_params_auto
    print(f"\n\n - Model name: {model_name}, summarize_model_params_auto():")
    summarize_model_params_auto(model)
    
    print("\n Forward pass completed successfully.\n")
