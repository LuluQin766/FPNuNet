
def get_model(cfg):
    print("\n ---- Loading model...")
    # === Added for SAM-NuSeg support ===
    if cfg.model.extra_encoder is not None:  # cfg.model.extra_encoder  hipt
        print("Using %s as an extra encoder" % cfg.model.extra_encoder)
        neck = True if cfg.model.extra_type == 'plus' else False
        print(" ================= neck: ", neck)
        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)
        elif cfg.model.extra_encoder == 'uni_v1': # sim added
            from network.get_network import get_uni
            extra_encoder = get_uni(cfg.model.extra_checkpoint, neck=neck)
        elif cfg.model.extra_encoder == 'uni_v1_adapter': # sim added
            from network.get_uni_adapter import get_uni_adapter
            extra_encoder = get_uni_adapter(cfg.model.extra_checkpoint, neck=neck)
        else:
            raise NotImplementedError
    else:
        extra_encoder = None

    print("\n\n ======== cfg.model.extra_encoder = ", cfg.model.extra_encoder)
    # print(" ======== extra_encoder: \n", extra_encoder, "\n\n")

    print("Using %s as the model.extra_type: " % cfg.model.extra_type)

    if cfg.model.extra_type in ['multihead_v12']:
        from network.sam_network_multihead_v12s import PromptSAMFusionMultiHeadV12S
        MODEL = PromptSAMFusionMultiHeadV12S
    elif cfg.model.extra_type in ['multihead_v13']:
        from network.sam_network_multihead_v13s import PromptSAMFusionMultiHeadV13S
        MODEL = PromptSAMFusionMultiHeadV13S
    elif cfg.model.extra_type in ['multihead_v14']:
        from network.sam_network_multihead_v14 import PromptSAMFusionMultiHeadV14
        MODEL = PromptSAMFusionMultiHeadV14
    elif cfg.model.extra_type in ['multihead_v15']:
        from network.sam_network_multihead_v15 import PromptSAMFusionMultiHeadV15
        MODEL = PromptSAMFusionMultiHeadV15
    elif cfg.model.extra_type in ['multihead_v16']:
        from network.SAM_NuSCNet_v16 import SAMNuSCNetV16
        MODEL = SAMNuSCNetV16
    elif cfg.model.extra_type in ['multihead_v17']:
        from network.SAM_NuSCNet_v17 import SAMNuSCNetV17
        MODEL = SAMNuSCNetV17
    elif cfg.model.extra_type in ['multihead_v18']:
        from network.SAM_NuSCNet_v18 import SAMNuSCNetV18
        MODEL = SAMNuSCNetV18
    elif cfg.model.extra_type in ['multihead_v19']:
        from network.SAM_NuSCNet_v19 import SAMNuSCNetV19
        MODEL = SAMNuSCNetV19
    elif cfg.model.extra_type in ['multihead_v20']:
        from network.SAM_NuSCNet_v20 import SAMNuSCNetV20
        MODEL = SAMNuSCNetV20
    elif cfg.model.extra_type in ['multihead_v21']:
        from network.SAM_NuSCNet_v21 import SAMNuSCNetV21
        MODEL = SAMNuSCNetV21
    elif cfg.model.extra_type in ['multihead_v22']:
        from network.SAM_NuSCNet_v22 import SAMNuSCNetV22
        MODEL = SAMNuSCNetV22
    elif cfg.model.extra_type in ['multihead_v23']:
        from network.SAM_NuSCNet_v23 import SAMNuSCNetV23
        MODEL = SAMNuSCNetV23
    elif cfg.model.extra_type in ['multihead_v231']:
        from network.SAM_NuSCNet_v231 import SAMNuSCNetV231
        MODEL = SAMNuSCNetV231
    elif cfg.model.extra_type in ['multihead_v231x']:
        from network.SAM_NuSCNet_v231x import SAMNuSCNetV231x
        MODEL = SAMNuSCNetV231x
    elif cfg.model.extra_type in ['multihead_v231heavy']:
        from network.SAM_NuSCNet_v231heavy import SAMNuSCNetV231Heavy
        MODEL = SAMNuSCNetV231Heavy
    elif cfg.model.extra_type in ['multihead_v232']:
        from network.mv232_temp import SAMNuSCNetV232
        MODEL = SAMNuSCNetV232
    elif cfg.model.extra_type in ['multihead_v233']:
        from network.SAM_NuSCNet_v233 import SAMNuSCNetV233
        MODEL = SAMNuSCNetV233
    elif cfg.model.extra_type in ['multihead_v234']:
        from network.SAM_NuSCNet_v234 import SAMNuSCNetV234
        MODEL = SAMNuSCNetV234
    elif cfg.model.extra_type in ['multihead_v234woImageFeats']:
        from network.SAM_NuSCNet_v234woimagefeats import SAMNuSCNetV234woImageFeats
        MODEL = SAMNuSCNetV234woImageFeats
    elif cfg.model.extra_type in ['multihead_v235']:
        from network.SAM_NuSCNet_v235 import SAMNuSCNetV235
        MODEL = SAMNuSCNetV235
    elif cfg.model.extra_type in ['multihead_v236']:
        from network.SAM_NuSCNet_v236 import SAMNuSCNetV236
        MODEL = SAMNuSCNetV236
    elif cfg.model.extra_type in ['multihead_v237']:
        from network.SAM_NuSCNet_v237 import SAMNuSCNetV237
        MODEL = SAMNuSCNetV237
    elif cfg.model.extra_type in ['multihead_v238']:
        from network.SAM_NuSCNet_v238 import SAMNuSCNetV238
        MODEL = SAMNuSCNetV238
    elif cfg.model.extra_type in ['multihead_v238']:
        from network.SAM_NuSCNet_v239 import SAMNuSCNetV239
        MODEL = SAMNuSCNetV239
    elif cfg.model.extra_type in ['multihead_v241']:
        from network.SAM_NuSCNet_v241 import SAMNuSCNetV241
        MODEL = SAMNuSCNetV241
    elif cfg.model.extra_type in ['multihead_v251']:
        from network.SAM_NuSCNet_v251 import SAMNuSCNetV251
        MODEL = SAMNuSCNetV251
    elif cfg.model.extra_type in ['multihead_v252']:
        from network.SAM_NuSCNet_v252 import SAMNuSCNetV252
        MODEL = SAMNuSCNetV252
    elif cfg.model.extra_type in ['multihead_v262']:
        from network.SAM_NuSCNet_v262 import SAMNuSCNetV262
        MODEL = SAMNuSCNetV262
    else:
        print("Model type not supported: %s" % cfg.model.extra_type)
        raise NotImplementedError

    model = MODEL(
        model_type = cfg.model.type,
        checkpoint = cfg.model.checkpoint,
        prompt_dim = cfg.model.prompt_dim,
        num_classes = cfg.dataset.num_classes,
        extra_encoder = extra_encoder,
        freeze_image_encoder = cfg.model.freeze.image_encoder,
        freeze_prompt_encoder = cfg.model.freeze.prompt_encoder,
        freeze_mask_decoder = cfg.model.freeze.mask_decoder,
        input_HW = cfg.dataset.image_hw,
        mask_HW = cfg.dataset.image_hw,
        feature_input = cfg.dataset.feature_input,
        prompt_decoder = cfg.model.prompt_decoder,
        dense_prompt_decoder=cfg.model.dense_prompt_decoder,
        no_sam=cfg.model.no_sam if "no_sam" in cfg.model else None,
        stain_flag=cfg.dataset.stain_flag if "stain_flag" in cfg.dataset else None,
    )
    
    # 输出模型基本信息
    print("\n\n -------- model basic information:")
    print(f" model_type: {cfg.model.type}")
    print(f" checkpoint: {cfg.model.checkpoint}")
    print(f" prompt_dim: {cfg.model.prompt_dim}")
    print(f" num_classes: {cfg.dataset.num_classes}")
    print(f" extra_encoder: {cfg.model.extra_encoder}")
    print(f" freeze_image_encoder: {cfg.model.freeze.image_encoder}")
    print(f" freeze_prompt_encoder: {cfg.model.freeze.prompt_encoder}")
    print(f" freeze_mask_decoder: {cfg.model.freeze.mask_decoder}")
    print(f" input_HW: {cfg.dataset.image_hw}")
    print(f" mask_HW: {cfg.dataset.image_hw}")
    print(f" feature_input: {cfg.dataset.feature_input}")
    print(f" prompt_decoder: {cfg.model.prompt_decoder}")
    print(f" dense_prompt_decoder: {cfg.model.dense_prompt_decoder}")
    print(f" no_sam: {cfg.model.no_sam if 'no_sam' in cfg.model else None}")
    print(f" stain_flag: {cfg.dataset.stain_flag if 'stain_flag' in cfg.dataset else None}")
    print("\n\n")

    return model
