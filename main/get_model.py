
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

    if cfg.model.extra_type in ['multihead_v231']:
        from network.SAM_NuSCNet_v231 import SAMNuSCNetV231
        MODEL = SAMNuSCNetV231
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
