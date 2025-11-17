
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

import re
from collections import defaultdict

def summarize_model_params_auto(model):
    summary = defaultdict(lambda: {'total': 0, 'trainable': 0})

    # 模糊匹配所有 color_fusion_?（支持 color_fusion_sam, color_fusion_uni, color_fusion_image)
    for name, module in model.named_children():
        if re.search(r'color_fusion_?', name, re.IGNORECASE):
            summary[name]['total'] = sum(p.numel() for p in module.parameters())
            summary[name]['trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    # 动态模块识别
    sam_module = getattr(model, 'sam', None) or getattr(model, 'model', None)
    if sam_module is not None:
        summary['sam']['total'] = sum(p.numel() for p in sam_module.parameters())
        summary['sam']['trainable'] = sum(p.numel() for p in sam_module.parameters() if p.requires_grad)

    if hasattr(model, 'extra_encoder'):
        ee = model.extra_encoder
        summary['extra_encoder']['total'] = sum(p.numel() for p in ee.parameters())
        summary['extra_encoder']['trainable'] = sum(p.numel() for p in ee.parameters() if p.requires_grad)

    if hasattr(model, 'fusion_neck'):
        fn = model.fusion_neck
        summary['fusion_neck']['total'] = sum(p.numel() for p in fn.parameters())
        summary['fusion_neck']['trainable'] = sum(p.numel() for p in fn.parameters() if p.requires_grad)

    # 模糊匹配所有 skip_neck（支持 skip_neck1, skip_neck2, uni_skip_neck_1, etc）
    for name, module in model.named_children():
        if re.search(r'skip[_]?neck', name, re.IGNORECASE):
            summary[name]['total'] = sum(p.numel() for p in module.parameters())
            summary[name]['trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    for decoder_name in ['binary_decoder', 'hv_decoder', 'type_decoder']:
        if hasattr(model, decoder_name):
            module = getattr(model, decoder_name)
            summary[decoder_name]['total'] = sum(p.numel() for p in module.parameters())
            summary[decoder_name]['trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    # 汇总
    total_params = sum(v['total'] for v in summary.values())
    trainable_params = sum(v['trainable'] for v in summary.values())

    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\n")

    for module_name, v in summary.items():
        print(f"🧩 {module_name:20s} | Total: {v['total']:,} | Trainable: {v['trainable']:,} "
              f"| {100 * v['trainable'] / v['total']:.2f}% remains trainable. ")

    return summary
