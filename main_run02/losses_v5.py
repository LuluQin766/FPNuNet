# losses_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss

DEBUG = False
# DEBUG = True    # if True, print debug information

# 定义 V14 model 的损失函数
# V14模型的输出为：
# {
#             'bin': bin_map,                   # [B, 1, H, W]
#             'boundary': boundary,             # [B, 1, H, W]
#             'hv': hv_out,                     # [B, 2, H, W]
#             'tp': type_out[-1],               # [B, C, H, W]
#             'bin_aux_outs': bin_aux_outs,     # list, [[B, 1, H/2, W/2], [B, 1, H/4, W/4]]
#             'hv_aux_outs': hv_aux_outs,       # list, [[B, 2, H/2, W/2], [B, 2, H/4, W/4]]
#             'type_aux_outs': type_out         # list, [[B, C, H/2, W/2], [B, C, H/4, W/4]]
#         }
# 其中'bin', 'boundary'是binary_decoder的最终输出，bin_aux_outs则为binary_decoder中间stage输出低分辨率的bin_map，这个中间预测结果将用于进行loss计算，以提升模型的精度。
# 由于'bin', 'boundary'都是binary_decoder，因此采用同一个cof进行loss计算。
# 'hv'、hv_aux_outs和 'tp'、type_aux_outs则分别为hv_decoder和type_decoder的输出和中间输出。

cfg = {
    "loss": {
        "bin": {
            "focal_cof": 1.0,
            "bce_cof": 1.0,
            "dice_cof": 1.0,
            "boun_loss_cof": 0.25,
            "aux_loss_cof": 0.25
        },
        "tp": {
            "focal_cof": 1.0,
            "dice_cof": 1.0,
            "ce_cof": 1.0,
            "iou_cof": 1.0,
            "aux_loss_cof": 0.25,
            "class_weights": None
        },
        "hv": {
            "mse_cof": 1.0,
            "msge_cof": 1.0,
            "aux_loss_cof": 0.25
        }
    }
}

class SAMLossV5(nn.Module):
    def __init__(self,
                 bin_cof: dict = {'focal_cof': 1.0, 'bce_cof': 1.0, 'dice_cof': 1.0, 'boun_loss_cof': 0.25, 'aux_loss_cof': 0.25},
                 tp_cof: dict = {'focal_cof': 1.0, 'dice_cof': 1.0, 'ce_cof': 1.0, 'iou_cof': 1.0, 'aux_loss_cof': 0.25, "class_weights": None},
                 hv_cof: dict = {'mse_cof': 1.0,'msge_cof': 1.0, 'aux_loss_cof': 0.25},
                 compute_aux_loss = False):
        super().__init__()
        self.bin_cof = bin_cof
        self.tp_cof = tp_cof
        self.hv_cof = hv_cof
        self.compute_aux_loss = compute_aux_loss

        # print 所有的cof
        if DEBUG:
            print("\n --- [Loss] V14 Loss Function __init__():")
            for k, v in self.bin_cof.items():
                print(f" --- [Bin] {k}: {v}")
            for k, v in self.tp_cof.items():
                print(f" --- [Type] {k}: {v}")
            for k, v in self.hv_cof.items():
                print(f" --- [HV] {k}: {v}")
        
        # Binary decoder losses, sigmoid input
        self.binary_dice_loss_fn = DiceLoss(sigmoid=True)
        self.binary_focal_loss_fn = FocalLoss(to_onehot_y=False)

        # Type decoder losses, softmax input
        class_weights = tp_cof.get("class_weights", None)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.FloatTensor(class_weights)
        if DEBUG:
            print(f" --- [Type] class_weights: {class_weights}")
        self.type_ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
        self.type_dice_loss_fn = DiceLoss(include_background=False, softmax=True, weight=class_weights[1:] if class_weights is not None else None)
        self.type_focal_loss_fn = FocalLoss(include_background=False, to_onehot_y=False, use_softmax=True,
                                            weight=class_weights[1:] if class_weights is not None else None)
        
    @torch.no_grad()
    def to_one_hot_label(self, targets, num_classes):
        """
        Convert target with shape [B, 1, H, W] to one-hot [B, C, H, W]
        """
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [B, H, W]

        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)                 # → [B, C, H, W]

        return targets_one_hot.float()

    def hv_loss_fn(self, pred_hv, gt_hv, mask):
        """
        pred_hv: [B, 2, H, W]
        gt_hv: [B, 2, H, W]
        mask: [B, H, W] - nuclei region mask
        """
        if DEBUG:
            print("\n ----- hv_loss caculation:")
            print(f" hv_loss input: pred_hv: {pred_hv.shape}, gt_hv: {gt_hv.shape}, mask: {mask.shape}")

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        if DEBUG:
            print(f" hv_loss unsqueeze mask: {mask.shape}")
        mask_expand = torch.cat([mask, mask], dim=1)  # [B, 2, H, W]
        if DEBUG:
            print(f" hv_loss mask_expand cat: {mask_expand.shape}")

        # hv_loss = mse * mse_cof + msge * msge_cof
        mse = F.mse_loss(pred_hv * mask_expand, gt_hv * mask_expand)
        msge = msge_loss(gt_hv, pred_hv, mask_expand)  # 使用原始 HoVerNet 方法

        if DEBUG:
            print(f" hv_loss output: mse: {mse.item():.6f}, msge: {msge.item():.6f}")

        if torch.isnan(mse):
            if DEBUG:
                print("⚠️ NaN in hv_loss: mse !")
            mse = torch.tensor(0.0, device=pred_hv.device, requires_grad=True)

        if torch.isnan(msge):
            if DEBUG:
                print("⚠️ NaN in hv_loss: msge!")
            msge = torch.tensor(0.0, device=pred_hv.device, requires_grad=True)

        loss = mse * self.hv_cof['mse_cof'] + msge * self.hv_cof['msge_cof']
        if DEBUG:
            print(f" hv_loss = {loss.item():.6f}", 
                  f" = mse ({mse.item():.6f}) x {self.hv_cof['mse_cof']:.3f}", 
                  f" + msge ({msge.item():.6f}) x {self.hv_cof['msge_cof']:.3f}")

        return {
            "loss": loss,
            "mse": mse,
            "msge": msge,
        }

    def binary_loss_fn(self, logits, targets):
        if DEBUG:
            print("\n ----- binary_loss_fn")
            print(f" binary_loss_fn input: logits: {logits.shape}, targets: {targets.shape}")

        # 处理logits和目标值
        if targets.shape[1] == 2:
            # 对于2通道的情况，使用softmax
            pred_bin = torch.softmax(logits, dim=1)[:, 1:2, :, :]  # shape: [B, 1, H, W]
            pred_bin = torch.clamp(pred_bin, min=1e-6, max=1 - 1e-6)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1).float()  # [B, 1, H, W]
            else:
                targets = targets.float()  # [B, 1, H, W]
            # 对于softmax输出，使用binary_cross_entropy
            bce = F.binary_cross_entropy(pred_bin, targets)
        else:
            # 对于单通道的情况，使用logits直接计算BCE（AMP安全）
            if targets.dim() == 3:
                targets = targets.unsqueeze(1).float()  # [B, 1, H, W]
            else:
                targets = targets.float()  # [B, 1, H, W]
            # 使用binary_cross_entropy_with_logits，AMP安全
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            # 为了dice和focal loss，仍然需要sigmoid输出
            pred_bin = torch.sigmoid(logits)  # shape: [B, 1, H, W]
        
        if DEBUG:
            print(f" binary_loss_fn output: pred_bin: {pred_bin.shape}, value range: {pred_bin.min()} ~ {pred_bin.max()}")
            print(f" binary_loss_fn targets: {targets.shape}, value range: {targets.min()} ~ {targets.max()}")
        dice = self.binary_dice_loss_fn(pred_bin, targets)
        focal = self.binary_focal_loss_fn(logits, targets)

        if DEBUG:
            print(f" binary_loss_fn output: bce: {bce.item():.6f}, dice: {dice.item():.6f}, focal: {focal.item():.6f}")

        if torch.isnan(bce):
            if DEBUG:
                print("⚠️ NaN in binary_loss_fn: bce!")
            bce = torch.tensor(0.0, device=logits.device, requires_grad=True)

        if torch.isnan(dice):
            if DEBUG:
                print("⚠️ NaN in binary_loss: dice!")
            dice = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        if torch.isnan(focal):
            if DEBUG:
                print("⚠️ NaN in binary_loss: focal!")
            focal = torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = bce * self.bin_cof['bce_cof'] + dice * self.bin_cof['dice_cof'] + self.bin_cof['focal_cof'] * focal
        if DEBUG:
            print(f" binary_loss = {loss.item():.6f}", 
                  f" = bce ({bce.item():.6f}) x {self.bin_cof['bce_cof']:.3f}", 
                  f" + dice ({dice.item():.6f}) x {self.bin_cof['dice_cof']:.3f}",
                  f" + focal ({focal.item():.6f}) x {self.bin_cof['focal_cof']:.3f}")

        return {
            "loss": loss,
            "bce": bce,
            "dice": dice,
            "focal": focal
        }

    def type_loss_fn(self, logits, targets):
        """
            logits: [B, C, H, W], C is number of classes including background
            targets: [B, H, W] with class indices (0 for background)
        """
        if DEBUG:
            print("\n ----- type_loss_fn")
            print(f" type_loss_fn input: logits: {logits.shape}, targets: {targets.shape}")
            pred_type_map = torch.argmax(logits, dim=1)  # [B,H,W]
            print(f" type_loss_fn pred_type_map: {pred_type_map.shape}, value range: {pred_type_map.min()} ~ {pred_type_map.max()}, unique: {torch.unique(pred_type_map)}")
            print(f" type_loss_fn targets: {targets.shape}, value range: {targets.min()} ~ {targets.max()}, unique: {torch.unique(targets)}")

        # === Step 1: 确保 targets 是 LongTensor 并去除通道维度 ===
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [B, H, W]
        targets = targets.long()

        num_classes = logits.shape[1]
        assert targets.max() < num_classes, f"Target has out-of-range value: max={targets.max()} >= {num_classes}"
        if DEBUG:
            print(f" --- num_classes: {num_classes}")

        # === Create valid mask to exclude ignore_index pixels ===
        valid_mask = (targets != 0)  # 0 is background
        if valid_mask.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=logits.device, requires_grad=True),
                "ce": torch.tensor(0.0, device=logits.device),
                "dice": torch.tensor(0.0, device=logits.device),
                "iou": torch.tensor(0.0, device=logits.device),
            }

        # === Step 1: CrossEntropy Loss ===
        try:
            ce = self.type_ce_loss_fn(logits, targets)
        except Exception as e:
            if DEBUG:
                print(" CrossEntropy error:", e)
            ce = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # === Step 2: Get probabilities and one-hot target ===
        probs = torch.softmax(logits, dim=1)
        one_hot_targets = self.to_one_hot_label(targets.unsqueeze(1), num_classes=num_classes)  # [B, C, H, W]
        if DEBUG:
            print(f" GT ont-hot target: {one_hot_targets.shape}, probs: {probs.shape}")

        # === Step 3: Dice Loss (probability-level) ===
        try:
            dice = self.type_dice_loss_fn(probs, one_hot_targets)
        except Exception as e:
            if DEBUG:
                print(" DiceLoss error:", e)
            dice = torch.tensor(0.0, device=logits.device)

        # === Step 4: IoU Loss (1 - mean IoU over valid classes) ===
        try:
            pred_mask = (probs >= 0.5).float()
            intersection = torch.sum(pred_mask * one_hot_targets, dim=(2, 3))
            union = torch.sum(pred_mask + one_hot_targets, dim=(2, 3)) - intersection
            iou_per_class = intersection / (union + 1e-6)
            mean_iou = iou_per_class[:, 1:].mean()  # exclude background
            iou_loss = 1.0 - mean_iou
        except Exception as e:
            if DEBUG:
                print(" ❎ IoU loss error:", e)
            iou_loss = torch.tensor(0.0, device=logits.device)
        
        # === Step 5: Focal Loss (optional) ===
        try:
            focal = self.type_focal_loss_fn(logits, one_hot_targets)
        except Exception as e:
            if DEBUG:
                print(" ❎ Focal loss error:", e)
            focal = torch.tensor(0.0, device=logits.device)
        
        # === Combine Losses ===
        loss = (
            ce * self.tp_cof.get("ce_cof", 1.0)
            + dice * self.tp_cof.get("dice_cof", 1.0)
            + focal * self.tp_cof.get("focal_cof", 0.0)
            + iou_loss * self.tp_cof.get("iou_cof", 1.0)
        )
        
        if DEBUG:
            print(f" type_loss = {loss.item():.6f}",
                  f" = ce ({ce.item():.6f}) x {self.tp_cof['ce_cof']:.3f}", 
                  f" + dice ({dice.item():.6f}) x {self.tp_cof['dice_cof']:.3f}",
                  f" + focal ({focal.item():.6f}) x {self.tp_cof['focal_cof']:.3f}", 
                  f" + iou ({iou_loss.item():.6f}) x {self.tp_cof['iou_cof']:.3f}")

        return {
            "loss": loss,
            "ce": ce,
            "dice": dice,
            "focal": focal,
            "iou": iou_loss,
        }
    
    def calc_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, class_weights=None):
        pred_mask = (pred_mask >= 0.5).float()
        intersection = torch.sum(pred_mask * gt_mask, dim=(2, 3))
        union = pred_mask.sum((2, 3)) + gt_mask.sum((2, 3)) - intersection
        iou = intersection / (union + 1e-7)
        if class_weights is not None:
            iou_loss = F.mse_loss(iou[:, 1:] * class_weights[1:], torch.ones_like(iou[:, 1:]) * class_weights[1:])
        else:
            iou_loss = F.mse_loss(iou[:, 1:], torch.ones_like(iou[:, 1:]))
        return iou_loss

    def forward(self, outputs, targets, ignored_masks=None):
        """
        outputs and targets are dicts as follows:
        {
            'bin',            # [B, 1, H, W]
            'boundary',       # [B, 1, H, W]
            'hv',             # [B, 2, H, W]
            'tp',             # [B, H, W]
            'bin_aux_outs',   # List, [[B, 1, H/2, W/2], [B, 1, H/4, W/4]]
            'hv_aux_outs',    # List, [[B, 2, H/2, W/2], [B, 2, H/4, W/4]]
            'type_aux_outs'   # List, [[B, C, H/2, W/2], [B, C, H/4, W/4]]
        }
        ignored_masks: optional Tensor [B, 1, H, W]
        """
        loss_dict = {}

        if DEBUG:
            print("\n===== SAMLossV4 forward =====")
            print(f"\n ----- outputs: {outputs.keys()}")
            for k, v in outputs.items():
                if isinstance(v, list):  # List of np.ndarrays
                    for i, item in enumerate(v):
                        print(f" --- {k}_{i}: {item.shape}")
                else:   # np.ndarray
                    print(f" --- {k}: {v.shape}")
            
            print(f"\n ----- targets: {targets.keys()}")
            for k, v in targets.items():
                if isinstance(v, list):  # List of np.ndarrays
                    for i, item in enumerate(v):
                        print(f" --- {k}_{i}: {item.shape}")
                else:   # np.ndarray
                    print(f" --- {k}: {v.shape}")

            print(f" ignored_masks: {ignored_masks.shape if ignored_masks is not None else None}\n")

        # === Calculate HV loss ===
        ig_mask = (targets['bin'] > 0).float()  # nuclei region mask, [B, H, W]
        pred_hv = outputs['hv']  # [B, 2, H, W]
        if torch.isnan(pred_hv).any():
            if DEBUG:
                print("⚠️ NaN in pred_hv!")
            pred_hv = torch.zeros_like(pred_hv)
        if torch.isinf(pred_hv).any():
            if DEBUG:
                print("⚠️ Inf in pred_hv!")
            pred_hv = torch.zeros_like(pred_hv)
        loss_dict['hv'] = self.hv_loss_fn(outputs['hv'], targets['hv'], ig_mask)
        if DEBUG:
            print(f" ----- hv loss_dict['hv'] = ", loss_dict['hv'])

        loss_dict['hv_loss'] = loss_dict['hv']['loss']

        if self.compute_aux_loss:
            for i, aux_out in enumerate(outputs['hv_aux_outs']):
                aux_gt = targets['hv_aux_outs'][i]  # [B, 2, H/2, W/2]
                ig_mask = (targets['bin_aux_outs'][i] > 0).float()
                if torch.isnan(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ NaN in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                if torch.isinf(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ Inf in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                loss_dict[f'hv_aux_{i}'] = self.hv_loss_fn(aux_out, aux_gt, ig_mask)
                if DEBUG:
                    print(f" ----- hv loss_dict['hv_aux_{i}'] = ", loss_dict[f'hv_aux_{i}'])
            loss_dict['hv_loss'] += sum(loss_dict[f'hv_aux_{i}']['loss'] * self.hv_cof['aux_loss_cof'] for i in range(len(outputs['hv_aux_outs'])))
            
        if DEBUG:
            print(f" ----- hv loss_dict['hv_loss'] = {loss_dict['hv_loss']}\n")

        # === Calculate Binary loss ===
        pred_bin = outputs['bin']
        if torch.isnan(pred_bin).any():
            if DEBUG:
                print("⚠️ NaN in pred_bin!")
            pred_bin = torch.zeros_like(pred_bin)
        if torch.isinf(pred_bin).any():
            if DEBUG:
                print("⚠️ Inf in pred_bin!")
            pred_bin = torch.zeros_like(pred_bin)
        loss_dict['bin'] = self.binary_loss_fn(pred_bin, targets['bin'])
        if DEBUG:
            print(f" ----- bin loss_dict['bin'] = ", loss_dict['bin'])
        
        pred_boun = outputs['boundary']
        if torch.isnan(pred_boun).any():
            if DEBUG:
                print("⚠️ NaN in pred_boun!")
            pred_boun = torch.zeros_like(pred_boun)
        if torch.isinf(pred_boun).any():
            if DEBUG:
                print("⚠️ Inf in pred_boun!")
            pred_boun = torch.zeros_like(pred_boun)
        loss_dict['boundary'] = self.binary_loss_fn(pred_boun, targets['boundary'])
        if DEBUG:
            print(f" ----- bin loss_dict['boundary'] = ", loss_dict['boundary'])
        
        loss_dict['bin_loss'] = loss_dict['bin']['loss'] + loss_dict['boundary']['loss'] * self.bin_cof['boun_loss_cof'] 

        if self.compute_aux_loss:
            for i, aux_out in enumerate(outputs['bin_aux_outs']):
                aux_gt = targets['bin_aux_outs'][i]  # [B, 1, H/2, W/2]
                if torch.isnan(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ NaN in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                if torch.isinf(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ Inf in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                loss_dict[f'bin_aux_{i}'] = self.binary_loss_fn(aux_out, aux_gt)
                if DEBUG:
                    print(f" ----- bin loss_dict['bin_aux_{i}'] = ", loss_dict[f'bin_aux_{i}'])
            
            loss_dict['bin_loss'] += sum(loss_dict[f'bin_aux_{i}']['loss'] * self.bin_cof['aux_loss_cof'] for i in range(len(outputs['bin_aux_outs'])))
        
        if DEBUG:
            print(f" ----- bin loss_dict['bin_loss'] = {loss_dict['bin_loss']}\n")

        # === Calculate Type loss ===
        pred_tp = outputs['tp']  # [B, H, W]
        if torch.isnan(pred_tp).any():
            if DEBUG:
                print("⚠️ NaN in pred_tp!")
            pred_tp = torch.zeros_like(pred_tp)
        if torch.isinf(pred_tp).any():
            if DEBUG:
                print("⚠️ Inf in pred_tp!")
            pred_tp = torch.zeros_like(pred_tp)
        loss_dict['tp'] = self.type_loss_fn(outputs['tp'], targets['tp'])
        if DEBUG:
            print(f" ----- tp loss_dict['tp'] = ", loss_dict['tp'])
        
        loss_dict['tp_loss'] = loss_dict['tp']['loss']
        
        if self.compute_aux_loss:
            for i, aux_out in enumerate(outputs['type_aux_outs']):
                aux_gt = targets['type_aux_outs'][i]  # [B, C, H/2, W/2]
                if torch.isnan(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ NaN in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                if torch.isinf(aux_out).any():
                    if DEBUG:
                        print(f"⚠️ Inf in aux_out_{i}!")
                    aux_out = torch.zeros_like(aux_out)
                loss_dict[f'tp_aux_{i}'] = self.type_loss_fn(aux_out, aux_gt)
                if DEBUG:
                    print(f" ----- tp loss_dict['tp_aux_{i}'] = ", loss_dict[f'tp_aux_{i}'])
            loss_dict['tp_loss'] += sum(loss_dict[f'tp_aux_{i}']['loss'] * self.tp_cof['aux_loss_cof'] for i in range(len(outputs['type_aux_outs'])))
        
        if DEBUG:
            print(f" ----- tp loss_dict['tp_loss'] = {loss_dict['tp_loss']}\n")

        if DEBUG:
            print(f"\n ----- loss_dict: {loss_dict.keys()}")
            for k, v in loss_dict.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f" --- {k}_{k2}: {v2.item():.6f}")
                else:
                    print(f" --- {k}: {v.item():.6f}")
            print(f"\n")

        # === Total ===        
        loss_dict['total_loss'] = loss_dict['hv_loss'] + loss_dict['bin_loss'] + loss_dict['tp_loss']
        
        return loss_dict

def msge_loss(true, pred, focus):
    """
    Mean Squared Gradient Error (MSGE) for HV maps

    Args:
        true: [B, 2, H, W]
        pred: [B, 2, H, W]
        focus: [B, 2, H, W], nuclei region mask
    """
    def get_sobel_kernel(size):
        assert size % 2 == 1, f"Kernel size must be odd, got {size}"
        h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=true.device)
        v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=true.device)
        h, v = torch.meshgrid(h_range, v_range, indexing='ij')
        kernel_h = h / (h ** 2 + v ** 2 + 1e-15)
        kernel_v = v / (h ** 2 + v ** 2 + 1e-15)
        return kernel_h.view(1, 1, 5, 5), kernel_v.view(1, 1, 5, 5)

    def get_gradient_hv(hv):
        kernel_h, kernel_v = get_sobel_kernel(5)
        h_ch = hv[:, 0:1, :, :]
        v_ch = hv[:, 1:2, :, :]
        h_dh = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv = F.conv2d(v_ch, kernel_v, padding=2)
        return torch.cat([h_dh, v_dv], dim=1).permute(0, 2, 3, 1).contiguous()

    focus = focus.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2]
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    # print(f"\n ----- true_grad: {true_grad.shape}, pred_grad: {pred_grad.shape}, focus: {focus.shape}")
    # temp = (pred_grad - true_grad)
    # print(f" ----- temp: {temp.shape}, temp ** 2: {(temp ** 2).shape}")
    
    loss = (pred_grad - true_grad) ** 2 * focus
    return loss.sum() / (focus.sum() + 1e-8)

    
# def test_sam_loss_v4():
#     # ==== Step 1: 构造模拟输出与标签 ====
#     B, C, H, W = 2, 5, 256, 256  # BatchSize, NumClasses, Height, Width

#     print(f"\n===== Test SAM Loss V4 =====")
#     print(f" Shape: (B, C, H, W) = ({B}, {C}, {H}, {W})")

#     outputs = {
#         'bin': torch.randn(B, 1, H, W),
#         'boundary': torch.randn(B, 1, H, W),
#         'bin_aux_outs': [torch.randn(B, 1, H//2, W//2), torch.randn(B, 1, H//4, W//4)],
#         'hv': torch.randn(B, 2, H, W),
#         'hv_aux_outs': [torch.randn(B, 2, H//2, W//2), torch.randn(B, 2, H//4, W//4)],
#         'tp': torch.randn(B, C, H, W),
#         'type_aux_outs': [
#             torch.randn(B, C, H//2, W//2),
#             torch.randn(B, C, H//4, W//4),
#         ],
#     }

#     print(f"\n --------- outputs: ")
#     for k, v in outputs.items():
#         if isinstance(v, list):  # List of tensor
#             print(f" -------- {k}: {len(v)}")
#             for i, item in enumerate(v):
#                 print(f" -- {k}_{i}: {item.shape}, dtype={item.dtype}")
#         else:   # tensor
#             print(f" ----- {k}: {v.shape}, dtype={v.dtype}")

#     targets = {
#         'bin': torch.randint(0, 2, (B, H, W)).float(),
#         'boundary': torch.randint(0, 2, (B, H, W)).float(),
#         'hv': torch.randn(B, 2, H, W),  # 假设有 HV map GT
#         'tp': torch.randint(0, C, (B, H, W)),  # 分类标签
#         'bin_aux_outs': [torch.randint(0, 2, (B, H//2, W//2)).float(), torch.randint(0, 2, (B, H//4, W//4)).float()],
#         'hv_aux_outs': [torch.randn(B, 2, H//2, W//2), torch.randn(B, 2, H//4, W//4)],
#         'type_aux_outs': [
#             torch.randint(0, C, (B, H//2, W//2)),
#             torch.randint(0, C, (B, H//4, W//4)),
#         ],
#     }

#     print(f"\n --------- targets: ")
#     for k, v in targets.items():
#         if isinstance(v, list):
#             print(f" -------- {k}: {len(v)}")
#             for i, vi in enumerate(v):
#                 print(f" -- {k}_{i}: {vi.shape}, dtype={vi.dtype}")
#         else:   # tensor
#             print(f" ----- {k}: {v.shape}, dtype={v.dtype}")

#     # ==== Step 2: 创建损失函数实例 ====
#     cfg['loss']['tp']['class_weights'] = [0, 1.0, 1.0, 0.5, 0.5]  # 假设有 4 类, 背景为 0, 一共 5 类
#     print(f"\n===== Loss Config =====")
#     for key, value in cfg['loss'].items():
#         if isinstance(value, dict):
#             print(f"{key.upper()}:")
#             for k, v in value.items():
#                 print(f"  {k}: {v}")
#         else:
#             print(f"{key}: {value}")
#     print(f"\n")

#     loss_fn = SAMLossV4(
#         bin_cof=cfg['loss']['bin'],
#         tp_cof=cfg['loss']['tp'],
#         hv_cof=cfg['loss']['hv']
#     )

#     # ==== Step 3: 计算损失 ====
#     loss_dict = loss_fn(outputs, targets)

#     # ==== Step 4: 打印结果 ====
#     print("\n===== SAMLossV4 Breakdown =====")
#     for key, value in loss_dict.items():
#         if isinstance(value, dict):
#             print(f" -- {key.upper()} is a Dict: ")
#             for k, v in value.items():
#                 print(f"  {k}: {v.item():.6f}" if isinstance(v, torch.Tensor) else f"  {k}: {v}")
#         else:
#             print(f" -- {key}: {value.item():.6f}, value.dtype={value.dtype} ")

# if __name__ == "__main__":
#     test_sam_loss_v4()

# ===== SAMLossV4 Breakdown =====
# HV:
#   loss: 15.166027
#   mse: 1.005585
#   msge: 9.133934
# HV_AUX_0:
#   loss: 10.200824
#   mse: 0.996127
#   msge: 9.204697
# HV_AUX_1:
#   loss: 9.905205
#   mse: 1.012114
#   msge: 8.893091
# hv_loss: 15.166027

# BIN:
#   loss: 1.596816
#   bce: 0.805574
#   dice: 0.445395
#   focal: 0.345848
# BOUNDARY:
#   loss: 1.598749
#   bce: 0.806263
#   dice: 0.445895
#   focal: 0.346592
# BIN_AUX_0:
#   loss: 1.612375
#   bce: 0.812195
#   dice: 0.448990
#   focal: 0.351190
# BIN_AUX_1:
#   loss: 1.600587
#   bce: 0.804865
#   dice: 0.450011
#   focal: 0.345712
# bin_loss: 2.799744

# TP:
#   loss: 5.495571
#   ce: 1.976567
#   dice: 0.600025
#   focal: 0.187109
#   iou: 0.898642
# TP_AUX_0:
#   loss: 3.664383
#   ce: 1.977769
#   dice: 0.600318
#   focal: 0.187193
#   iou: 0.899103
# TP_AUX_1:
#   loss: 3.668529
#   ce: 1.979807
#   dice: 0.599341
#   focal: 0.189607
#   iou: 0.899774
# tp_loss: 5.495571

# total_loss: 23.461342