import os
import json
import gc
import random
import numpy as np
import cv2
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import le, nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter

from losses_v5 import SAMLossV5 as SAM_Loss
from misc.viz_utils import visualize_maps_batch_train

DEBUG = False
# DEBUG = True    # True for printing debug messages

# 获取当前文件名
file_name = os.path.basename(__file__)
if "debug" in file_name:
    DEBUG = True

if DEBUG:
    print(f"\n ------ Debug mode is {DEBUG}  ✅ , setting on {file_name} ------ \n")

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "valid"
    elif dataloader_idx == 1:
        return "test"
    else:
        raise NotImplementedError

def check_training_inputs(images, bin_map, inst_map, type_map, hv_map, patch_id):
    print(f"\n🟡 Checking training batch {patch_id}")
    
    def check_tensor(name, tensor):
        print(f"\n -----  Checking {name}, dtype={tensor.dtype}, device={tensor.device}")
        if tensor is None:
            print(f"  ❌ {name} is None")
            return
        if torch.isnan(tensor).any():
            print(f"  ❌ {name} contains NaNs")
        if torch.isinf(tensor).any():
            print(f"  ❌ {name} contains Infs")
        unique_vals = torch.unique(tensor)
        print(f"  ✅ {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, unique={unique_vals[:8].tolist()}")

        if unique_vals.numel() > 8:
            print(f"    ... total unique: {unique_vals.numel()}")
        if name == 'bin_map' and not ((unique_vals == 0) | (unique_vals == 1)).all():
            print(f"  ⚠️ Warning: {name} contains values outside [0, 1]")

    check_tensor("images", images)
    check_tensor("bin_map", bin_map)
    check_tensor("inst_map", inst_map)
    check_tensor("type_map", type_map)
    # check_tensor("hv_map", hv_map)

    # Extra: check range of hv_map (should be reasonably bounded)
    hv_max = hv_map.max().item()
    hv_min = hv_map.min().item()
    if abs(hv_max) > 100 or abs(hv_min) > 100:
        print(f"  ⚠️ HV map values out of expected range: min={hv_min}, max={hv_max}")

def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n ✅ Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

class SamSegMultiHeadV6(LightningModule):
    def __init__(
            self,
            cfg,
            sam_model: nn.Module,
            metrics: MetricCollection,
            num_classes: int,
            lr: float = 1e-4,
            weight_decay: float = 1e-2,
            lr_steps: list = (10, 20),
            warmup_steps: int = 0,
            ignored_index=None,
            output_aux_tokens = False,
    ):
        """
        Args:
            cfg: configuration object containing paths and logging settings.
            sam_model (nn.Module): A multi-output SAM model that returns six outputs:
                (pred_masks_semantic, pred_masks_type, pred_masks_boundary,
                 ious_semantic, ious_type, ious_boundary).
            metrics (MetricCollection): A collection of metrics.
            num_classes (int): Number of segmentation classes.
            focal_cof, dice_cof, iou_cof, ce_cof: Loss coefficients.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            lr_steps (list): Learning rate schedule steps.
            warmup_steps (int): Warmup steps.
            ignored_index: Class index to ignore (if any).
        """
        super().__init__()
        # save hyperparams except large objects
        self.ignored_index = ignored_index
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        
        # 分阶段训练参数
        self.unfreeze_epoch = cfg.opt.unfreeze_epoch if "unfreeze_epoch" in cfg.opt else None
        self.parameters_unfrozen = False  # 标记参数是否已经释放
        
        self.save_hyperparameters(ignore=["sam_model", "metrics"])
        self.model = sam_model
        self.output_aux_tokens = output_aux_tokens

        # the model outputs:
        # {
        #     'bin': bin_map,
        #     'boundary': boundary,
        #     'hv': hv_out,
        #     'tp': type_out[-1],
        #     'bin_aux_outs': bin_aux_outs,
        #     'hv_aux_outs': hv_aux_outs,
        #     'type_aux_outs': type_out
        # }
        # 其中 aux_outs 是各个decoder的辅助输出，用于进行损失计算

        print_trainable_params(self.model)
        
        # 打印分阶段训练配置
        if self.unfreeze_epoch is not None:
            print(f"\n🎯 分阶段训练配置:")
            print(f"   - 前 {self.unfreeze_epoch} 个epoch: 冻结部分参数训练")
            print(f"   - 第 {self.unfreeze_epoch} 个epoch开始: 释放所有参数训练")
        else:
            print(f"\n📝 常规训练模式: 使用配置文件中的参数冻结设置")
        
        self.loss = SAM_Loss(
            bin_cof=cfg.loss.bin,
            tp_cof=cfg.loss.tp,
            hv_cof=cfg.loss.hv
        )

        # Suppose num_classes = number of cell types (excluding background)
        self.num_classes = num_classes + 1  # include background class

        print(" self.num_classes = ", self.num_classes)
        
        # metrics dict of metrics, each metric is a pytorch-lightning metric object
        # keys: 'bin', 'boundary', 'tp', 'hv'
        # each metric object is a MetricCollection object with a list of metrics including 'iou_micro', 'dice_macro', 'f1_macro'

        self.train_metrics = nn.ModuleDict({
            head: metrics[head].clone() for head in metrics
        })
        self.valid_metrics = nn.ModuleList([
            nn.ModuleDict({
                head: metrics[head].clone() for head in metrics
            }), 
            nn.ModuleDict({
                head: metrics[head].clone() for head in metrics
            })
        ])
        self.test_metrics = nn.ModuleDict({
            head: metrics[head].clone() for head in metrics
        })

        print("\n --------- SAMSegMultiHeadV4 initialized metrices: ")
        print("\n Train metrics: ", self.train_metrics)
        print("\n Valid metrics: ", self.valid_metrics)
        print("\n Test metrics: ", self.test_metrics)
        print("\n")

        # optimization params
        self.lr = lr
        self.lr_steps = lr_steps
        self.hparams.lr_steps = lr_steps if lr_steps is not None else [10000, 20000]  # default steps if None
        
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # logging setup
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(cfg.log_dir, 'training_log_new.json')
        self.plot_save_dir = os.path.join(cfg.log_dir, 'training_plots')
        os.makedirs(self.plot_save_dir, exist_ok=True)
        
        # TensorBoard setup
        self.tensorboard_dir = os.path.join(cfg.log_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # 记录频率控制
        self.log_frequency = 20  # 每20步记录一次loss和metrics
        self.image_log_frequency = 200  # 每200步记录一次图像
        self.save_log_frequency = 100  # 每100步保存一次日志到磁盘
        self.log_buffer_size = 0  # 日志缓冲区大小计数器
        
        # 打印日志记录频率配置
        print(f"\n📊 日志记录频率配置:")
        print(f"   - Loss记录频率: 每{self.log_frequency}步")
        print(f"   - 图像记录频率: 每{self.image_log_frequency}步")
        print(f"   - 磁盘保存频率: 每{self.save_log_frequency}步")
        # self._init_log()
        # initialize structured log data for multi-head losses and metrics
        
        def get_loss_units(head):
            if head == 'bin':
                return {'loss':[], 'bce':[], 'dice':[], 'focal':[]}
            elif head == 'hv':
                return {'loss':[], 'mse':[], 'msge':[]}
            else:
                return {'loss':[], 'ce':[], 'dice':[], 'focal':[], 'iou':[]}    # tp

        def get_metric_units(head, num_classes=None):
            # print(f"\n -------- get_metric_units for {head} with num_classes={num_classes}")
            if head == 'cls':
                metric_dict = {
                    'iou_micro': [],
                    'dice_macro': [],
                    'f1_macro': [],
                }
                # 自动添加 class-wise
                if num_classes is not None:
                    for i in range(1, num_classes):  # skip background (0)
                        metric_dict[f'iou_class_{i}'] = []
                        metric_dict[f'dice_class_{i}'] = []
                return metric_dict
            else:   # hv
                return {'mse':[],'mae':[], "msge": []}
        
        def get_loss_items_with_aux():
            return {
            "losses": {
                "bin": {
                    "bin": get_loss_units('bin'), 
                    "boundary": get_loss_units('bin'), 
                    "bin_aux_0": get_loss_units('bin'), 
                    "bin_aux_1": get_loss_units('bin'), 
                    "bin_loss": []
                },
                "hv": {
                    "hv": get_loss_units('hv'), 
                    "hv_aux_0": get_loss_units('hv'), 
                    "hv_aux_1": get_loss_units('hv'), 
                    "hv_loss": []
                },
                "tp": {
                    "tp": get_loss_units('tp'), 
                    "tp_aux_0": get_loss_units('tp', num_classes=self.num_classes), 
                    "tp_aux_1": get_loss_units('tp', num_classes=self.num_classes), 
                    "tp_aux_2": get_loss_units('tp', num_classes=self.num_classes), 
                    "tp_loss": []
                },
                "total_loss": []
            },
            "metrics": {
                "bin": {
                    "bin": get_metric_units('cls'),
                    "bin_aux_0": get_metric_units('cls'),
                    "bin_aux_1": get_metric_units('cls'),
                },
                "boundary": get_metric_units('cls'),
                "tp": {
                    "tp": get_metric_units('cls', num_classes=self.num_classes),
                    "tp_aux_0": get_metric_units('cls', num_classes=self.num_classes),
                    "tp_aux_1": get_metric_units('cls', num_classes=self.num_classes),
                    "tp_aux_2": get_metric_units('cls', num_classes=self.num_classes),
                },
                "hv": {
                    "hv": get_metric_units('hv'),
                    "hv_aux_0": get_metric_units('hv'),
                    "hv_aux_1": get_metric_units('hv'),
                },
            },
            "epochs": []
        }
        
        def get_loss_items():
            return {
            "losses": {
                "bin": {
                    "bin": get_loss_units('bin'), 
                    "boundary": get_loss_units('bin'), 
                    "bin_loss": []
                },
                "hv": {
                    "hv": get_loss_units('hv'), 
                    "hv_loss": []
                },
                "tp": {
                    "tp": get_loss_units('tp'), 
                    "tp_loss": []
                },
                "total_loss": []
            },
            "metrics": {
                "bin": {
                    "bin": get_metric_units('cls'),
                },
                "boundary": get_metric_units('cls'),
                "tp": {
                    "tp": get_metric_units('cls', num_classes=self.num_classes),
                },
                "hv": {
                    "hv": get_metric_units('hv'),
                },
            },
            "epochs": []
        }

        if self.output_aux_tokens:
            self.log_data = {
                "train": get_loss_items_with_aux(),
                "valid": get_loss_items_with_aux(),
                "test": get_loss_items_with_aux(),
                "train_epoch": get_loss_items_with_aux(),
                "valid_epoch": get_loss_items_with_aux(),
                "test_epoch": get_loss_items_with_aux()
            }
        else:
            self.log_data = {
                "train": get_loss_items(),
                "valid": get_loss_items(),
                "test": get_loss_items(),
                "train_epoch": get_loss_items(),
                "valid_epoch": get_loss_items(),
                "test_epoch": get_loss_items()
            }
        # # 初始化 log_data 结构（例如 train_epoch）
        # for split in ["train", "valid", "test"]:
        #     for epoch in ["", "_epoch"]:
        #         key = split + epoch
        #         if key not in self.log_data:
        #             self.log_data[key] = {"metrics": {}}
        #         for head in ["bin", "boundary", "tp", "hv"]:
        #             if head not in self.log_data[key]["metrics"]:
        #                 self.log_data[key]["metrics"][head] = {head: {}}

        self._save_log()

    def _save_log(self):
        """优化的日志保存方法，使用更高效的写入策略"""
        try:
            # 使用临时文件避免写入过程中的数据损坏
            temp_file_path = self.log_file_path + '.tmp'
            with open(temp_file_path, 'w') as f:
                json.dump(self.log_data, f, indent=4)
            
            # 原子性重命名，确保数据完整性
            import shutil
            shutil.move(temp_file_path, self.log_file_path)
            
        except Exception as e:
            print(f"Error saving log file to {self.log_file_path}: {e}")
            pass
    
    def _force_save_log(self):
        """强制保存日志，用于训练结束时确保数据不丢失"""
        self._save_log()
        print(f"💾 强制保存日志到: {self.log_file_path}")
    
    def _log_performance_stats(self):
        """记录性能统计信息"""
        if hasattr(self, 'log_buffer_size') and self.log_buffer_size > 0:
            avg_save_frequency = self.log_buffer_size / max(1, self.current_epoch + 1)
            print(f"📊 日志性能统计:")
            print(f"   - 总步数: {self.log_buffer_size}")
            print(f"   - 平均每epoch保存次数: {avg_save_frequency:.1f}")
            print(f"   - 磁盘写入优化: 每{self.save_log_frequency}步保存一次")
            print(f"   - 图像记录频率: 每{self.image_log_frequency}步记录一次")

    def forward(self, images):
        # returns dict: {'bin':B×1×H×W, 'tp':B×C×H×W, 'hv':B×2×H×W, 'aux_preds':[...]} 
        outputs = self.model(images)
        # if DEBUG: 
        #     print("\n -------- Forward pass complete ")
        #     self.print_dict(outputs)
        return outputs

    @torch.no_grad()
    def process_masks(self, gt_masks):
        # gt_masks: [B,H,W]
        ignored = (gt_masks == 0).unsqueeze(1).long()
        return gt_masks.long(), ignored
    
    def predict_mask_process(self, preds: dict):
        """
        Convert model outputs into predicted masks for each head.
        """
        # Binary head
        def bin_sigmoid_threshold(logit, threshold=0.5):
            logit = torch.sigmoid(logit)  # [B, 1, H, W]
            return (logit > threshold).long().squeeze(1)  # [B, H, W]
        
        def class_one_hot_from_logits(logits: torch.Tensor):
            # logits: [B, C, H, W] -> one-hot: [B, C, H, W]
            class_idx = torch.argmax(logits, dim=1)               # [B, H, W]
            one_hot = F.one_hot(class_idx, self.num_classes)           # [B, H, W, C]
            one_hot = one_hot.permute(0, 3, 1, 2).contiguous()     # [B, C, H, W]
            return one_hot
        
        def class_idx_from_logits(logits: torch.Tensor):
            # logits: [B, C, H, W] -> class indices: [B, H, W]
            return torch.argmax(logits, dim=1)

        output_dict = {
            'bin': bin_sigmoid_threshold(preds['bin']),             # shape: [B, 1, H, W] -> squeeze(1) => [B, H, W]
            'boundary': bin_sigmoid_threshold(preds['boundary']),   # shape: [B, 1, H, W] -> squeeze(1) => [B, H, W]
            'hv': torch.sigmoid(preds['hv']).float(),      # [B, 2, H, W], HV 是 regression map，保持不变
            'tp': class_idx_from_logits(preds['tp']),        # [B, C, H, W] -> argmax(dim=1) => [B, H, W]
        }

        if self.output_aux_tokens:
            output_dict.update({
                'bin_aux_0': bin_sigmoid_threshold(preds['bin_aux_outs'][0]),         # [B, 1, H//4, W//4] -> squeeze(1) => [B, H//4, W//4]
                'bin_aux_1': bin_sigmoid_threshold(preds['bin_aux_outs'][1]),         # [B, 1, H//2, W//2] -> squeeze(1) => [B, H//2, W//2]
                'type_aux_0': class_one_hot_from_logits(preds['type_aux_outs'][0]),   # [B, C+1, H//4, W//4] -> argmax(dim=1) => [B, H//4, W//4]
                'type_aux_1': class_one_hot_from_logits(preds['type_aux_outs'][1]),   # [B, C+1, H//2, W//2] -> argmax(dim=1) => [B, H//2, W//2]
                'type_aux_2': class_one_hot_from_logits(preds['type_aux_outs'][2]),   # [B, C+1, H, W] -> argmax(dim=1) => [B, H, W]
                'hv_aux_0': torch.sigmoid(preds['hv_aux_outs'][0]).float(),     # [B, 2, H//4, W//4]，HV 是 regression，保持不变
                'hv_aux_1': torch.sigmoid(preds['hv_aux_outs'][1]).float(),     # [B, 2, H//2, W//2]，HV 是 regression，保持不变
            })

        return output_dict
    
    def gts_mask_process(self, input_dict):
        output_dict = {
            'bin': input_dict['bin'].squeeze(1),    # [B, 1, H, W] -> squeeze(1) => [B, H, W]
            'boundary': input_dict['boundary'].squeeze(1),   # [B, 1, H, W] -> squeeze(1) => [B, H, W]
            'hv': input_dict['hv'].float(),      # [B, 2, H, W], HV 是 regression map，保持不变
            'tp': input_dict["tp"].long(),    # [B, H, W] -> 保持类别索引格式
        }

        if self.output_aux_tokens:
            output_dict.update({
                'bin_aux_0': input_dict['bin_aux_outs'][0],   # [B, 1, H//4, W//4] -> squeeze(1) => [B, H//4, W//4]
                'bin_aux_1': input_dict['bin_aux_outs'][1],   # [B, 1, H//2, W//2] -> squeeze(1) => [B, H//2, W//2]
                'type_aux_0': F.one_hot(input_dict["type_aux_outs"][0], self.num_classes).permute(0,3,1,2).contiguous(),   # [B, H//4, W//4] -> one_hot -> [B, C+1, H//4, W//4]
                'type_aux_1': F.one_hot(input_dict["type_aux_outs"][1], self.num_classes).permute(0,3,1,2).contiguous(),   # [B, H//2, W//2] -> one_hot -> [B, C+1, H//2, W//2]
                'type_aux_2': F.one_hot(input_dict["tp"], self.num_classes).permute(0,3,1,2).contiguous(),    # [B, H, W] -> one_hot -> [B, C, H, W]
                'hv_aux_0': input_dict['hv_aux_outs'][0].float(),     # [B, 2, H//4, W//4]，HV 是 regression，保持不变
                'hv_aux_1': input_dict['hv_aux_outs'][1].float(),     # [B, 2, H//2, W//2]，HV 是 regression，保持不变
            })
        return output_dict
    
    def update_metrics(self, metric_dict, preds_onehots, gts_onehots):
        """
        metric_dict: nn.ModuleDict, 每个分支一个 MetricCollection
        preds_onehots: dict, 包含模型预测的 one-hot 标签
        gts_onehots: dict, 包含 GT 标签的 one-hot 标签
        """
        for key in metric_dict.keys():
            if key not in preds_onehots or key not in gts_onehots:
                continue
            pred = preds_onehots[key]
            gt = gts_onehots[key]

            # 处理辅助输出（注意 shape 格式）
            if isinstance(pred, list) and isinstance(gt, list):
                for i, (pi, gi) in enumerate(zip(pred, gt)):
                    sub_key = f"{key}_{i}"
                    if sub_key in metric_dict:
                        metric_dict[sub_key].update(pi, gi)
            else:
                metric_dict[key].update(pred, gt)


    def print_dict(self, data_dict):
        print("\n -------- data_dict keys: ", data_dict.keys())
        for k, v in data_dict.items():
            if isinstance(v, (list, tuple)):
                print(f" ---- {k}: len={len(v)}")
                for i, vi in enumerate(v):
                    print(f"  {k}[{i}]: shape={vi.shape}, dtype={vi.dtype}, values range: [{vi.min().item()}, {vi.max().item()}]")
            else:
                print(f" ---- {k}: shape={v.shape}, dtype={v.dtype}, values range: [{v.min().item()}, {v.max().item()}]")
        print("\n")
    
    def log_losses(self, loss_dict, prefix="train", on_step=False, on_epoch=True):
        """
        Recursively log all scalar values in a nested dictionary.
        """
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                self.log_losses(value, prefix=f"{prefix}/losses/{key}", on_step=on_step, on_epoch=on_epoch)
            elif isinstance(value, (float, int, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                self.log(f"{prefix}/{key}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
    
    def log_to_tensorboard(self, losses, metrics=None, prefix="train", step=None):
        """
        将losses和metrics记录到TensorBoard
        """
        if step is None:
            step = self.global_step
        
        # 记录losses
        self._log_losses_to_tb(losses, prefix, step)
        
        # 记录metrics
        if metrics is not None:
            self._log_metrics_to_tb(metrics, prefix, step)
    
    def _log_losses_to_tb(self, losses, prefix, step):
        """
        递归记录losses到TensorBoard
        """
        for key, value in losses.items():
            if isinstance(value, dict):
                self._log_losses_to_tb(value, f"{prefix}/{key}", step)
            elif isinstance(value, (float, int, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                self.writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def _log_metrics_to_tb(self, metrics, prefix, step):
        """
        记录metrics到TensorBoard
        """
        for head_key, collection in metrics.items():
            if not hasattr(collection, "compute"):
                continue
            
            computed_metrics = collection.compute()
            for metric_name, metric_val in computed_metrics.items():
                if isinstance(metric_val, (list, tuple)):
                    values = metric_val
                elif isinstance(metric_val, torch.Tensor) and metric_val.ndim == 1:
                    values = metric_val.tolist()
                else:
                    values = [metric_val.item()]
                
                if metric_name in ["iou_classwise", "dice_classwise"]:
                    for cls_idx, val in enumerate(values):
                        if cls_idx == 0:  # skip background
                            continue
                        key = f"{metric_name.split('_')[0]}_class_{cls_idx}"
                        self.writer.add_scalar(f"{prefix}/metrics/{head_key}/{key}", val, step)
                else:
                    val = values[0]
                    self.writer.add_scalar(f"{prefix}/metrics/{head_key}/{metric_name}", val, step)
    
    def log_images_to_tensorboard(self, images, preds, gts, prefix="train", step=None):
        """
        将图像记录到TensorBoard，按照指定的2行6列布局
        第一行：输入图，GT inst_map，GT np_map，GT type_map，GT h map，GT v map
        第二行：overlay图，pred inst_map，pred np_map，pred type_map，pred h map，pred v map
        """
        if step is None:
            step = self.current_epoch
        
        # 只记录第一个batch的图像
        batch_size = min(1, images.shape[0])  # 只记录1张图像
        
        for i in range(batch_size):
            # 原始图像
            img = images[i].detach().cpu().numpy()  # [H, W, 3]
            img = np.transpose(img, (2, 0, 1))  # [3, H, W]
            
            # 归一化到[0,1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # 获取预测和真实标签
            pred_bin = preds['bin'][i].detach().cpu().numpy()  # [H, W]
            pred_tp = preds['tp'][i].detach().cpu().numpy()  # [H, W] - 已经是类别索引格式
            pred_hv = preds['hv'][i].detach().cpu().numpy()  # [2, H, W]
            
            gt_bin = gts['bin'][i].detach().cpu().numpy()  # [H, W]
            gt_tp = gts['tp'][i].detach().cpu().numpy()  # [H, W] - 已经是类别索引格式
            gt_hv = gts['hv'][i].detach().cpu().numpy()  # [2, H, W]
            
            # 生成预测的inst_map（使用后处理）
            pred_inst_map = self._generate_pred_inst_map(pred_bin, pred_hv)
            
            # 生成overlay图像
            overlay_img = self._generate_overlay_image(img, pred_inst_map, pred_tp)
            
            # 创建颜色映射的type_map
            gt_type_colored = self._create_colored_type_map(gt_tp)
            pred_type_colored = self._create_colored_type_map(pred_tp)
            
            # 创建2行6列的图像网格
            # 第一行：输入图，GT inst_map，GT np_map，GT type_map，GT h map，GT v map
            row1_images = [
                img,  # 输入图 [3, H, W]
                self._create_inst_rgb_mask(gt_tp),  # GT inst_map (使用type_map作为proxy)
                gt_bin,  # GT np_map [H, W] -> [1, H, W]
                gt_type_colored,  # GT type_map [H, W, 3] -> [3, H, W]
                gt_hv[0:1],  # GT h map [1, H, W]
                gt_hv[1:2],  # GT v map [1, H, W]
            ]
            
            # 第二行：overlay图，pred inst_map，pred np_map，pred type_map，pred h map，pred v map
            row2_images = [
                overlay_img,  # overlay图 [3, H, W]
                self._create_inst_rgb_mask(pred_inst_map),  # pred inst_map [3, H, W]
                pred_bin,  # pred np_map [H, W] -> [1, H, W]
                pred_type_colored,  # pred type_map [H, W, 3] -> [3, H, W]
                pred_hv[0:1],  # pred h map [1, H, W]
                pred_hv[1:2],  # pred v map [1, H, W]
            ]
            
            # 确保所有图像都是3通道格式
            processed_images = []
            for img_list in [row1_images, row2_images]:
                for img_data in img_list:
                    if img_data.ndim == 2:  # [H, W] -> [1, H, W] -> [3, H, W]
                        img_data = np.stack([img_data, img_data, img_data], axis=0)
                    elif img_data.ndim == 3 and img_data.shape[0] == 1:  # [1, H, W] -> [3, H, W]
                        img_data = np.repeat(img_data, 3, axis=0)
                    elif img_data.ndim == 3 and img_data.shape[2] == 3:  # [H, W, 3] -> [3, H, W]
                        img_data = np.transpose(img_data, (2, 0, 1))
                    processed_images.append(img_data)
            
            # 创建图像网格 (2行6列)
            image_grid = torch.from_numpy(np.stack(processed_images, axis=0))  # [12, 3, H, W]
            
            # 记录到TensorBoard
            self.writer.add_images(f"{prefix}/visualization", image_grid, step)
            
            # 同时保存到文件夹
            self._save_visualization_images(img, processed_images, prefix, step)
    
    def _generate_pred_inst_map(self, pred_bin, pred_hv):
        """
        使用后处理生成预测的inst_map
        """
        try:
            # 导入后处理模块
            import sys
            sys.path.append('/root/SAM2PATH-main')
            from postprocess.post_process_v2 import __proc_np_hv
            
            # 转换数据格式
            np_map = pred_bin.astype(np.float32)
            hv_map = pred_hv.transpose(1, 2, 0).astype(np.float32)  # [H, W, 2]
            
            # 调用后处理函数
            pred_inst_map = __proc_np_hv(np_map, hv_map, threshold=0.5)
            
            return pred_inst_map
            
        except Exception as e:
            if DEBUG:
                print(f"Error in post-processing: {e}")
            # 如果后处理失败，返回简单的二值图
            return (pred_bin > 0.5).astype(np.int32)
    
    def _generate_overlay_image(self, img, inst_map, type_map):
        """
        生成overlay图像，将每个核实例的边界线绘制在输入图上
        """
        try:
            # 转换图像格式 [3, H, W] -> [H, W, 3]
            overlay_img = img.transpose(1, 2, 0).copy()
            
            # 获取所有唯一的instance ID
            unique_instances = np.unique(inst_map)
            unique_instances = unique_instances[unique_instances > 0]  # 排除背景
            
            # 为每个instance绘制overlay
            for inst_id in unique_instances:
                # 获取当前instance的mask
                inst_mask = (inst_map == inst_id)
                
                # 获取该instance的类型
                inst_pixels = inst_mask & (type_map > 0)
                if not np.any(inst_pixels):
                    continue
                
                # 获取该instance的主要类型
                inst_type_values = type_map[inst_pixels]
                inst_type_id = np.bincount(inst_type_values).argmax()
                
                # 获取该类型的颜色
                type_id_str = str(inst_type_id)
                if type_id_str in self.cfg.dataset.color_dict:
                    type_name, type_color = self.cfg.dataset.color_dict[type_id_str]
                    color_normalized = np.array(type_color) / 255.0
                else:
                    color_normalized = np.array([1.0, 1.0, 1.0])  # 默认白色
                
                # 找到该instance的轮廓
                from skimage import measure
                contours = measure.find_contours(inst_mask.astype(float), 0.5)
                
                # 在overlay图像上绘制轮廓
                for contour in contours:
                    contour_coords = np.round(contour).astype(int)
                    valid_coords = (contour_coords[:, 0] >= 0) & (contour_coords[:, 0] < inst_mask.shape[0]) & \
                                  (contour_coords[:, 1] >= 0) & (contour_coords[:, 1] < inst_mask.shape[1])
                    contour_coords = contour_coords[valid_coords]
                    
                    if len(contour_coords) > 0:
                        # 绘制轮廓线
                        for y, x in contour_coords:
                            if 0 <= y < overlay_img.shape[0] and 0 <= x < overlay_img.shape[1]:
                                overlay_img[y, x] = color_normalized
            
            # 转换回 [3, H, W] 格式
            return overlay_img.transpose(2, 0, 1)
            
        except Exception as e:
            if DEBUG:
                print(f"Error generating overlay: {e}")
            return img
    
    def _create_colored_type_map(self, type_map):
        """
        根据JSON文件创建颜色映射的type_map
        """
        try:
            colored_map = np.zeros((*type_map.shape, 3), dtype=np.float32)
            
            for type_id_str, (type_name, type_color) in self.cfg.dataset.color_dict.items():
                type_id = int(type_id_str)
                mask = (type_map == type_id)
                color_normalized = np.array(type_color) / 255.0
                colored_map[mask] = color_normalized
            
            return colored_map
            
        except Exception as e:
            if DEBUG:
                print(f"Error creating colored type map: {e}")
            # 返回灰度图
            return np.stack([type_map / self.num_classes] * 3, axis=2)
    
    def _create_inst_rgb_mask(self, inst_map):
        """
        创建基于inst_map的RGB mask，每个instance随机分配一个颜色
        """
        try:
            # 获取所有唯一的instance ID
            unique_instances = np.unique(inst_map)
            unique_instances = unique_instances[unique_instances > 0]  # 排除背景
            
            # 创建RGB mask
            rgb_mask = np.zeros((*inst_map.shape, 3), dtype=np.float32)
            
            # 为每个instance分配随机颜色
            np.random.seed(42)  # 设置随机种子以确保结果可重现
            for inst_id in unique_instances:
                # 生成随机RGB颜色
                color = np.random.rand(3)
                
                # 应用颜色到该instance的所有像素
                inst_mask = (inst_map == inst_id)
                rgb_mask[inst_mask] = color
            
            return rgb_mask.transpose(2, 0, 1)  # [3, H, W]
            
        except Exception as e:
            if DEBUG:
                print(f"Error creating inst RGB mask: {e}")
            # 返回灰度图
            return np.stack([inst_map / (inst_map.max() + 1e-8)] * 3, axis=0)
    
    def _save_visualization_images(self, img, processed_images, prefix, step):
        """
        保存可视化图像到文件夹
        """
        try:
            # 创建保存目录
            visuals_dir = os.path.join(self.cfg.log_dir, f"{prefix}_visuals")
            os.makedirs(visuals_dir, exist_ok=True)
            
            # 保存图像
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 6, figsize=(24, 8))
            fig.suptitle(f'{prefix.capitalize()} Visualization - Epoch {step}', fontsize=16)
            
            # 图像标题
            titles = [
                'Input Image', 'GT Inst Map', 'GT NP Map', 'GT Type Map', 'GT H Map', 'GT V Map',
                'Overlay', 'Pred Inst Map', 'Pred NP Map', 'Pred Type Map', 'Pred H Map', 'Pred V Map'
            ]
            
            for i, (img_data, title) in enumerate(zip(processed_images, titles)):
                row = i // 6
                col = i % 6
                
                if img_data.ndim == 3 and img_data.shape[0] == 3:  # [3, H, W]
                    img_display = img_data.transpose(1, 2, 0)
                else:
                    img_display = img_data[0] if img_data.ndim == 3 else img_data
                
                axes[row, col].imshow(img_display)
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(visuals_dir, f'epoch_{step:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if DEBUG:
                print(f"Visualization saved to: {save_path}")
                
        except Exception as e:
            if DEBUG:
                print(f"Error saving visualization: {e}")
        

    def safe_log_losses(self, losses: dict, prefix: str, on_step=False, on_epoch=True):
        text = "_epoch" if on_epoch else ""

        # === Losses ===
        # print("\n🔻 --------- Losses:")
        # for head, items in losses.items():
        #     if isinstance(items, dict):
        #         print(f"  ▶ {head}:")
        #         for sub_key, vals in items.items():
        #             if isinstance(vals, dict):  # e.g., dice/bce/focal breakdown
        #                 line = f"    {sub_key}: "
        #                 for metric_name, val_list in vals.items():
        #                     preview = val_list[-max_entries:] if isinstance(val_list, list) else val_list
        #                     line += f"{metric_name}={preview}  "
        #                 print(line)
        #             else:
        #                 preview = vals[-max_entries:] if isinstance(vals, list) else vals
        #                 print(f"    {sub_key}: {preview}")
        #     else:
        #         preview = items[-max_entries:] if isinstance(items, list) else items
        #         print(f"  ▶ {head}: {preview}")
        # print("\n")

        for key, value in losses.items():
            if isinstance(value, dict):
                # e.g. losses["tp"] = {"loss": ..., "ce": ..., ...}
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, dict):
                        # Deep nested dict, e.g. losses["tp"]["tp"]["ce"]
                        for subsub_key, subsub_val in sub_val.items():
                            val = subsub_val.item() if torch.is_tensor(subsub_val) else subsub_val
                            try:
                                self.log_data[prefix]["losses"][key][sub_key][subsub_key].append(val)
                            except KeyError:
                                if key == "boundary":
                                    self.log_data[prefix]["losses"]["bin"][key][subsub_key].append(val)
                                else:
                                    raise KeyError(f"Missing log entry: [{prefix}]['losses'][{key}][{sub_key}][{subsub_key}]")
                    else:
                        # One level dict, e.g. losses["tp"]["ce"]
                        val = sub_val.item() if torch.is_tensor(sub_val) else sub_val
                        try:
                            self.log_data[prefix]["losses"][key][key][sub_key].append(val)
                        except KeyError:
                            if key == "boundary":
                                self.log_data[prefix]["losses"]["bin"][key][sub_key].append(val)
                            else:
                                raise KeyError(f"Missing log entry: [{prefix}]['losses'][{key}][{key}][{sub_key}]")
            else:
                # Summary losses: e.g. bin_loss, hv_loss, tp_loss, total_loss
                val = value.item() if torch.is_tensor(value) else value
                if key in ["bin_loss", "hv_loss", "tp_loss"]:
                    for branch in ["bin", "hv", "tp"]:
                        if key in self.log_data[prefix]["losses"].get(branch, {}):
                            self.log_data[prefix]["losses"][branch][key].append(val)
                            break
                elif key == "total_loss":
                    self.log_data[prefix]["losses"]["total_loss"].append(val)
                else:
                    raise KeyError(f"Unexpected loss key: {key}")
    

    def safe_log_metrics(self, metric_collections, prefix: str, on_step=False, on_epoch=True):
        text = "_epoch" if on_epoch else ""

        # # === Print Metrics Results ===
        # print(f"\n -------- 🔷 {prefix}{text} metric_collections, keys: ", metric_collections.keys())
        # for head, items in metric_collections.items():
        #     if isinstance(items, dict):
        #         print(f"  ▶ {head} is a dict, keys: {items.keys()}")
        #         for sub_key, val_dict in items.items():
        #             line = f"    {sub_key}: "
        #             for metric_name, val_list in val_dict.items():
        #                 max_entries = 5 if len(val_list) > 5 else len(val_list)  # 防止 print 时出错
        #                 preview = val_list[-max_entries:] if isinstance(val_list, list) else val_list
        #                 line += f"{metric_name}={preview}  "
        #             print(line)
        #     else:
        #         preview = items[-max_entries:] if isinstance(items, list) else items
        #         print(f"  ▶ {head}: {preview}, len={len(items)}")
        # print("\n")

        # # === Print selg.log_data Metrics ===
        # # print(f"\n --------- self.log_data[{prefix}{text}] keys: ", self.log_data.keys())
        # print(f"\n --------- self.log_data[{prefix}{text}][metrics] keys: ", self.log_data[prefix+text]['metrics'].keys())
        # for key, val in self.log_data[prefix+text]['metrics'].items():
        #     if isinstance(val, dict):
        #         print(f" ---- {key}: len={len(val)}, keys: {val.keys()}")
        #         for sub_key, sub_val in val.items():
        #             if isinstance(sub_val, dict):
        #                 print(f"  {sub_key}: len={len(sub_val)}, keys: {sub_val.keys()}")
        #                 for sub_sub_key, sub_sub_val in sub_val.items():
        #                     print(f"   {sub_sub_key}: len={len(sub_sub_val)}, values: {sub_sub_val}")
        #             if isinstance(sub_val, list):
        #                 print(f"  {sub_key}: len={len(sub_val)}")
        #                 for i, sub_sub_val in enumerate(sub_val):
        #                     print(f"   {i}: {sub_sub_val}")
        #             else:
        #                 print(f"  {sub_key}: values: {sub_val}")
        #     else:
        #         print(f" ---- {key}: values: {val}")
        # print("\n")

        # === Log Metrics ===
        for head_key, collection in metric_collections.items():
            if not hasattr(collection, "compute"):
                continue  # skip non-MetricCollection objects

            computed_metrics = collection.compute()  # dict: {metric_name: value}

            for metric_name, metric_val in computed_metrics.items():
                # class-wise: list/tuple/tensor with shape [num_classes]
                # print(f"\n ------- 🔷 {prefix}{text} metric_collections[{head_key}][{metric_name}]: {metric_val}")

                if isinstance(metric_val, (list, tuple)):
                    values = metric_val
                elif isinstance(metric_val, torch.Tensor) and metric_val.ndim == 1:
                    values = metric_val.tolist()
                else:
                    values = [metric_val.item()]  # scalar

                if metric_name in ["iou_classwise", "dice_classwise"]:
                    for cls_idx, val in enumerate(values):
                        # skip background (class 0)
                        if cls_idx == 0:
                            continue
                        key = f"{metric_name.split('_')[0]}_class_{cls_idx}"
                        try:
                            if head_key in self.log_data[f"{prefix}{text}"]["metrics"]:
                                if head_key not in self.log_data[f"{prefix}{text}"]["metrics"]:
                                    self.log_data[f"{prefix}{text}"]["metrics"][head_key] = {head_key: {}}
                                if head_key not in self.log_data[f"{prefix}{text}"]["metrics"][head_key]:
                                    self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key] = {}
                                if key not in self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key]:
                                    self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][key] = []

                                self.log(f"{prefix}{text}/metrics/{head_key}/{head_key}/{key}", val,
                                        on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                                self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][key].append(val)
                        except KeyError as e:
                            raise KeyError(f"[safe_log_metrics] Missing entry for {head_key}/{key}: {e}")

                        # try:
                        #     if head_key in ["bin", "tp", "hv"]:
                        #         self.log(f"{prefix}{text}/metrics/{head_key}/{head_key}/{key}", val,
                        #                 on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                        #         self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][key].append(val)
                        #     elif head_key == "boundary":
                        #         self.log(f"{prefix}{text}/metrics/boundary/{key}", val,
                        #                 on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                        #         self.log_data[f"{prefix}{text}"]["metrics"]["boundary"][key].append(val)
                        # except KeyError:
                        #     raise KeyError(f"Missing entry for [{prefix}][metrics][{head_key}][...][{key}]")
                else:
                    # normal metric
                    val = values[0]
                    try:
                        if head_key in self.log_data[f"{prefix}{text}"]["metrics"]:
                            if head_key == "boundary":
                                if metric_name not in self.log_data[f"{prefix}{text}"]["metrics"][head_key]:
                                    self.log_data[f"{prefix}{text}"]["metrics"][head_key][metric_name] = []
                                self.log(f"{prefix}{text}/metrics/{head_key}/{metric_name}", val,
                                        on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                                self.log_data[f"{prefix}{text}"]["metrics"][head_key][metric_name].append(val)
                            else:
                                if metric_name not in self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key]:
                                    self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][metric_name] = []
                                self.log(f"{prefix}{text}/metrics/{head_key}/{head_key}/{metric_name}", val,
                                        on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                                self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][metric_name].append(val)
                    except KeyError as e:
                        raise KeyError(f"[safe_log_metrics] Missing entry for {head_key}/{metric_name}: {e}")
                
                    # try:
                    #     if head_key in ["bin", "tp", "hv"]:
                    #         self.log(f"{prefix}{text}/metrics/{head_key}/{head_key}/{metric_name}", val,
                    #                 on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                    #         self.log_data[f"{prefix}{text}"]["metrics"][head_key][head_key][metric_name].append(val)
                    #     elif head_key == "boundary":
                    #         self.log(f"{prefix}{text}/metrics/boundary/{metric_name}", val,
                    #                 on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
                    #         self.log_data[f"{prefix}{text}"]["metrics"]["boundary"][metric_name].append(val)
                    # except KeyError:
                    #     raise KeyError(f"Missing entry for [{prefix}][metrics][{head_key}][...][{metric_name}]")
            
    def print_log_data(log_data, mode="train", epoch=True, max_entries=3):
        """
        Pretty-print key parts of log_data for quick inspection.

        Args:
            log_data (dict): The full log_data dictionary.
            mode (str): One of 'train', 'valid', 'test' to select mode.
            epoch (bool): Whether to print the '_epoch' version (recommended).
            max_entries (int): Number of recent entries to display per field.
        """
        key = f"{mode}_epoch" if epoch else mode
        data = log_data.get(key, {})

        print(f"\n📌 === Log Summary: {key} ===")

        # === Losses ===
        print("\n🔻 Losses:")
        losses = data.get("losses", {})
        for head, items in losses.items():
            if isinstance(items, dict):
                print(f"  ▶ {head}:")
                for sub_key, vals in items.items():
                    if isinstance(vals, dict):  # e.g., dice/bce/focal breakdown
                        line = f"    {sub_key}: "
                        for metric_name, val_list in vals.items():
                            preview = val_list[-max_entries:] if isinstance(val_list, list) else val_list
                            line += f"{metric_name}={preview}  "
                        print(line)
                    else:
                        preview = vals[-max_entries:] if isinstance(vals, list) else vals
                        print(f"    {sub_key}: {preview}")
            else:
                preview = items[-max_entries:] if isinstance(items, list) else items
                print(f"  ▶ {head}: {preview}")

        # === Metrics ===
        print("\n🔷 Metrics:")
        metrics = data.get("metrics", {})
        for head, items in metrics.items():
            if isinstance(items, dict):
                print(f"  ▶ {head}:")
                for sub_key, val_dict in items.items():
                    line = f"    {sub_key}: "
                    for metric_name, val_list in val_dict.items():
                        preview = val_list[-max_entries:] if isinstance(val_list, list) else val_list
                        line += f"{metric_name}={preview}  "
                    print(line)
            else:
                preview = items[-max_entries:] if isinstance(items, list) else items
                print(f"  ▶ {head}: {preview}")

        print("✅ Done.\n")
    
    def sanitize_output(self, output_dict):
        def clean_tensor(t):
            if torch.is_tensor(t):
                t = t.clone()
                t[torch.isnan(t)] = 0.0
                t[torch.isinf(t)] = 0.0
            return t

        for key, val in output_dict.items():
            if isinstance(val, torch.Tensor):
                output_dict[key] = clean_tensor(val)
            elif isinstance(val, list):
                output_dict[key] = [clean_tensor(v) for v in val]
            elif isinstance(val, dict):
                output_dict[key] = self.sanitize_output(val)
        return output_dict

    def training_step(self, batch, batch_idx):
        input_dict, patch_id = batch
        # input_dict = {
        #     'image': image,
        #     'bin': bin_map,
        #     'boundary': boundary_map,
        #     'inst': inst_map,
        #     'hv': hv_map,
        #     'tp': type_map,
        #     'bin_aux_outs': bin_aux_outs,
        #     'hv_aux_outs': hv_aux_outs,
        #     'type_aux_outs': type_aux_outs,
        # }

        if DEBUG:
            print(f"\n ====== training_step()")
            print(f" ====== Training step {batch_idx}, Patch ID: {patch_id}")
            self.print_dict(input_dict)
        
        # =========== Forward pass
        output_dict = self(input_dict["image"])
        if DEBUG:
            print("\n ====== Forward pass complete, output_dict: ")
            self.print_dict(output_dict)

        output_dict = self.sanitize_output(output_dict)
        if DEBUG:
            print("\n ====== sanitize_output dict: ")
            self.print_dict(output_dict)
        
        # compute loss
        with torch.autograd.set_detect_anomaly(True):
            losses = self.loss(output_dict, input_dict, ignored_masks=None)
        
        # Log all individual loss components
        self.log_losses(losses, prefix="train")
        # for key, value in loss_dict.items():
        #     if isinstance(value, dict):
        #         self.log_losses(value, prefix=f"{prefix}/{key}", on_step=on_step, on_epoch=on_epoch)
        #     elif isinstance(value, (float, int, torch.Tensor)):
        #         if isinstance(value, torch.Tensor):
        #             value = value.detach()
        #         self.log(f"{prefix}/{key}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
        

        # log losses
        self.log(f"train_loss", losses["total_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        #  每 N 步执行一次 step-level compute + log
        if (batch_idx + 1) % self.log_frequency == 0:
            # log losses
            self.safe_log_losses(losses, "train", on_step=True, on_epoch=False)
            # 记录到TensorBoard
            self.log_to_tensorboard(losses, prefix="train", step=self.global_step)
        
        # # 计算 metrics
        # output_onehots = self.predict_mask_process(output_dict)
        # if DEBUG:
        #     print("\n ====== output_onehots dict: ")
        #     self.print_dict(output_dict)
        
        # gts_onehots = self.gts_mask_process(input_dict)
        # if DEBUG:
        #     print("\n ====== gts_onehots dict: ")
        #     self.print_dict(gts_onehots)
        # self.update_metrics(self.train_metrics, output_onehots, gts_onehots)
        # self.safe_log_metrics(self.train_metrics, "train", on_step=True, on_epoch=False)

        #  Visualization the input and output of the model every 200 steps
        output_masks = None
        gts_masks = None
        if (batch_idx + 1) % self.image_log_frequency == 0:
            try:
                # 获取预测结果
                output_masks = self.predict_mask_process(output_dict)
                gts_masks = self.gts_mask_process(input_dict)
                
                # 使用新的可视化方法
                images = input_dict["image"]
                
                # 保存图像到TensorBoard和文件夹
                self.log_images_to_tensorboard(images, output_masks, gts_masks, 
                                             prefix="train", step=self.global_step)
                
                print(f"✅ Saved training visualization at step {batch_idx}, epoch {self.current_epoch}")
                
                del output_masks, gts_masks
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error saving training visualization: {e}")
                import traceback
                traceback.print_exc()

        # 保存最后一个batch的数据用于epoch结束时的图像记录
        self.last_train_batch = (input_dict, patch_id)
        
        # 优化日志保存频率：只在特定间隔或epoch结束时保存
        self.log_buffer_size += 1
        if (self.log_buffer_size % self.save_log_frequency == 0) or (batch_idx == 0):
            self._save_log()        # 持久化保存 log_data
        
        # 内存管理 - 清理所有变量
        del input_dict, output_dict
        torch.cuda.empty_cache()

        return losses['total_loss']       


    def on_train_start(self):
        print("\n Available metrics:")
        print(" Training metrics:", self.trainer.callback_metrics.keys())
    
    def on_train_epoch_end(self):
        # 分阶段训练：在指定epoch释放所有参数
        if (self.unfreeze_epoch is not None and 
            self.current_epoch >= self.unfreeze_epoch and 
            not self.parameters_unfrozen):
            
            print(f"\n🔄 Epoch {self.current_epoch}: Unfreezing all parameters for full parameter training...")
            if hasattr(self.model, 'unfreeze_all_parameters'):
                self.model.unfreeze_all_parameters()
                self.parameters_unfrozen = True
                print(f"✅ All parameters unfrozen at epoch {self.current_epoch}")
            else:
                print(f"⚠️ Model does not support unfreeze_all_parameters method")
        
        prefix = "train"
        # 更新 epoch 序号
        if self.current_epoch not in self.log_data[prefix]["epochs"]:
            self.log_data[prefix].setdefault("epochs", []).append(self.current_epoch)

        # 图像记录现在在training_step中每200步进行，不再在epoch结束时记录
        # 这样可以更频繁地监控训练过程，特别是在长epoch的情况下

        # 持久化保存 log_data
        self._save_log()

        if DEBUG:
            print(f"\n\n ====== {prefix}:{self.current_epoch} complete, log_data updated")
            self.print_log_data(self.log_data, mode="train", epoch=True)
            print(f"\n\n")
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = get_prefix_from_val_id(dataloader_idx)
        input_dict, patch_id = batch
        # input_dict = {
        #     'image': image,
        #     'bin': bin_map,
        #     'boundary': boundary_map,
        #     'inst': inst_map,
        #     'hv': hv_map,
        #     'tp': type_map,
        #     'bin_aux_outs': bin_aux_outs,
        #     'hv_aux_outs': hv_aux_outs,
        #     'type_aux_outs': type_aux_outs,
        # }

        if DEBUG:
            print(f"\n ====== validation_step()")
            print(f" ====== Validation step {batch_idx}, Patch ID: {patch_id}")
            self.print_dict(input_dict)

        # 2. 前向推理
        output_dict = self(input_dict["image"])

        if DEBUG:
            print("\n ====== Forward pass complete, output_dict: ")
            self.print_dict(output_dict)

        # # 3. 清除 nan 值
        # for head in gt_masks:
        #     if torch.isnan(gt_masks[head]).any():
        #         print(f"[WARN] NaN in val_gt {head}, patch_id: {patch_id}")
        #         gt_masks[head][torch.isnan(gt_masks[head])] = 0
        
        # 4. 计算损失
        losses = self.loss(output_dict, input_dict, ignored_masks=None)

        # Log all individual loss components
        self.log_losses(losses, prefix=prefix)
        # for key, value in loss_dict.items():
        #     if isinstance(value, dict):
        #         self.log_losses(value, prefix=f"{prefix}/{key}", on_step=on_step, on_epoch=on_epoch)
        #     elif isinstance(value, (float, int, torch.Tensor)):
        #         if isinstance(value, torch.Tensor):
        #             value = value.detach()
        #         self.log(f"{prefix}/{key}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=self.batch_size)
        
        # log losses
        self.log(f"val_loss", losses["total_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        # 5. 构造预测和one-hot标签
        preds_onehots = self.predict_mask_process(output_dict)
        if DEBUG:
            print("\n ====== Predictions generated, preds_onehots: ")
            self.print_dict(preds_onehots)

        # 6. 构造GT one-hot标签
        gts_onehots = self.gts_mask_process(input_dict)
        if DEBUG:
            print(f"\n ====== Validation_step(), GT datas: ")
            self.print_dict(input_dict)

        # Compute metrics
        self.update_metrics(self.valid_metrics[dataloader_idx], preds_onehots, gts_onehots)

        # 7. 记录学习率（当前 optimizer）
        self.log(f"{prefix}/lr", self.lr, on_step=True, on_epoch=False)

        # 每 N 步执行一次 step-level compute + log
        if (batch_idx + 1) % self.log_frequency == 0:
            self.safe_log_losses(losses, "valid", on_step=True, on_epoch=False)
            self.safe_log_metrics(self.valid_metrics[dataloader_idx], "valid", on_step=True, on_epoch=False)
            # 记录到TensorBoard
            self.log_to_tensorboard(losses, self.valid_metrics[dataloader_idx], prefix=prefix, step=self.global_step)

        #  Visualization every 200 steps for validation
        if (batch_idx + 1) % self.image_log_frequency == 0:
            try:
                # 使用新的可视化方法
                images = input_dict["image"]
                
                # 保存图像到TensorBoard和文件夹
                self.log_images_to_tensorboard(images, preds_onehots, gts_onehots, 
                                             prefix=prefix, step=self.global_step)
                
                print(f"✅ Saved {prefix} visualization at step {batch_idx}, epoch {self.current_epoch}")
                
            except Exception as e:
                print(f"❌ Error saving {prefix} visualization: {e}")
                import traceback
                traceback.print_exc()

        # 保存最后一个验证batch的数据用于epoch结束时的图像记录
        self.last_valid_batch = (input_dict, patch_id)
        
        # 验证步骤不频繁保存日志，只在epoch结束时保存
        # 内存管理
        del input_dict, output_dict, preds_onehots, gts_onehots
        torch.cuda.empty_cache()

        return losses["total_loss"]
    
    def on_validation_epoch_end(self, dataloader_idx=0):
        if self.trainer.sanity_checking:
            return

        if DEBUG:
            print(f"\n\n ====== on_validation_epoch_end() dataloader_idx: {dataloader_idx}")
            print(f" ======  self.valid_metrics: {self.valid_metrics}")

        for dataloader_idx, metric in enumerate(self.valid_metrics):
            prefix = get_prefix_from_val_id(dataloader_idx)
            
            # 更新 epoch 序号
            if self.current_epoch not in self.log_data[prefix]["epochs"]:
                self.log_data[prefix].setdefault("epochs", []).append(self.current_epoch)

            # 验证集图像记录现在在validation_step中每200步进行，不再在epoch结束时记录
            # 这样可以更频繁地监控验证过程

            # 验证epoch结束时保存日志
            self._save_log()

        if DEBUG:
            print(f"\n\n ====== {prefix}:{self.current_epoch} complete, log_data updated")
            self.print_log_data(self.log_data, mode="valid", epoch=True)
            print(f"\n\n")

    def configure_optimizers(self):
        # 分离 adapter 与其他模块
        adapter_params = [p for n, p in self.named_parameters() if p.requires_grad and 'prompt_generator' in n]
        other_params = [p for n, p in self.named_parameters() if p.requires_grad and 'prompt_generator' not in n]

        opt = torch.optim.AdamW([
            {'params': adapter_params, 'lr': self.lr * 2},        # 更快适配任务
            {'params': other_params, 'lr': self.lr}               # decoder 等常规模块
        ], weight_decay=self.weight_decay)

        def lr_lambda(step):
            if step < self.warmup_steps: return step / self.warmup_steps
            if step < self.lr_steps[0]: return 1.0
            if step < self.lr_steps[1]: return 0.1
            return 0.01

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, verbose=False)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'interval': 'step'
            }
        }
    
    def on_train_end(self):
        """训练结束时关闭TensorBoard writer"""
        # 显示性能统计
        self._log_performance_stats()
        
        # 强制保存最终日志
        self._force_save_log()
        
        if hasattr(self, 'writer'):
            self.writer.close()
            # print(f"TensorBoard logs saved to: {self.tensorboard_dir}")
    
    def on_validation_end(self):
        """验证结束时关闭TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            # print(f"TensorBoard logs saved to: {self.tensorboard_dir}")

