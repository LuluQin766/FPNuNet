"""
训练函数模块
包含核心的训练逻辑，可以被多个运行脚本调用
"""

import os
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from utils_run02 import get_data_module, get_pl_module
from get_model import get_model
from metrics_v21 import get_metrics


def save_model_config(cfg, run_name):
    """
    保存模型配置信息到JSON文件
    
    Args:
        cfg: 配置对象
        run_name: 运行名称
    """
    import json
    
    # 创建配置信息字典
    config_info = {
        "run_name": run_name,
        "model": {
            "extra_type": cfg.model.extra_type,
            "extra_encoder": cfg.model.extra_encoder,
            "type": cfg.model.type,
            "checkpoint": cfg.model.checkpoint,
            "prompt_dim": cfg.model.prompt_dim,
            "freeze": {
                "image_encoder": cfg.model.freeze.image_encoder,
                "prompt_encoder": cfg.model.freeze.prompt_encoder,
                "mask_decoder": cfg.model.freeze.mask_decoder,
            }
        },
        "dataset": {
            "num_classes": cfg.dataset.num_classes,
            "image_hw": cfg.dataset.image_hw,
            "ignored_classes": cfg.dataset.ignored_classes,
        },
        "training": {
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.opt.num_epochs,
            "learning_rate": cfg.opt.learning_rate,
            "weight_decay": cfg.opt.weight_decay,
            "precision": cfg.opt.precision,
            "unfreeze_epoch": cfg.opt.unfreeze_epoch if "unfreeze_epoch" in cfg.opt else None,
            "early_stopping_patience": cfg.opt.early_stopping_patience if "early_stopping_patience" in cfg.opt else None,
        },
        "devices": cfg.devices,
        "project": cfg.project,
        "seed": cfg.random_seed,
    }
    
    # 保存到JSON文件
    config_file = os.path.join(cfg.log_dir, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=4)
    
    print(f" -------- Model config saved to: {config_file}")


def train_model(cfg, run_name=None):
    """
    训练单个模型的函数
    
    Args:
        cfg: 配置对象
        run_name: 运行名称，如果为None则自动生成
        
    Returns:
        trainer: 训练器对象
        pl_module: PyTorch Lightning模块
    """
    print(f"\n\n ----------- Starting Training: {cfg.get('name', 'Unknown')} -----------")
    
    # 设置运行名称和日志目录
    if run_name is None:
        # 直接使用cfg.name，它已经在config_manager中包含了时间戳
        run_name = cfg.name
    
    log_dir = os.path.join(cfg.out_dir, run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    cfg.log_dir = log_dir
    cfg.log_train_images_path = os.path.join(log_dir, 'train_images')
    os.makedirs(cfg.log_train_images_path, exist_ok=True)
    cfg.log_val_images_path = os.path.join(log_dir, 'val_images')
    os.makedirs(cfg.log_val_images_path, exist_ok=True)
    cfg.log_test_images_path = os.path.join(log_dir, 'test_images')
    os.makedirs(cfg.log_test_images_path, exist_ok=True)
    
    print(f" -------- project_path: {cfg.out_dir}")
    print(f" -------- run_name: {run_name}")
    print(f" -------- log_dir: {log_dir}")
    print(f" -------- cfg.log_train_images_path: {cfg.log_train_images_path}")
    print(f" -------- cfg.log_val_images_path: {cfg.log_val_images_path}")
    print(f" -------- cfg.log_test_images_path: {cfg.log_test_images_path}")

    # 准备数据模块
    data_module = get_data_module(cfg)

    # 准备模型
    sam_model = get_model(cfg)

    print(f" ✅ [MEMORY] allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, \n"
          f" ✅  [MEMORY] reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # 准备指标
    metrics = get_metrics(cfg.dataset.num_classes, ignore_index=cfg.dataset.ignored_classes)

    # 设置训练和验证过程
    # 从配置中获取模块版本，默认为v21
    pl_module_version = getattr(cfg, 'pl_module_version', 'v21')
    
    print(f"\n🚀 开始初始化训练模块...")
    print(f"📋 配置信息:")
    print(f"   - 实验名称: {cfg.get('name', 'Unknown')}")
    print(f"   - 模块版本: {pl_module_version}")
    print(f"   - 批次大小: {cfg.batch_size}")
    print(f"   - 学习率: {cfg.opt.learning_rate}")
    print(f"   - 权重衰减: {cfg.opt.weight_decay}")
    
    pl_module = get_pl_module(cfg, model=sam_model, metrics=metrics, version=pl_module_version)
    
    # 注意：TensorBoard 记录已在 pl_module_multiHead_v21.py 中自定义实现
    # 这里只设置 WandB logger，避免重复记录
    
    # 设置WandB logger
    wandb_logger = WandbLogger(
        project=cfg.project, 
        name=run_name, 
        save_dir=cfg.log_dir, 
        log_model=True
    )
    wandb_logger.watch(sam_model, log='all', log_freq=10)
    print(f" -------- wandb_logger: {wandb_logger}")
    print(f" -------- wandb_logger.save_dir: {wandb_logger.save_dir}")
    print(f" -------- TensorBoard logs will be saved to: {cfg.log_dir}/tensorboard")

    # 只使用 WandB logger
    loggers = [wandb_logger]
    
    # 设置学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accumulate_grad_batches = cfg.accumulate_grad_batches if "accumulate_grad_batches" in cfg else 1

    # 添加模型检查点回调
    ckpt_path = os.path.join(cfg.log_dir, 'model_checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model_checkpoint = ModelCheckpoint(
        dirpath=ckpt_path,  # 保存模型的目录
        filename=run_name + '-{epoch}-{val_loss:.4f}',  # 文件名格式
        monitor='val_loss',  # 监控的指标 
        mode='min',  # 保存最佳模型的模式
        save_top_k=5,  # 保存最好的5个模型
        save_last=True,  # 总是保存最后一个epoch的模型
        every_n_epochs=1  # 每个epoch都保存
    )
    
    print(f" -------- model_checkpoint.dirpath: {model_checkpoint.dirpath}")
    print(f" -------- cfg.devices: {cfg.devices}")

    # 添加早停机制回调
    early_stopping_patience = cfg.opt.early_stopping_patience if "early_stopping_patience" in cfg.opt else None
    callbacks = [lr_monitor, model_checkpoint]
    
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            monitor='val_loss',           # 监控验证损失
            mode='min',                   # 损失越小越好
            patience=early_stopping_patience,  # 耐心值
            min_delta=1e-4,              # 最小改善阈值
            verbose=True,                # 打印早停信息
            strict=True,                 # 严格模式
            check_finite=True,          # 检查有限值
            stopping_threshold=None,    # 停止阈值
            divergence_threshold=None,  # 发散阈值
            check_on_train_epoch_end=False  # 不在训练epoch结束时检查
        )
        callbacks.append(early_stopping)
        print(f" -------- Early stopping enabled with patience: {early_stopping_patience}")
    else:
        print(f" -------- Early stopping disabled")

    # 保存模型配置信息
    save_model_config(cfg, run_name)

    # 创建训练器
    # 注意：PyTorch Lightning 默认会在当前目录创建 lightning_logs/ 目录
    # 但我们的模型检查点都保存在 model_checkpoints/ 目录中
    trainer = Trainer(
        default_root_dir=cfg.log_dir, 
        logger=loggers,  # 使用合并的loggers
        devices=cfg.devices,    # e.g., [2,3] or just 4
        max_epochs=cfg.opt.num_epochs,
        accelerator="gpu", 
        strategy="auto",
        log_every_n_steps=5,   # Log metrics every 5 global steps
        num_sanity_val_steps=0,
        precision=cfg.opt.precision,
        callbacks=callbacks,  # 使用包含早停机制的回调列表
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=False
    )
    
    print(f" -------- trainer.default_root_dir: {trainer.default_root_dir}")

    # 开始训练
    trainer.fit(pl_module, data_module)
    
    # 确保 TensorBoard writer 被正确关闭
    if hasattr(pl_module, 'writer') and pl_module.writer is not None:
        pl_module.writer.close()
        print(f" ✅ TensorBoard writer closed for {run_name}")
    
    print(f" ✅ Training completed for {run_name}")
    return trainer, pl_module


def setup_environment():
    """
    设置训练环境
    """
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 现在通过 --devices 参数控制
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["WANDB_MODE"]="offline"
    
    import torch
    torch.set_float32_matmul_precision('high')  # 或 'medium'
    
    print("\n ✅ Environment setup completed")


def print_config_summary(cfg):
    """
    打印配置摘要
    """
    print("\n -------- Configuration Summary:")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"  ---- {k}:")
            for kk, vv in v.items():
                print(f"   {kk}: {vv}")
        else:
            print(f"  ---- {k}: {v}")
    print("\n")
