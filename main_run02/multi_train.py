"""
多模型并行训练脚本
支持同时训练多个不同配置的模型
"""

import sys
sys.path.append('/root/SAM2PATH-main')

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from trainer import train_model, setup_environment, print_config_summary
from config_manager import load_config, setup_config, parse_devices


def create_training_config(config_path, project, name, devices, seed=42):
    """
    创建训练配置
    
    Args:
        config_path: 配置文件路径
        project: 项目名称
        name: 实验名称
        devices: GPU设备
        seed: 随机种子
        
    Returns:
        cfg: 配置对象
    """
    # 加载基础配置
    cfg = load_config(config_path)
    
    # 设置参数
    cfg["project"] = project
    cfg["name"] = name
    cfg["seed"] = seed
    cfg.devices = parse_devices(devices)
    cfg.batch_size = 8
    cfg.model.extra_encoder = 'uni_v1_adapter'
    
    # 设置随机种子
    from lightning_fabric import seed_everything
    seed_everything(cfg["seed"])
    
    return cfg


def train_single_model(config_info):
    """
    训练单个模型的函数
    
    Args:
        config_info: 包含配置信息的字典
    """
    try:
        print(f"\n🚀 Starting training: {config_info['name']}")
        print(f"   Config: {config_info['config_path']}")
        print(f"   Devices: {config_info['devices']}")
        print(f"   Project: {config_info['project']}")
        
        # 创建配置
        cfg = create_training_config(
            config_info['config_path'],
            config_info['project'],
            config_info['name'],
            config_info['devices'],
            config_info.get('seed', 42)
        )
        
        # 打印配置摘要
        print_config_summary(cfg)
        
        # 开始训练
        trainer, pl_module = train_model(cfg)
        
        print(f"✅ Training completed: {config_info['name']}")
        
    except Exception as e:
        print(f"❌ Training failed for {config_info['name']}: {str(e)}")
        import traceback
        traceback.print_exc()


def run_parallel_training():
    """
    运行并行训练
    """
    # 设置环境
    setup_environment()
    
    # 定义多个训练配置
    training_configs = [
        {
            'config_path': 'configs_run2.cd47nusc_HV_h5x128_debug',
            'project': 'FPNuNet-cd47nuscx128',
            'name': 'FPNuNet_cd47nuscx128_Mv231_GPU0',
            'devices': '0',
            'seed': 42
        },
        {
            'config_path': 'configs_run2.cd47nusc_HV_h5x128_debug',
            'project': 'FPNuNet-cd47nuscx128',
            'name': 'FPNuNet_cd47nuscx128_Mv231_GPU1',
            'devices': '1',
            'seed': 43
        },
        # 可以添加更多配置...
    ]
    
    print(f"🎯 Starting parallel training with {len(training_configs)} configurations")
    
    # 使用线程池进行并行训练
    with ThreadPoolExecutor(max_workers=len(training_configs)) as executor:
        # 提交所有训练任务
        futures = [executor.submit(train_single_model, config) for config in training_configs]
        
        # 等待所有任务完成
        for future in futures:
            future.result()
    
    print("🎉 All training tasks completed!")


def run_sequential_training():
    """
    运行顺序训练（一个接一个）
    """
    # 设置环境
    setup_environment()
    
    # 定义多个训练配置
    training_configs = [
        {
            'config_path': 'configs_run2.cd47nusc_HV_h5x128_debug',
            'project': 'FPNuNet-cd47nuscx128',
            'name': 'FPNuNet_cd47nuscx128_Mv231_Sequential1',
            'devices': '0',
            'seed': 42
        },
        {
            'config_path': 'configs_run2.cd47nusc_HV_h5x128_debug',
            'project': 'FPNuNet-cd47nuscx128',
            'name': 'FPNuNet_cd47nuscx128_Mv231_Sequential2',
            'devices': '0',
            'seed': 43
        },
    ]
    
    print(f"🎯 Starting sequential training with {len(training_configs)} configurations")
    
    # 顺序执行训练
    for i, config in enumerate(training_configs):
        print(f"\n📋 Training {i+1}/{len(training_configs)}: {config['name']}")
        train_single_model(config)
        print(f"✅ Completed {i+1}/{len(training_configs)}")
    
    print("🎉 All training tasks completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-model training script')
    parser.add_argument('--mode', type=str, default='sequential', 
                       choices=['sequential', 'parallel'],
                       help='Training mode: sequential or parallel')
    
    args = parser.parse_args()
    
    if args.mode == 'parallel':
        run_parallel_training()
    else:
        run_sequential_training()
