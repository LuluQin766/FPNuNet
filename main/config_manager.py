"""
配置管理模块
用于加载和管理训练配置
"""

import sys
import torch
from argparse import ArgumentParser
from lightning_fabric import seed_everything


def parse_devices(dev_input):
    """
    解析设备输入参数，支持多种格式
    
    Args:
        dev_input: 设备输入，可以是字符串、列表或整数
                  - 字符串: "0" 或 "0,1,2"
                  - 列表: [0, 1, 2]
                  - 整数: 0
                  
    Returns:
        list: 设备ID列表
        
    Raises:
        ValueError: 当输入格式无效时
    """
    if isinstance(dev_input, str):
        if dev_input.isdigit():
            return [int(dev_input)]
        return [int(d.strip()) for d in dev_input.split(',')]
    elif isinstance(dev_input, list):
        return [int(d) for d in dev_input]
    elif isinstance(dev_input, int):
        return [dev_input]
    else:
        raise ValueError("Invalid format for --devices")



def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如 "configs.CD47_nuclei_HV_h5_128x128"
        
    Returns:
        cfg: 配置对象
    """
    # 处理配置文件路径
    if config_path.startswith('configs/'):
        # 移除路径前缀，只保留模块名
        config_path = config_path.replace('configs/', '').replace('.py', '')
    elif config_path.startswith('configs.'):
        # 如果已经是 configs. 格式，直接使用
        config_path = config_path.replace('configs.', '')
    
    # 添加configs前缀
    full_module_path = f"configs.{config_path}"
    module = __import__(full_module_path, globals(), locals(), ['cfg'])
    cfg = module.cfg
    return cfg


def setup_config(cfg, args):
    """
    设置配置参数
    
    Args:
        cfg: 配置对象
        args: 命令行参数
        
    Returns:
        cfg: 更新后的配置对象
    """
    import time
    
    # 设置基本参数
    cfg["project"] = args.project
    cfg["seed"] = args.seed

    # 设置显卡设备
    cfg.devices = parse_devices(args.devices)
    print(f"[Device Config] Using GPU devices: {cfg.devices}")

    # 设置模型参数
    cfg.batch_size = 8
    
    # 设置模型类型：如果命令行指定了extra_type，则覆盖配置文件中的设置
    if args.extra_type is not None:
        original_type = cfg.model.extra_type
        cfg.model.extra_type = args.extra_type
        print(f" -------- Model extra_type overridden: {original_type} -> {cfg.model.extra_type}")
    else:
        print(f" -------- Using config model extra_type: {cfg.model.extra_type}")
    
    # 设置PyTorch Lightning模块版本：如果命令行指定了pl_module_version，则覆盖配置文件中的设置
    if args.pl_module_version is not None:
        original_version = cfg.get('pl_module_version', 'None')
        cfg["pl_module_version"] = args.pl_module_version
        print(f" -------- PL module version overridden: {original_version} -> {cfg['pl_module_version']}")
    else:
        print(f" -------- Using config PL module version: {cfg.get('pl_module_version', 'None')}")
    
    # cfg.model.extra_encoder = 'uni_v1_adapter'
    print(f" -------- cfg.model.extra_encoder: {cfg.model.extra_encoder}")

    # 检测debug模式：如果配置文件名包含"debug"，则启用debug模式
    if 'debug' in args.config.lower():
        cfg["debug_mode"] = True
        print(f"🐛 Debug mode detected from config file: {args.config}")
    else:
        cfg["debug_mode"] = False

    # 自动生成实验名称
    if args.run_name is not None:
        # 如果提供了自定义run_name，直接使用
        cfg["name"] = args.run_name
        print(f" -------- Using custom run name: {cfg['name']}")
    else:
        # 否则自动生成实验名称
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        cfg["name"] = f"{args.name}_{cfg.model.extra_type}_run02-{current_time}"
        print(f" -------- Generated experiment name: {cfg['name']}")

    print(f"\n -------- cfg.loss.bin: {cfg.loss.bin}")
    print(f" -------- cfg.loss.tp: {cfg.loss.tp}")
    print(f" -------- cfg.loss.hv: {cfg.loss.hv}\n")

    # 设置随机种子
    seed_everything(cfg["seed"])
    
    return cfg


def create_argument_parser():
    """
    创建命令行参数解析器
    
    Returns:
        parser: ArgumentParser对象
    """
    parser = ArgumentParser()
    
    # 基本参数
    parser.add_argument("--config", default="configs.CD47_nuclei_HV_h5_128x128", type=str,
                      help="Configuration file path")
    parser.add_argument('--project', type=str, default="FPNuNet_cd47nuscx128",
                      help="Project name for logging")
    parser.add_argument('--name', type=str, default="FPNuNet_cd47nuscx128",
                      help="Base experiment name (will be extended with model type and timestamp)")
    parser.add_argument('--seed', type=int, default=42,
                      help="Random seed")
    parser.add_argument('--devices', type=str, default="0",
                      help="GPU devices to use, e.g., '0' for single GPU, '0,1' for multiple GPUs")
    parser.add_argument('--extra_type', type=str, default=None,
                      help="Model extra_type to override config (e.g., 'multihead_v231', 'multihead_v232')")
    parser.add_argument('--run_name', type=str, default=None,
                      help="Custom run name (if not provided, will auto-generate)")
    parser.add_argument('--pl_module_version', type=str, default=None,
                      help="PyTorch Lightning module version ('v21' or 'v22')")
    
    return parser


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数对象
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    return args
