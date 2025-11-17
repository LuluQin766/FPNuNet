"""
工具函数模块
统一管理数据模块和PyTorch Lightning模块的创建函数
支持多版本模块选择
"""

from h5dataloder_v21 import H5GeneralDataModule

# 动态导入 PanNuke 数据加载器（如果可用）
try:
    from h5dataloder_pannuke import PanNukeMultiFoldDataModule, get_pannuke_fold_datamodules, get_augmentation as get_pannuke_augmentation
    PANNUKE_AVAILABLE = True
except ImportError:
    PANNUKE_AVAILABLE = False
    print("⚠️  PanNuke data loader not available")

# 动态导入不同版本的模块
def _check_config_completeness(cfg, version):
    """
    检查配置文件是否包含版本所需的参数
    
    Args:
        cfg: 配置对象
        version (str): 版本号
        
    Returns:
        bool: 配置是否完整
    """
    print(f"\n🔍 检查 {version} 版本配置完整性...")
    
    missing_params = []
    warnings = []
    
    if version == "v21":
        # v21版本需要的参数
        required_params = [ 
            ("opt.learning_rate", "学习率"),
            ("opt.weight_decay", "权重衰减"),
            ("opt.steps", "学习率调度步数"),
            ("opt.warmup_steps", "预热步数"),
        ]
        
        for param_path, param_name in required_params:
            if not _has_nested_attr(cfg, param_path):
                missing_params.append(f"   ❌ {param_name} ({param_path})")
            else:
                print(f"   ✅ {param_name}: {_get_nested_attr(cfg, param_path)}")
                
    elif version == "v22":
        # v22版本需要的参数
        required_params = [
            ("opt.learning_rate", "学习率"),
            ("opt.weight_decay", "权重衰减"),
            ("opt.warmup_steps", "预热步数"),
        ]
        
        optional_params = [
            ("opt.gradient_clip_val", "梯度裁剪值"),
            ("opt.scheduler", "调度器类型"),
        ]
        
        # 检查必需参数
        for param_path, param_name in required_params:
            if not _has_nested_attr(cfg, param_path):
                missing_params.append(f"   ❌ {param_name} ({param_path})")
            else:
                print(f"   ✅ {param_name}: {_get_nested_attr(cfg, param_path)}")
        
        # 检查可选参数
        for param_path, param_name in optional_params:
            if not _has_nested_attr(cfg, param_path):
                warnings.append(f"   ⚠️ {param_name} ({param_path}) - 将使用默认值")
            else:
                print(f"   ✅ {param_name}: {_get_nested_attr(cfg, param_path)}")
        
        # 检查是否包含v21的参数（可能不需要）
        if _has_nested_attr(cfg, "opt.steps"):
            warnings.append(f"   ⚠️ opt.steps - v22版本不需要此参数，将被忽略")
    
    # 输出结果
    if missing_params:
        print(f"\n❌ 配置不完整，缺少以下必需参数:")
        for param in missing_params:
            print(param)
        print(f"\n💡 请检查配置文件是否正确设置了 {version} 版本所需的参数")
        return False
    
    if warnings:
        print(f"\n⚠️ 配置警告:")
        for warning in warnings:
            print(warning)
    
    print(f"✅ {version} 版本配置检查完成")
    return True


def _has_nested_attr(obj, attr_path):
    """检查嵌套属性是否存在"""
    try:
        attrs = attr_path.split('.')
        current = obj
        for attr in attrs:
            current = getattr(current, attr)
        return True
    except (AttributeError, KeyError):
        return False


def _get_nested_attr(obj, attr_path):
    """获取嵌套属性的值"""
    try:
        attrs = attr_path.split('.')
        current = obj
        for attr in attrs:
            current = getattr(current, attr)
        return current
    except (AttributeError, KeyError):
        return None


def _import_pl_module(version="v21"):
    """
    根据版本号动态导入对应的PyTorch Lightning模块
    
    Args:
        version (str): 版本号，支持 "v21" 或 "v22"
        
    Returns:
        SamSegMultiHead: 对应版本的模块类
    """
    if version == "v21":
        from pl_module_multiHead_v21 import SamSegMultiHeadV6 as SamSegMultiHead
        print(f"\n📦 使用 PyTorch Lightning 模块版本: {version}")
        print(f"🎯 学习率调度方法: LambdaLR (固定步数衰减)")
        print(f"   - 支持参数: steps, warmup_steps")
        print(f"   - 调度方式: warmup → 1.0 → 0.1 → 0.01\n")
        return SamSegMultiHead
    elif version == "v22":
        from pl_module_multiHead_v22 import SamSegMultiHeadV6 as SamSegMultiHead
        print(f"\n📦 使用 PyTorch Lightning 模块版本: {version}")
        print(f"🎯 学习率调度方法: CosineAnnealingLR + Warmup (余弦退火)")
        print(f"   - 支持参数: warmup_steps, gradient_clip_val, scheduler")
        print(f"   - 调度方式: warmup → 余弦退火衰减\n")
        return SamSegMultiHead
    else:
        raise ValueError(f"\n❌ 不支持的版本: {version}。支持的版本: v21, v22\n")


def get_data_module(cfg):
    """
    创建统一的数据模块
    
    Args:
        cfg: 配置对象，包含数据集路径和参数
        
    Returns:
        DataModule: 配置好的数据模块（H5GeneralDataModule 或 PanNukeDataModule）
    """
    
    # 检测是否使用 PanNuke 数据集
    use_pannuke = (hasattr(cfg.dataset, 'pannuke_dataloader') and 
                   cfg.dataset.pannuke_dataloader == "pannuke" and
                   PANNUKE_AVAILABLE)
    
    if use_pannuke:
        # 使用 PanNuke 数据加载器
        print("\n📊 Using PanNuke data loader")
        
        # 检测debug模式
        debug_mode = getattr(cfg, 'debug_mode', False)
        
        # 使用默认的 fold3
        from h5dataloder_pannuke import PanNukeDataModule, get_augmentation
        data_module = PanNukeDataModule(
            data_root=cfg.dataset.pannuke_data_root,
            split_name='fold3',
            augmentation=get_augmentation(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            dataset_mean=cfg.dataset.dataset_mean,
            dataset_std=cfg.dataset.dataset_std,
            output_aux_tokens=False,
            debug_mode=debug_mode,
        )
        return data_module
    
    # 使用传统的 H5 数据加载器
    print("\n📊 Using H5 data loader")
    data_file_dict = {
        "train": cfg.dataset.train_h5_file_path,
        "test": cfg.dataset.test_h5_file_path,
    }

    common_cfg_dict = {
        "dataset_mean": cfg.dataset.dataset_mean,
        "dataset_std": cfg.dataset.dataset_std,
        "ignored_classes": cfg.dataset.ignored_classes,  # only supports None, 0 or [0, ...]
    }

    # 检测debug模式：如果配置文件名以"debug"结尾，则启用debug模式
    debug_mode = False
    
    # 方法1: 检查cfg.debug_mode配置
    if hasattr(cfg, 'debug_mode') and cfg.debug_mode:
        debug_mode = True
        print(f"🐛 Debug mode enabled from config.debug_mode")
    
    # 方法2: 检查配置文件名（通过检查数据路径中的文件名）
    elif (hasattr(cfg.dataset, 'train_h5_file_path') and 
          cfg.dataset.train_h5_file_path is not None and 
          'debug' in cfg.dataset.train_h5_file_path.lower()):
        debug_mode = True
        print(f"🐛 Debug mode detected from dataset path containing 'debug'")
    
    # 方法3: 检查实验名称是否包含debug
    elif hasattr(cfg, 'name') and cfg.name is not None and 'debug' in cfg.name.lower():
        debug_mode = True
        print(f"🐛 Debug mode detected from experiment name: {cfg.name}")
    
    if debug_mode:
        print(" --- Debug mode: limiting samples to 50 for faster testing")
    else:
        print(" --- Using image input")
    
    # === 初始化 DataModule ===
    data_module = H5GeneralDataModule(
        data_file_dict=data_file_dict,
        common_cfg_dict=common_cfg_dict,
        dataset_classs=cfg.dataset.num_classes, 
        augs_augmentation=None,
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        output_aux_tokens=False,
        debug_mode=debug_mode,
    )
    
    return data_module


def get_pl_module(cfg, model, metrics, version="v21"):
    """
    创建统一的PyTorch Lightning模块
    
    Args:
        cfg: 配置对象
        model: SAM模型
        metrics: 指标集合
        version (str): 模块版本，支持 "v21" 或 "v22"，默认为 "v21"
        
    Returns:
        SamSegMultiHead: 配置好的PyTorch Lightning模块
    """
    # 检查配置中是否指定了版本
    config_version = getattr(cfg, 'pl_module_version', None)
    if config_version and config_version != version:
        print(f"⚠️ 配置文件中指定的版本 ({config_version}) 与参数版本 ({version}) 不一致")
        print(f"   使用参数版本: {version}")
    
    # 检查配置完整性
    if not _check_config_completeness(cfg, version):
        raise ValueError(f"配置文件不完整，无法创建 {version} 版本的模块")
    
    # 动态导入对应版本的模块
    SamSegMultiHead = _import_pl_module(version)
    
    # 创建模块实例
    pl_module = SamSegMultiHead(
        cfg=cfg,
        sam_model=model,
        metrics=metrics,
        num_classes=cfg.dataset.num_classes,
        lr=cfg.opt.learning_rate,
        weight_decay=cfg.opt.weight_decay,
        lr_steps=getattr(cfg.opt, 'steps', None),  # 兼容v22版本可能没有steps参数
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
        output_aux_tokens=False,
    )
    return pl_module

