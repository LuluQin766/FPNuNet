# main_FPNuNet_v21n01.py
import sys
from matplotlib.pylab import f
sys.path.append('/root/SAM2PATH-main')

from trainer import train_model, setup_environment, print_config_summary
from config_manager import load_config, setup_config, parse_arguments


def main():
    """
    主运行函数
    """
    # 设置环境
    setup_environment()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 设置配置
    cfg = setup_config(cfg, args)
    
    # 打印配置摘要
    print_config_summary(cfg)
    
    # 开始训练
    trainer, pl_module = train_model(cfg, run_name=cfg.name)
    
    print("  ----- Training Done! ")


if __name__ == '__main__':
    main()
