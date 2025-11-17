#!/usr/bin/env python3
"""
CD47数据集可视化脚本
用于验证处理后的数据是否正确
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from skimage import measure, morphology

# 添加项目根目录到路径
sys.path.append('/root/aMI_projects/SRSA-Net-main')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 静默模式设置 - 设置为True时不输出保存路径信息
SILENT_MODE = True

class CD47DataVisualizer:
    """CD47数据可视化器"""
    
    def __init__(self, data_dir, type_info_path=None):
        """
        初始化可视化器
        
        Args:
            data_dir: 处理后的数据目录
            type_info_path: 类型信息JSON文件路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载类型信息
        if type_info_path is None:
            type_info_path = Path(__file__).parent / 'type_info_cd47_nuclei.json'
        self.type_info = self._load_type_info(type_info_path)
        self.num_classes = len(self.type_info)
    
    def _load_type_info(self, type_info_path):
        """
        加载类型信息
        
        Args:
            type_info_path: 类型信息JSON文件路径
            
        Returns:
            dict: 类型信息字典
        """
        try:
            with open(type_info_path, 'r') as f:
                type_info = json.load(f)
            print(f"Loaded type info with {len(type_info)} classes")
            return type_info
        except Exception as e:
            print(f"Warning: Could not load type info from {type_info_path}: {e}")
            # 返回默认的3类信息
            return {
                "0": ["background", [255, 255, 255]],
                "1": ["type1", [255, 0, 0]],
                "2": ["type2", [0, 255, 0]]
            }
    
    def load_sample_data(self, split='train', sample_idx=0):
        """
        加载样本数据
        
        Args:
            split: 'train' 或 'val'
            sample_idx: 样本索引
            
        Returns:
            tuple: (image, mask_data) 或 None
        """
        image_dir = self.data_dir / 'images' / split
        mask_dir = self.data_dir / 'masks' / split
        
        if not image_dir.exists() or not mask_dir.exists():
            print(f"错误: 目录不存在 {image_dir} 或 {mask_dir}")
            return None
        
        # 获取文件列表
        image_files = sorted(list(image_dir.glob('*.tif')))
        mask_files = sorted(list(mask_dir.glob('*.npy')))
        
        if len(image_files) == 0:
            print(f"错误: 在 {image_dir} 中没有找到图像文件")
            return None
        
        if sample_idx >= len(image_files):
            print(f"错误: 样本索引 {sample_idx} 超出范围 (0-{len(image_files)-1})")
            return None
        
        # 加载图像
        img_path = image_files[sample_idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"错误: 无法读取图像 {img_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载mask数据
        mask_path = mask_files[sample_idx]
        mask_data = np.load(mask_path, allow_pickle=True).item()
        
        return image, mask_data, img_path.stem
    
    def visualize_sample(self, image, mask_data, sample_name, split='train'):
        """
        可视化单个样本
        
        Args:
            image: 图像数据 (H, W, 3)
            mask_data: mask数据字典
            sample_name: 样本名称
            split: 数据分割
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'CD47 Data Sample Visualization - {sample_name} ({split})', fontsize=16)
        
        # First row: Original image, NP branch, HV branches
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # NP branch: Binary segmentation mask
        np_mask = mask_data['np_map']
        axes[0, 1].imshow(np_mask, cmap='gray')
        axes[0, 1].set_title('NP Branch: Binary Segmentation')
        axes[0, 1].axis('off')
        
        # HV branch: Horizontal distance
        hv_map = mask_data['hv_map']
        hv_h = hv_map[:, :, 1]  # Horizontal distance
        im1 = axes[0, 2].imshow(hv_h, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 2].set_title('HV Branch: Horizontal Distance')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # HV branch: Vertical distance
        hv_v = hv_map[:, :, 0]  # Vertical distance
        im2 = axes[0, 3].imshow(hv_v, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 3].set_title('HV Branch: Vertical Distance')
        axes[0, 3].axis('off')
        plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Second row: 改进的布局
        type_map = mask_data['type_map']
        inst_map = mask_data['inst_map']
        
        # 第二行第一张：显示create_nuclei_overlay的overlay
        overlay_image = self._create_nuclei_overlay_for_visualization(image, mask_data)
        axes[1, 0].imshow(overlay_image)
        axes[1, 0].set_title('Nuclei Overlay')
        axes[1, 0].axis('off')
        
        # 第二行第二张：NC Branch Colored Type Map
        if type_map.ndim == 3:
            # 多通道格式：显示第一个非背景通道
            actual_num_channels = type_map.shape[0]
            if actual_num_channels > 1:
                axes[1, 1].imshow(type_map[1], cmap='gray')
                axes[1, 1].set_title('NC Branch: Type Channel 1')
            else:
                axes[1, 1].imshow(type_map[0], cmap='gray')
                axes[1, 1].set_title('NC Branch: Type Channel 0')
        else:
            # 单通道格式：创建颜色映射的可视化
            type_map_colored = self._create_colored_type_map(type_map)
            axes[1, 1].imshow(type_map_colored)
            axes[1, 1].set_title('NC Branch: Colored Type Map')
        axes[1, 1].axis('off')
        
        # 第二行第三张：修正的统计信息图
        self._create_improved_statistics_plot(axes[1, 2], image, mask_data, type_map)
        
        # 第二行第四张：inst_map的RGB mask
        inst_rgb_mask = self._create_inst_rgb_mask(inst_map)
        axes[1, 3].imshow(inst_rgb_mask)
        axes[1, 3].set_title('Instance RGB Mask')
        axes[1, 3].axis('off')
        
        # Third row: 填充剩余空间
        for i in range(4):
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / split / f'{sample_name}_{split}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if not SILENT_MODE:
            print(f"可视化结果已保存到: {output_path}")
    
    def _create_colored_type_map(self, type_map):
        """
        创建颜色映射的类型图
        
        Args:
            type_map: 单通道类型图 (H, W)
            
        Returns:
            np.ndarray: 颜色映射后的图像 (H, W, 3)
        """
        colored_map = np.zeros((*type_map.shape, 3), dtype=np.float32)
        
        for type_id_str, (type_name, type_color) in self.type_info.items():
            type_id = int(type_id_str)
            mask = (type_map == type_id)
            color_normalized = np.array(type_color) / 255.0
            colored_map[mask] = color_normalized
        
        return colored_map
    
    def _create_nuclei_overlay_for_visualization(self, image, mask_data):
        """
        为可视化创建核边缘overlay图像
        
        Args:
            image: 图像数据 (H, W, 3)
            mask_data: mask数据字典
            
        Returns:
            np.ndarray: overlay图像 (H, W, 3)
        """
        inst_map = mask_data['inst_map']
        type_map = mask_data['type_map']
        
        # 创建RGB图像用于overlay
        overlay_image = image.copy().astype(np.float32) / 255.0
        
        # 获取所有唯一的instance ID
        unique_instances = np.unique(inst_map)
        unique_instances = unique_instances[unique_instances > 0]  # 排除背景
        
        # 为每个instance绘制overlay
        for inst_id in unique_instances:
            # 获取当前instance的mask
            inst_mask = (inst_map == inst_id)
            
            # 获取该instance在type_map中的类型
            if type_map.ndim == 3:
                # 多通道格式：找到该instance对应的类型
                inst_type_id = None
                for type_id_str in self.type_info.keys():
                    type_id = int(type_id_str)
                    if type_id == 0:  # 跳过背景
                        continue
                    if type_id >= type_map.shape[0]:  # 跳过超出实际通道数的类型
                        continue
                    
                    # 检查该instance是否属于这个类型
                    type_mask = type_map[type_id]
                    if np.any(inst_mask & (type_mask > 0)):
                        inst_type_id = type_id
                        break
                
                if inst_type_id is None:
                    continue
                    
                # 获取该类型的颜色
                type_id_str = str(inst_type_id)
                if type_id_str not in self.type_info:
                    continue
                type_name, type_color = self.type_info[type_id_str]
                
            else:
                # 单通道格式：直接获取该instance的类型
                inst_pixels = inst_mask & (type_map > 0)
                if not np.any(inst_pixels):
                    continue
                
                # 获取该instance的主要类型（出现最多的类型）
                inst_type_values = type_map[inst_pixels]
                inst_type_id = np.bincount(inst_type_values).argmax()
                
                # 获取该类型的颜色
                type_id_str = str(inst_type_id)
                if type_id_str not in self.type_info:
                    continue
                type_name, type_color = self.type_info[type_id_str]
            
            # 将RGB颜色归一化到[0,1]
            color_normalized = np.array(type_color) / 255.0
            
            # 找到该instance的轮廓
            contours = measure.find_contours(inst_mask.astype(float), 0.5)
            
            # 在overlay图像上绘制轮廓
            for contour in contours:
                # 创建轮廓mask
                contour_mask = np.zeros_like(inst_mask)
                contour_coords = np.round(contour).astype(int)
                valid_coords = (contour_coords[:, 0] >= 0) & (contour_coords[:, 0] < inst_mask.shape[0]) & \
                              (contour_coords[:, 1] >= 0) & (contour_coords[:, 1] < inst_mask.shape[1])
                contour_coords = contour_coords[valid_coords]
                
                if len(contour_coords) > 0:
                    # 绘制更粗的轮廓线（2像素宽度）
                    for i in range(len(contour_coords)):
                        y, x = contour_coords[i]
                        # 绘制2x2的像素块
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < inst_mask.shape[0] and 0 <= nx < inst_mask.shape[1]:
                                    contour_mask[ny, nx] = 1
                    
                    # 应用颜色overlay
                    for c in range(3):
                        overlay_image[:, :, c] = np.where(contour_mask > 0, 
                                                        color_normalized[c], 
                                                        overlay_image[:, :, c])
        
        return overlay_image
    
    def _create_improved_statistics_plot(self, ax, image, mask_data, type_map):
        """
        创建改进的统计信息图
        
        Args:
            ax: matplotlib轴对象
            image: 图像数据
            mask_data: mask数据字典
            type_map: 类型映射
        """
        np_mask = mask_data['np_map']
        hv_map = mask_data['hv_map']
        hv_h = hv_map[:, :, 1]
        hv_v = hv_map[:, :, 0]
        
        # 获取类型分布（忽略type 0）
        if type_map.ndim == 3:
            # 多通道格式
            actual_num_channels = type_map.shape[0]
            non_zero_types = []
            non_zero_counts = []
            non_zero_colors = []
            non_zero_names = []
            
            for i in range(actual_num_channels):
                type_id_str = str(i)
                if type_id_str in self.type_info:
                    type_name = self.type_info[type_id_str][0]
                    type_color = np.array(self.type_info[type_id_str][1]) / 255.0
                else:
                    type_name = f'Channel {i}'
                    type_color = [0.5, 0.5, 0.5]
                
                pixel_count = np.sum(type_map[i])
                if pixel_count > 0 and i > 0:  # 忽略type 0
                    non_zero_types.append(i)
                    non_zero_counts.append(pixel_count)
                    non_zero_colors.append(type_color)
                    non_zero_names.append(type_name)
        else:
            # 单通道格式
            unique_types, counts = np.unique(type_map, return_counts=True)
            non_zero_types = []
            non_zero_counts = []
            non_zero_colors = []
            non_zero_names = []
            
            for type_id, count in zip(unique_types, counts):
                if type_id > 0:  # 忽略type 0
                    type_id_str = str(type_id)
                    if type_id_str in self.type_info:
                        type_name = self.type_info[type_id_str][0]
                        type_color = np.array(self.type_info[type_id_str][1]) / 255.0
                    else:
                        type_name = f'Type {type_id}'
                        type_color = [0.5, 0.5, 0.5]
                    
                    non_zero_types.append(type_id)
                    non_zero_counts.append(count)
                    non_zero_colors.append(type_color)
                    non_zero_names.append(type_name)
        
        # 创建条形图
        if len(non_zero_types) > 0:
            bars = ax.bar(range(len(non_zero_types)), non_zero_counts, 
                         color=non_zero_colors, alpha=0.7)
            ax.set_title('Type Distribution (Excluding Background)')
            ax.set_xlabel('Type ID')
            ax.set_ylabel('Pixel Count')
            ax.set_xticks(range(len(non_zero_types)))
            ax.set_xticklabels([f'{t}' for t in non_zero_types])
            
            # 在右上角添加颜色图例
            legend_x = 0.98
            legend_y = 0.98
            for i, (type_id, type_name, type_color) in enumerate(zip(non_zero_types, non_zero_names, non_zero_colors)):
                # 绘制小方块
                rect = plt.Rectangle((legend_x - 0.15, legend_y - 0.05), 0.1, 0.03, 
                                   color=type_color, transform=ax.transAxes)
                ax.add_patch(rect)
                
                # 绘制类型名称
                ax.text(legend_x - 0.02, legend_y, f'{type_id}: {type_name}', 
                       transform=ax.transAxes, fontsize=8, 
                       verticalalignment='center', ha='right')
                legend_y -= 0.08
        else:
            ax.text(0.5, 0.5, 'No non-background types found', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Type Distribution (Excluding Background)')
    
    def _create_inst_rgb_mask(self, inst_map):
        """
        创建基于inst_map的RGB mask，每个instance随机分配一个颜色
        
        Args:
            inst_map: instance映射 (H, W)
            
        Returns:
            np.ndarray: RGB mask (H, W, 3)
        """
        # 获取所有唯一的instance ID
        unique_instances = np.unique(inst_map)
        unique_instances = unique_instances[unique_instances > 0]  # 排除背景
        
        # 创建RGB mask
        rgb_mask = np.zeros((*inst_map.shape, 3), dtype=np.uint8)
        
        # 为每个instance分配随机颜色
        np.random.seed(42)  # 设置随机种子以确保结果可重现
        for inst_id in unique_instances:
            # 生成随机RGB颜色
            color = np.random.randint(0, 256, 3)
            
            # 应用颜色到该instance的所有像素
            inst_mask = (inst_map == inst_id)
            rgb_mask[inst_mask] = color
        
        return rgb_mask
    
    def create_nuclei_overlay(self, image, mask_data, sample_name, split='train', save_overlay_only=False, overlay_output_dir=None):
        """
        创建基于inst_map和type_map的核颜色overlay可视化
        
        Args:
            image: 图像数据 (H, W, 3)
            mask_data: mask数据字典
            sample_name: 样本名称
            split: 数据分割
            save_overlay_only: 是否只保存overlay图像（不创建对比图）
            overlay_output_dir: overlay图像保存目录
        """
        inst_map = mask_data['inst_map']
        type_map = mask_data['type_map']
        
        # 创建RGB图像用于overlay
        overlay_image = image.copy().astype(np.float32) / 255.0
        
        # 获取所有唯一的instance ID
        unique_instances = np.unique(inst_map)
        unique_instances = unique_instances[unique_instances > 0]  # 排除背景
        
        # 为每个instance绘制overlay
        for inst_id in unique_instances:
            # 获取当前instance的mask
            inst_mask = (inst_map == inst_id)
            
            # 获取该instance在type_map中的类型
            if type_map.ndim == 3:
                # 多通道格式：找到该instance对应的类型
                inst_type_id = None
                for type_id_str in self.type_info.keys():
                    type_id = int(type_id_str)
                    if type_id == 0:  # 跳过背景
                        continue
                    if type_id >= type_map.shape[0]:  # 跳过超出实际通道数的类型
                        continue
                    
                    # 检查该instance是否属于这个类型
                    type_mask = type_map[type_id]
                    if np.any(inst_mask & (type_mask > 0)):
                        inst_type_id = type_id
                        break
                
                if inst_type_id is None:
                    continue
                    
                # 获取该类型的颜色
                type_id_str = str(inst_type_id)
                if type_id_str not in self.type_info:
                    continue
                type_name, type_color = self.type_info[type_id_str]
                
            else:
                # 单通道格式：直接获取该instance的类型
                inst_pixels = inst_mask & (type_map > 0)
                if not np.any(inst_pixels):
                    continue
                
                # 获取该instance的主要类型（出现最多的类型）
                inst_type_values = type_map[inst_pixels]
                inst_type_id = np.bincount(inst_type_values).argmax()
                
                # 获取该类型的颜色
                type_id_str = str(inst_type_id)
                if type_id_str not in self.type_info:
                    continue
                type_name, type_color = self.type_info[type_id_str]
            
            # 将RGB颜色归一化到[0,1]
            color_normalized = np.array(type_color) / 255.0
            
            # 找到该instance的轮廓
            contours = measure.find_contours(inst_mask.astype(float), 0.5)
            
            # 在overlay图像上绘制轮廓
            for contour in contours:
                # 创建轮廓mask
                contour_mask = np.zeros_like(inst_mask)
                contour_coords = np.round(contour).astype(int)
                valid_coords = (contour_coords[:, 0] >= 0) & (contour_coords[:, 0] < inst_mask.shape[0]) & \
                              (contour_coords[:, 1] >= 0) & (contour_coords[:, 1] < inst_mask.shape[1])
                contour_coords = contour_coords[valid_coords]
                
                if len(contour_coords) > 0:
                    # 绘制更粗的轮廓线（2像素宽度）
                    for i in range(len(contour_coords)):
                        y, x = contour_coords[i]
                        # 绘制2x2的像素块
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < inst_mask.shape[0] and 0 <= nx < inst_mask.shape[1]:
                                    contour_mask[ny, nx] = 1
                    
                    # 应用颜色overlay
                    for c in range(3):
                        overlay_image[:, :, c] = np.where(contour_mask > 0, 
                                                        color_normalized[c], 
                                                        overlay_image[:, :, c])
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'CD47 Nuclei Overlay Visualization - {sample_name} ({split})', fontsize=16)
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 核边缘overlay
        axes[1].imshow(overlay_image)
        axes[1].set_title('Nuclei Contours Overlay')
        axes[1].axis('off')
        
        # 在右侧添加颜色图例
        legend_ax = fig.add_axes([0.85, 0.1, 0.1, 0.8])
        legend_ax.axis('off')
        legend_ax.set_title('Color Legend', fontsize=12, color='black')
        
        # 获取实际使用的类型
        used_types = []
        if type_map.ndim == 3:
            # 多通道格式
            actual_num_channels = type_map.shape[0]
            for type_id_str in self.type_info.keys():
                type_id = int(type_id_str)
                if type_id == 0:  # 跳过背景
                    continue
                if type_id >= actual_num_channels:  # 跳过超出实际通道数的类型
                    continue
                if np.any(type_map[type_id] > 0):
                    used_types.append((type_id_str, self.type_info[type_id_str]))
        else:
            # 单通道格式
            unique_types = np.unique(type_map)
            unique_types = unique_types[unique_types > 0]  # 排除背景
            for type_id in unique_types:
                type_id_str = str(type_id)
                if type_id_str in self.type_info:
                    used_types.append((type_id_str, self.type_info[type_id_str]))
        
        # 绘制颜色图例
        legend_y = 0.9
        for type_id_str, (type_name, type_color) in used_types:
            type_id = int(type_id_str)
            color_normalized = np.array(type_color) / 255.0
            
            # 绘制彩色圆圈
            circle = plt.Circle((0.2, legend_y), 0.05, color=color_normalized, 
                              transform=legend_ax.transAxes)
            legend_ax.add_patch(circle)
            
            # 绘制文字（黑色）
            legend_ax.text(0.35, legend_y, f'{type_id}: {type_name}', 
                          transform=legend_ax.transAxes, 
                          color='black', fontweight='bold', fontsize=10,
                          verticalalignment='center')
            legend_y -= 0.08
        
        plt.tight_layout()
        
        # 保存对比可视化图像
        compare_output_path = self.output_dir / split / f'{sample_name}_{split}_overlay.png'
        plt.savefig(compare_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 如果指定了overlay输出目录，单独保存overlay图像
        if save_overlay_only and overlay_output_dir is not None:
            overlay_output_dir = Path(overlay_output_dir)
            overlay_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存overlay图像（转换为0-255范围）
            overlay_image_uint8 = (overlay_image * 255).astype(np.uint8)
            overlay_only_path = overlay_output_dir / f'{sample_name}.png'
            cv2.imwrite(str(overlay_only_path), cv2.cvtColor(overlay_image_uint8, cv2.COLOR_RGB2BGR))
            if not SILENT_MODE:
                print(f"Overlay图像已保存到: {overlay_only_path}")
        
        if not SILENT_MODE:
            print(f"核边缘overlay可视化已保存到: {compare_output_path}")
        
    
    def visualize_multiple_samples(self, split='train', num_samples=5, include_overlay=True):
        """
        可视化多个样本
        
        Args:
            split: 'train' 或 'val'
            num_samples: 要可视化的样本数量
            include_overlay: 是否包含核边缘overlay可视化
        """
        print(f"开始可视化 {split} 数据的 {num_samples} 个样本...")
        
        for i in range(num_samples):
            result = self.load_sample_data(split, i)
            if result is None:
                break
            
            image, mask_data, sample_name = result
            
            # 标准可视化
            self.visualize_sample(image, mask_data, sample_name, split)
            
            # 核边缘overlay可视化
            if include_overlay:
                self.create_nuclei_overlay(image, mask_data, sample_name, split)
        
        print(f"{split} 数据可视化完成！")
    
    def create_overview(self):
        """创建数据概览"""
        print("创建数据概览...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CD47 Dataset Overview', fontsize=16)
        
        # 统计train和val数据
        train_images = len(list((self.data_dir / 'images' / 'train').glob('*.tif')))
        val_images = len(list((self.data_dir / 'images' / 'val').glob('*.tif')))
        train_masks = len(list((self.data_dir / 'masks' / 'train').glob('*.npy')))
        val_masks = len(list((self.data_dir / 'masks' / 'val').glob('*.npy')))
        
        # Data statistics
        axes[0, 0].bar(['Train', 'Val'], [train_images, val_images])
        axes[0, 0].set_title('Image Count Statistics')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate([train_images, val_images]):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center')
        
        axes[0, 1].bar(['Train', 'Val'], [train_masks, val_masks])
        axes[0, 1].set_title('Mask Count Statistics')
        axes[0, 1].set_ylabel('Count')
        for i, v in enumerate([train_masks, val_masks]):
            axes[0, 1].text(i, v + 0.1, str(v), ha='center')
        
        # Data directory structure
        axes[1, 0].text(0.1, 0.9, 'Data Directory Structure:', transform=axes[1, 0].transAxes, fontweight='bold')
        axes[1, 0].text(0.1, 0.8, f'{self.data_dir}/', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.7, '├── images/', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.6, '│   ├── train/ (xxx.tif)', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.5, '│   └── val/ (xxx.tif)', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.4, '└── masks/', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.3, '    ├── train/ (xxx.npy)', transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.2, '    └── val/ (xxx.npy)', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Directory Structure')
        axes[1, 0].axis('off')
        
        # Mask data format
        axes[1, 1].text(0.1, 0.9, 'Mask Data Format:', transform=axes[1, 1].transAxes, fontweight='bold')
        axes[1, 1].text(0.1, 0.8, 'mask_data = {', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, "    'np_map': np_mask,", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, "    'hv_map': hv_map,", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"    'type_map': nc_mask", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, '}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, 'NP: Binary segmentation mask', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, 'HV: Distance vector map', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.1, f'NC: Single-channel classification mask', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Data Format')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存概览
        overview_path = self.output_dir / 'dataset_overview.png'
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if not SILENT_MODE:
            print(f"数据概览已保存到: {overview_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化CD47数据集')
    parser.add_argument('--data_dir', type=str,
                       default='/root/aMI_DATASET/SRSANet_dataset/CD47_SRSANet_traindata_256x256',
                       help='处理后的数据目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='要可视化的数据分割')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='要可视化的样本数量')
    parser.add_argument('--overview_only', action='store_true',
                       help='只创建数据概览')
    parser.add_argument('--no_overlay', action='store_true',
                       help='不生成核边缘overlay可视化')
    parser.add_argument('--type_info_path', type=str, default=None,
                       help='类型信息JSON文件路径')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = CD47DataVisualizer(args.data_dir, args.type_info_path)
    
    if args.overview_only:
        visualizer.create_overview()
    else:
        # 创建概览
        visualizer.create_overview()
        
        # 可视化样本
        include_overlay = not args.no_overlay
        visualizer.visualize_multiple_samples(args.split, args.num_samples, include_overlay)
    
    print("可视化完成！")

if __name__ == "__main__":
    main()
