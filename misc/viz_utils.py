# misc/viz_utils.py

import os
import cv2
import math
import random
import colorsys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import get_bounding_box, normalize


def colorize(channel: np.ndarray, vmin: float = None, vmax: float = None,
             cmap_name: str = "jet") -> np.ndarray:
    """
    Convert a single-channel array into a colorized RGB image using a colormap.

    Args:
        channel (ndarray): Input 2D array (H, W).
        vmin (float, optional): Minimum value for normalization. If None, use channel.min().
        vmax (float, optional): Maximum value for normalization. If None, use channel.max().
        target_shape (tuple, optional): (H, W) to pad/crop to this shape. If None, no resizing.
        cmap_name (str): Matplotlib colormap name, e.g., "jet", "viridis", etc.

    Returns:
        ndarray: Colorized RGB image (H, W, 3) in uint8.
    """
    if channel.ndim == 3:
        channel = np.squeeze(channel)  # remove channel dimension if it exists
    # print(f" \n ----------- Colorize channel shape: {channel.shape}, dtype: {channel.dtype}")
    assert channel.ndim == 2, "Input channel must be a 2D array."

    ch = channel.astype(np.float32)
    vmin = ch.min() if vmin is None else vmin
    vmax = ch.max() if vmax is None else vmax
    ch_norm = np.clip((ch - vmin) / (vmax - vmin + 1e-16), 0, 1)

    cmap = plt.get_cmap(cmap_name)
    ch_rgb = (cmap(ch_norm)[..., :3] * 255).astype(np.uint8)

    return ch_rgb


def random_colors(N, bright=True):
    """
    Generate visually distinct colors using HSV space.

    Args:
        N (int): Number of colors to generate.
        bright (bool): Whether to use bright colors.

    Returns:
        list: List of RGB tuples (R, G, B) with values in [0, 1].
    """
    # brightness = 1.0 if bright else 0.7
    # hsv = [(i / N, 1, brightness) for i in range(N)]
    # print(f" \n - random_colors: hsv: {hsv}")
    # print(f" \n - random_colors: len(hsv): {len(hsv)}")
    # colors = list(map(colorsys.hsv_to_rgb, hsv))
    # random.shuffle(colors)
    # return colors

    import colorsys
    import random
    import itertools

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(itertools.starmap(colorsys.hsv_to_rgb, hsv))  # ✅ 解包 tuple
    random.shuffle(colors)
    return colors
    


def visualize_instances_map(input_image, inst_map, type_map=None, type_colour=None, line_thickness=2):
    """
    Overlay instance segmentation contours on the input image.

    Args:
        input_image (ndarray): RGB or grayscale image (H, W, 3).
        inst_map (ndarray): Instance label map (H, W).
        type_map (ndarray or None): Optional type map (H, W).
        type_colour (dict): Optional mapping from type id to color (R, G, B).
        line_thickness (int): Contour thickness.

    Returns:
        ndarray: Overlayed image (H, W, 3).
    """
    overlay = input_image.copy().astype(np.uint8)
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)  # remove background

    inst_colors = (np.array(random_colors(len(inst_list))) * 255).astype(np.uint8)

    for i, inst_id in enumerate(inst_list):
        mask = (inst_map == inst_id).astype(np.uint8)
        y1, y2, x1, x2 = get_bounding_box(mask)
        y1, x1 = max(0, y1 - 2), max(0, x1 - 2)
        y2, x2 = min(inst_map.shape[0], y2 + 2), min(inst_map.shape[1], x2 + 2)

        crop = mask[y1:y2, x1:x2]
        contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = np.squeeze(contours[0]).astype(np.int32)
        if contour.ndim != 2:
            continue  # skip invalid contours

        contour += np.array([x1, y1])  # offset

        if type_map is not None and type_colour is not None:
            type_crop = type_map[y1:y2, x1:x2]
            type_id = np.max(type_crop)
            inst_color = type_colour.get(type_id, inst_colors[i].tolist())
        else:
            inst_color = inst_colors[i].tolist()

        cv2.drawContours(overlay, [contour], -1, inst_color, line_thickness)

    return overlay


def visualize_instances_dict(input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2):
    """
    Overlay instance dictionary (e.g. HoVerNet-style) on image.

    Args:
        input_image (ndarray): RGB input image (H, W, 3).
        inst_dict (dict): Dictionary with instance keys. Each value has keys: "contour", "type", "centroid".
        draw_dot (bool): Whether to draw centroids.
        type_colour (dict): Optional dict mapping from type_id to (label, color).
        line_thickness (int): Contour line thickness.

    Returns:
        ndarray: Overlayed image.
    """
    overlay = input_image.copy()
    # print(f" \n - visualize_instances_dict: inst_dict keys: {inst_dict.keys()}")
    # 打印前5个实例的信息
    # for i, (inst_id, inst_info) in enumerate(inst_dict.items()):
    #     if i == 5:
    #         break
    #     print(f" \n - visualize_instances_dict: inst_id: {inst_id}, inst_info: {inst_info}")
    # print(f" \n - visualize_instances_dict: type_colour: {type_colour}")

    inst_colors = (np.array(random_colors(len(inst_dict))) * 255).astype(np.uint8)

    for idx, (inst_id, inst_info) in enumerate(inst_dict.items()):
        contour = inst_info["contour"]
        color = inst_colors[idx].tolist()

        if "type" in inst_info and type_colour is not None:
            color = type_colour.get(inst_info["type"], (None, color))[1]

        cv2.drawContours(overlay, [contour], -1, color, line_thickness)

        if draw_dot and "centroid" in inst_info:
            centroid = tuple(map(int, inst_info["centroid"]))
            overlay = cv2.circle(overlay, centroid, 3, (255, 0, 0), -1)

    return overlay


def gen_figure(imgs_list, titles, fig_inch, shape=None, share_ax="all", show=False, colormap=plt.get_cmap("jet")):
    """
    Generate a matplotlib figure showing a grid of images.

    Args:
        imgs_list (list): List of images.
        titles (list): List of titles (same length as imgs_list).
        fig_inch (tuple): Figure size in inches.
        shape (tuple or None): (nrows, ncols) of subplots. If None, auto-layout.
        share_ax (str): Share axes ('all', 'none', etc.).
        show (bool): Whether to show the figure.
        colormap (Colormap): Matplotlib colormap.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    num_imgs = len(imgs_list)
    if num_imgs == 0:
        raise ValueError("Empty image list for figure generation.")

    if shape is None:
        ncols = math.ceil(math.sqrt(num_imgs))
        nrows = math.ceil(num_imgs / ncols)
    else:
        nrows, ncols = shape

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_inch, sharex=share_ax, sharey=share_ax)
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx >= num_imgs:
            ax.axis('off')
            continue
        ax.set_title(titles[idx])
        ax.imshow(imgs_list[idx], cmap=colormap)
        ax.axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    return fig

def random_inst_color_map(inst_map):
    """
    Assign a random color to each instance ID in the map (except 0).
    """
    # if inst_map is tensor, convert to numpy
    if isinstance(inst_map, torch.Tensor):
        inst_map = inst_map.numpy()
    output = np.zeros((*inst_map.shape, 3), dtype=np.uint8)
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        color = [random.randint(50, 255) for _ in range(3)]
        output[inst_map == inst_id] = color
    return output

def colorize_type_map(type_map, color_dict):
    """
    Convert type_map to a color RGB image based on color_dict.
    """
    if type_map.ndim!= 2:
        type_map = np.squeeze(type_map)

    h, w = type_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for k, v in color_dict.items():
        if k == "0":
            continue
        label = int(k)
        color = v[1]
        output[type_map == label] = color
    return output  

def apply_jet_colormap(hv_map: torch.Tensor, vmin: float, vmax: float, shape: tuple) -> np.ndarray:
    """
    Normalize and apply jet colormap to a single-channel H/V map.
    Args:
        hv_map (torch.Tensor): H/V map with shape (H, W).
        vmin (float): Min value for normalization.
        vmax (float): Max value for normalization.
        shape (tuple): Output shape (H, W, 3).
    Returns:
    """
    cmap = plt.get_cmap("jet")
    if isinstance(hv_map, torch.Tensor):
        hv_map = hv_map.numpy()

    hv_map = hv_map.astype(np.float32)
    hv_map_norm = np.clip((hv_map - vmin) / (vmax - vmin + 1e-16), 0, 1)
    hv_rgb = (cmap(hv_map_norm)[..., :3] * 255).astype(np.uint8)
    return hv_rgb

def visualize_maps_batch_train(
    images, gt_bin_map, gt_h_map, gt_v_map, gt_type_map,
    gt_inst_map, pred_bin_map, pred_h_map, pred_v_map, pred_type_map,
    color_dict, save_root=None, save_name=None, dpi=150
):
    """
    Visualize GT and predicted maps in two rows (GT, Pred) for each sample in the batch.
    """
    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)

    B = images.shape[0]
    B = 1 # only visualize the first sample in the batch, default to 1
    # print("\n ====== visualize_maps_batch_train ====== ")

    for i in range(B):
        # 打印元信息
        # print(f"\n --------- Sample {i}:")
        # print(f"Image shape: {images.shape}, dtype: {images.dtype}, min: {images.min()}, max: {images.max()}")
        # print(f"GT Instance Map shape: {gt_inst_map.shape}, dtype: {gt_inst_map.dtype}, unique values: {np.unique(gt_inst_map)}")
        # print(f"GT Binary Map shape: {gt_bin_map.shape}, dtype: {gt_bin_map.dtype}, min: {gt_bin_map.min()}, max: {gt_bin_map.max()}")
        # print(f"GT H Map shape: {gt_h_map.shape}, dtype: {gt_h_map.dtype}, min: {gt_h_map.min()}, max: {gt_h_map.max()}")
        # print(f"GT V Map shape: {gt_v_map.shape}, dtype: {gt_v_map.dtype}, min: {gt_v_map.min()}, max: {gt_v_map.max()}")
        # print(f"GT Type Map shape: {gt_type_map.shape}, dtype: {gt_type_map.dtype}, unique values: {np.unique(gt_type_map)}\n")
        # print(f"Pred Binary Map shape: {pred_bin_map.shape}, dtype: {pred_bin_map.dtype}, min: {pred_bin_map.min()}, max: {pred_bin_map.max()}")
        # print(f"Pred H Map shape: {pred_h_map.shape}, dtype: {pred_h_map.dtype}, min: {pred_h_map.min()}, max: {pred_h_map.max()}")
        # print(f"Pred V Map shape: {pred_v_map.shape}, dtype: {pred_v_map.dtype}, min: {pred_v_map.min()}, max: {pred_v_map.max()}")
        # print(f"Pred Type Map shape: {pred_type_map.shape}, dtype: {pred_type_map.dtype}, unique values: {np.unique(pred_type_map)}\n")

        # Get image and normalize to [0, 255]
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

        # GT maps
        inst_rgb = random_inst_color_map(gt_inst_map[i].detach().cpu())
        bin_gt = gt_bin_map[i].detach().cpu().squeeze().numpy()
        bin_gt = np.stack([bin_gt * 255]*3, axis=-1).astype(np.uint8)

        type_gt = gt_type_map[i].detach().cpu().numpy()
        # print(f"type_gt shape: {type_gt.shape}, dtype: {type_gt.dtype}, unique values: {np.unique(type_gt)}")
        type_rgb = colorize_type_map(type_gt, color_dict)

        h_gt = colorize(gt_h_map[i].detach().cpu().squeeze().numpy()+1, 0, 2)
        v_gt = colorize(gt_v_map[i].detach().cpu().squeeze().numpy()+1, 0, 2)

        # Predicted maps
        h_pred = colorize(pred_h_map[i].detach().cpu().squeeze().numpy()+1, 0, 2)
        v_pred = colorize(pred_v_map[i].detach().cpu().squeeze().numpy()+1, 0, 2)
        
        bin_pred = pred_bin_map[i].detach().cpu().numpy()
        bin_pred = np.stack([bin_pred * 255]*3, axis=-1).astype(np.uint8)

        type_pred = pred_type_map[i].detach().cpu().numpy()
        # print(f"type_pred shape: {type_pred.shape}, dtype: {type_pred.dtype}, unique values: {np.unique(type_pred)}")

        pred_type_rgb = colorize_type_map(type_pred, color_dict)

        # print(f" show GT and predicted maps for sample {i} ")
        # print(f"GT Instance Map shape: {inst_rgb.shape}, dtype: {inst_rgb.dtype}")
        # print(f"GT Binary Map shape: {bin_gt.shape}, dtype: {bin_gt.dtype}")
        # print(f"GT Type Map shape: {type_rgb.shape}, dtype: {type_rgb.dtype}")
        # print(f"GT H Map shape: {h_gt.shape}, dtype: {h_gt.dtype}")
        # print(f"GT V Map shape: {v_gt.shape}, dtype: {v_gt.dtype}")
        # print(f"Pred H Map shape: {h_pred.shape}, dtype: {h_pred.dtype}")
        # print(f"Pred V Map shape: {v_pred.shape}, dtype: {v_pred.dtype}")
        # print(f"Pred Binary Map shape: {bin_pred.shape}, dtype: {bin_pred.dtype}")
        # print(f"Pred Type Map shape: {pred_type_rgb.shape}, dtype: {pred_type_rgb.dtype}\n")

        # Two rows of subplots
        imgs_list_01 = [img, bin_gt, h_gt, v_gt, type_rgb]
        titles_01 = ["Input Image", "GT Binary Mask", "GT H Map", "GT V Map", "GT Type Map"]

        imgs_list_02 = [inst_rgb, bin_pred, h_pred, v_pred, pred_type_rgb]
        titles_02 = ["GT Instance Map", "Pred Binary Mask", "Pred H Map", "Pred V Map", "Pred Type Map"]

        fig, axes = plt.subplots(2, 5, figsize=(16, 8), dpi=dpi)

        for ax, img_, title in zip(axes[0], imgs_list_01, titles_01):
            ax.imshow(img_)
            ax.set_title(title)
            ax.axis("off")

        for ax, img_, title in zip(axes[1], imgs_list_02, titles_02):
            ax.imshow(img_)
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        plt.show()

        # Save figure
        if save_root is not None:
            save_path = os.path.join(save_root, save_name)
            plt.savefig(save_path)
            # print(f"Saved visualization to {save_path}")
        plt.close()


# === 可视化交叉掩码比较函数 ===
def compare_bin_masks(gt_mask, pred_mask):
    """
    返回 RGB 颜色编码图：
    - green  ：交集（正确预测）
    - red    ：GT 有但 Pred 没有（漏检）
    - yellow ：Pred 有但 GT 没有（误检）
    """
    cmp_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    intersection = (gt_mask == 1) & (pred_mask == 1)
    gt_only = (gt_mask == 1) & (pred_mask == 0)
    pred_only = (gt_mask == 0) & (pred_mask == 1)
    
    cmp_rgb[intersection] = [0, 255, 0]      # Green
    cmp_rgb[gt_only] = [255, 0, 0]           # Red
    cmp_rgb[pred_only] = [255, 255, 0]       # Yellow
    return cmp_rgb

def get_boundary_map_from_inst_map(inst_map, mode='inner'):
    """
    Generate a binary boundary map from an instance segmentation map.

    Args:
        inst_map (np.ndarray): 2D array of shape (H, W), where each non-zero integer represents an instance ID.
        mode (str): 'inner' (default) marks all boundaries between instances.
                    'outer' marks only the outer boundaries (useful for touching instances).
                    Options: 'inner', 'outer', or 'thick'.

    Returns:
        np.ndarray: A binary boundary map of shape (H, W), dtype=np.uint8.
    """
    from skimage.segmentation import find_boundaries
    assert inst_map.ndim == 2, "inst_map should be a 2D array"
    assert mode in ['inner', 'outer', 'thick'], "Invalid mode"

    boundary_map = find_boundaries(inst_map, mode=mode).astype(np.uint8)

    # 对边界进行膨胀操作，使得边界更加明显
    import skimage.morphology as morph
    boundary_map = morph.dilation(boundary_map, morph.square(3))
    
    return boundary_map

def visualize_intermediate_images(intermediate_image: dict, img_name: str = "Prediction Overview", save_path: str = None):
    """
    Visualize intermediate image results from a dictionary.

    Args:
        intermediate_image (dict): Dictionary containing numpy arrays of images to display.
        img_name (str): Title for the whole figure.
    """
    need_norm_keys = ['images_colorfusion_sam', 'images_colorfusion_uni', 'images_colorfusion_fusion']

    for k in need_norm_keys:
        if k in intermediate_image.keys():
            # get the image and normalize to [0, 1], then save it to the dictionary as a new key: f"{k}_norm"
            img = intermediate_image[k]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            intermediate_image[f"{k}_norm"] = img_norm

    # print(f"\n\n ----- in visualize_intermediate_images(), img_name={img_name}")
    # for k, v in intermediate_image.items():
    #     if isinstance(v, np.ndarray):
    #         print(f"  {k}: {v.shape} (np.dtype: {v.dtype}), value range: [{v.min()}, {v.max()}]")
    #     elif isinstance(v, dict):
    #         for k2, v2 in v.items():
    #             if isinstance(v2, np.ndarray):
    #                 print(f"   --- {k}.{k2}: {v2.shape} (np.dtype: {v2.dtype}), value range: [{v2.min()}, {v2.max()}]")
    #             else:
    #                 print(f"   --- {k}.{k2}: {v2} (type: {type(v2)}), Not a numpy array")
    #     else:
    #         print(f"  {k}: {v} (type: {type(v)}), not a numpy array or a dict")
    # print("\n")


    # Filter to only valid ndarray keys
    image_items = [(k, v) for k, v in intermediate_image.items() if isinstance(v, np.ndarray)]

    num_images = len(image_items)
    cols = 4
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(cols * 4, rows * 4))
    for idx, (key, img) in enumerate(image_items):
        plt.subplot(rows, cols, idx + 1)
        
        # Handle grayscale and RGB
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        elif img.ndim == 3 and img.shape[2] == 1:
            plt.imshow(img.squeeze(-1), cmap='gray')
        elif img.ndim == 3 and img.shape[2] == 3:
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "Unsupported Shape", ha='center', va='center')
            print(f"[WARN] Cannot display: {key}, shape={img.shape}")
        
        plt.title(key)
        plt.axis('off')

    plt.suptitle(img_name, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

