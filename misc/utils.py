# misc/utils.py

import os
import shutil
import logging
import inspect
import numpy as np
import cv2
from scipy import ndimage


def normalize(mask, dtype=np.uint8):
    """
    Normalize the mask from [0, 1] to [0, 255] scale.

    Args:
        mask (ndarray): Input mask (typically float or int).
        dtype (np.dtype): Output data type.

    Returns:
        ndarray: Normalized mask.
    """
    max_val = np.amax(mask)
    if max_val == 0:
        return np.zeros_like(mask, dtype=dtype)
    return (255 * mask / max_val).astype(dtype)


def get_bounding_box(img):
    """
    Compute bounding box coordinates for the non-zero region of a binary image.

    Args:
        img (ndarray): Binary mask of shape (H, W).

    Returns:
        list: [rmin, rmax, cmin, cmax] bounding box coordinates.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    return [rmin, rmax + 1, cmin, cmax + 1]


def cropping_center(x, crop_shape, batch=False, verbose=False):
    """
    Center-crop image or batch of images.

    Args:
        x (ndarray): Input image of shape (H, W, C) or batch of shape (B, H, W, C).
        crop_shape (tuple): Target (height, width) for the crop.
        batch (bool): If True, treat the input as a batch (4D or 3D).
        verbose (bool): If True, print detailed crop information.

    Returns:
        ndarray: Center-cropped image or batch.
    """
    # print('\n -----  cropping_center in misc/utils.py  -----')
    # print(' ----- cropping_center: x.shape:', x.shape, 'crop_shape:', crop_shape)
    orig_shape = x.shape
    h_crop, w_crop = crop_shape

    if not batch:
        assert orig_shape[0] >= h_crop and orig_shape[1] >= w_crop, \
            f"Crop size {crop_shape} is larger than input shape {orig_shape[:2]}"
        h0 = (orig_shape[0] - h_crop) // 2
        w0 = (orig_shape[1] - w_crop) // 2
        cropped = x[h0:h0 + h_crop, w0:w0 + w_crop, ...]
    else:
        assert orig_shape[1] >= h_crop and orig_shape[2] >= w_crop, \
            f"Crop size {crop_shape} is larger than input shape {orig_shape[1:3]}"
        h0 = (orig_shape[1] - h_crop) // 2
        w0 = (orig_shape[2] - w_crop) // 2
        cropped = x[:, h0:h0 + h_crop, w0:w0 + w_crop, ...]

    if verbose:
        print(f"[cropping_center] Original shape: {orig_shape}, Crop shape: {crop_shape}, "
              f"Start indices: (h0={h0}, w0={w0}) → Cropped shape: {cropped.shape}")

    return cropped

def rm_n_mkdir(dir_path):
    """Remove and recreate a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)


def mkdir(dir_path):
    """Create directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)


def get_inst_centroid(inst_map):
    """
    Compute centroids of instances in the instance map.

    Args:
        inst_map (ndarray): Instance label map (H, W), with 0 as background.

    Returns:
        ndarray: N x 2 array of (x, y) centroids.
    """
    inst_centroid_list = []
    for inst_id in np.unique(inst_map)[1:]:  # skip background
        mask = (inst_map == inst_id).astype(np.uint8)
        moment = cv2.moments(mask)
        if moment["m00"] != 0:
            cx = moment["m10"] / moment["m00"]
            cy = moment["m01"] / moment["m00"]
            inst_centroid_list.append([cx, cy])
    return np.array(inst_centroid_list)


def center_pad_to_shape(img, size, cval=255):
    """
    Pad image to a given shape, centering the original image.

    Args:
        img (ndarray): Input image of shape (H, W[, C]).
        size (tuple): Desired size (height, width).
        cval (int): Padding value.

    Returns:
        ndarray: Padded image.
    """
    h_pad = max(size[0] - img.shape[0], 0)
    w_pad = max(size[1] - img.shape[1], 0)
    pad_shape = ((h_pad // 2, h_pad - h_pad // 2),
                 (w_pad // 2, w_pad - w_pad // 2))
    if img.ndim == 3:
        pad_shape += ((0, 0),)
    return np.pad(img, pad_shape, mode='constant', constant_values=cval)


def color_deconvolution(rgb, stain_mat):
    """
    Perform color deconvolution on an RGB image.

    Args:
        rgb (ndarray): RGB image (H, W, 3).
        stain_mat (ndarray): 3x3 stain matrix.

    Returns:
        ndarray: Deconvolved image of shape (H, W, 3).
    """
    log255 = np.log(255)
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    return np.floor(output + 0.5).astype(np.uint8)


def log_debug(msg):
    """
    Print indented debug message based on call depth.
    frame_info includes: frame, filename, line_number, function_name, lines, index
    """
    frame_info = inspect.getouterframes(inspect.currentframe())[1]
    indentation = frame_info.code_context[0].find(frame_info.code_context[0].lstrip())
    logging.debug(f"{'.' * indentation} {msg}")


def log_info(msg):
    """
    Print indented info message based on call depth.
    frame_info includes: frame, filename, line_number, function_name, lines, index
    """
    frame_info = inspect.getouterframes(inspect.currentframe())[1]
    indentation = frame_info.code_context[0].find(frame_info.code_context[0].lstrip())
    logging.info(f"{'.' * indentation} {msg}")


def remove_small_objects(pred, min_size=64, connectivity=1):
    """
    Remove small connected components which are less than min_size from a labeled image.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred (ndarray): Labeled image.
        min_size (int): Minimum size of objects to keep.
        connectivity (int): Pixel connectivity.

    Returns:
        ndarray: Filtered labeled image.
    """
    if min_size == 0:
        return pred

    if pred.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = pred

    try:
        sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = sizes < min_size
    pred[too_small[ccs]] = 0
    return pred


def show_img_gray(img, img_title, gray, gray_title, show=True, save_path=None, dpi=150):
    """
    Show image and grayscale image side by side using matplotlib.

    Args:
        img (ndarray): Original image, expected shape (H, W, 3) or (H, W).
        img_title (str): Title for original image.
        gray (ndarray): Grayscale image, expected shape (H, W).
        gray_title (str): Title for grayscale image.
        show (bool): Whether to display the image using plt.show().
        save_path (str or None): If given, saves the figure to this path.
        dpi (int): Resolution for saved image if save_path is given.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[0].set_title(img_title)
    axes[0].axis('off')

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title(gray_title)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"Saved image to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()