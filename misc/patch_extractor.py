# misc/patch_extractor.py

import sys
sys.path.append('/root/SAM2PATH-main')

import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils import cropping_center

from misc.viz_utils import random_inst_color_map, colorize_type_map
from dataset_process.hovernet_gt_generator import colorize_hv_map

color_dict = {
            "0" : ["background", [255,   1, 255]],  # Blue
            "1" : ["other", [255,   1,   1]],  # Yellow
            "2" : ["inflammatory", [1  , 255,   1]],  # Green
            "3" : ["epithelial", [1  ,   1, 255]],  # Red
            "4" : ["spindle", [255, 255,   1]]  # Cyan
        }


def show_xsource_xpadded(x, x_padded, save_path=None):
    """
    show the image and annotation

    Args:
        x (ndarray): Input numpy array of shape (H, W, C).
        x_padded (ndarray): Padded numpy array of shape (H, W, C).
    """

    def split_data(x, data_name='x'):
        img = x[..., :3]
        inst_map = x[..., 3]
        type_map = x[..., 4]
        h_map = x[..., 5]
        v_map = x[..., 6]

        print(f"\n ------------ {data_name} stats:")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}, max: {img.max()}, min: {img.min()}")
        print(f"inst_map shape: {inst_map.shape}, dtype: {inst_map.dtype}, max: {inst_map.max()}, min: {inst_map.min()}, unique: {np.unique(inst_map)}")
        print(f"type_map shape: {type_map.shape}, dtype: {type_map.dtype}, max: {type_map.max()}, min: {type_map.min()}, unique: {np.unique(type_map)}")
        print(f"h_map shape: {h_map.shape}, dtype: {h_map.dtype}, max: {h_map.max()}, min: {h_map.min()}")
        print(f"v_map shape: {v_map.shape}, dtype: {v_map.dtype}, max: {v_map.max()}, min: {v_map.min()}\n")

        inst_map_rgb = random_inst_color_map(inst_map)
        type_map_rgb = colorize_type_map(type_map, color_dict)
        h_map_rgb = colorize_hv_map(h_map+1, 0, 2)
        v_map_rgb = colorize_hv_map(v_map+1, 0, 2)

        titles = ["Image", "Inst Map", "Type Map", "H Map", "V Map"]        
        titles = [data_name+":"+t for t in titles]

        return (
            [img, inst_map_rgb, type_map_rgb, h_map_rgb, v_map_rgb],
            titles
        )


    print("\n ===== Showing the image and annotation ====")
    if save_path is not None:
        print(f"save_path: {save_path}")
    x_img_list, x_titles = split_data(x, 'x')
    x_padded_img_list, x_padded_titles = split_data(x_padded, 'x_padded')

    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for i, (img, title, padded_img, padded_title) in enumerate(zip(x_img_list, x_titles, x_padded_img_list, x_padded_titles)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        axes[1, i].imshow(padded_img)
        axes[1, i].set_title(padded_title)
        axes[1, i].axis('off')
    
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

class PatchExtractor:
    """
    Extractor to generate image patches with or without padding.

    Args:
        win_size (tuple): Patch window size (height, width)
        step_size (tuple): Stride size (height, width)
        debug (bool): Whether to visualize patches for debugging
        debug_delay (float): Time delay in seconds for debug visualization

    Example:
        >>> extractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = extractor.extract(img, 'mirror')
    """

    def __init__(self, win_size: tuple, step_size: tuple, debug: bool = False, debug_delay: float = 0.5):
        self.win_size = win_size
        self.step_size = step_size
        self.debug = debug
        self.debug_delay = debug_delay
        self.counter = 0
        self.patch_type = "mirror"

    def __get_patch(self, x: np.ndarray, ptx: tuple) -> np.ndarray:
        """
        Extract a single patch from location ptx.

        Args:
            x (ndarray): Input image (HWC).
            ptx (tuple): Top-left corner of the patch.

        Returns:
            ndarray: Cropped patch of size self.win_size.
        """
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        patch = x[ptx[0]:pty[0], ptx[1]:pty[1]]

        assert patch.shape[0] == self.win_size[0] and patch.shape[1] == self.win_size[1], \
            f"[BUG] Incorrect Patch Size {patch.shape}, expected {self.win_size}"

        if self.debug:
            vis_patch = patch.copy()
            cen = cropping_center(vis_patch, self.step_size)
            cen[..., self.counter % 3].fill(150)
            cv2.rectangle(vis_patch, (0, 0), (self.win_size[1]-1, self.win_size[0]-1), (255, 0, 0), 2)
            plt.imshow(vis_patch)
            plt.title(f"Patch #{self.counter}")
            plt.axis('off')
            plt.show(block=False)
            plt.pause(self.debug_delay)
            plt.close()
            self.counter += 1

        return patch, (ptx[0], pty[0], ptx[1], pty[1])

    def __extract_valid(self, x: np.ndarray) -> list:
        """
        Extract patches without padding.

        This method slides a window over the image. If the image does not divide evenly,
        additional patches are taken at the right and bottom edges.

        Args:
            x (ndarray): Input image of shape (H, W, C)

        Returns:
            list of ndarray: Extracted patches.
        """
        im_h, im_w = x.shape[:2]

        def compute_last_step(length, win, step):
            has_tail = (length - win) % step != 0
            last_start = ((length - win) // step) * step
            return has_tail, last_start + step

        h_flag, h_last = compute_last_step(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = compute_last_step(im_w, self.win_size[1], self.step_size[1])

        sub_patches = []
        sub_patches_coords = []

        # Main grid
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                patch, coords = self.__get_patch(x, (row, col))
                sub_patches.append(patch)
                sub_patches_coords.append(coords)

        # Bottom edge
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                patch, coords = self.__get_patch(x, (row, col))
                sub_patches.append(patch)
                sub_patches_coords.append(coords)

        # Right edge
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                patch, coords = self.__get_patch(x, (row, col))
                sub_patches.append(patch)
                sub_patches_coords.append(coords)

        # Bottom-right corner
        if h_flag and w_flag:
            patch, coords = self.__get_patch(x, (im_h - self.win_size[0], im_w - self.win_size[1]))
            sub_patches.append(patch)
            sub_patches_coords.append(coords)

        return sub_patches, sub_patches_coords

    def __extract_mirror(self, x: np.ndarray) -> list:
        """
        Extract patches with mirrored padding at boundaries.

        Ensures each patch's central region lies within original image coverage.

        Args:
            x (ndarray): Input image (H, W, C)

        Returns:
            list of ndarray: Extracted patches. 
            list of tuple: Coordinates of each patch (top, bottom, left, right).
        """
        # print("\n -------- Extracting patches with mirror padding...")
        # print(" x.shape: ", x.shape)
        # print(" win_size: ", self.win_size)
        # print(" step_size: ", self.step_size)
        # print(" x.ndim: ", x.ndim)
        # print(" x.shape[2] > x.shape[0] = ", (x.shape[2] > x.shape[0]))
        assert x.ndim == 3 and x.shape[0] > 10 and x.shape[1] > 10 and x.shape[2] < 10, "Expected HWC image as input."

        # assert x.ndim == 3 and x.shape[2] in [1, 3, 4], "Expected HWC image as input."

        # diff_h = self.win_size[0] - self.step_size[0]
        # diff_w = self.win_size[1] - self.step_size[1]
        # pad_t, pad_b = diff_h // 2, diff_h - (diff_h // 2)
        # pad_l, pad_r = diff_w // 2, diff_w - (diff_w // 2)

        # print("pad_t, pad_b, pad_l, pad_r: ", pad_t, pad_b, pad_l, pad_r)

        # pad_type = "constant" if self.debug else "reflect"
        # x_padded = np.pad(x, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=pad_type)

        # # for debugging
        # show_xsource_xpadded(x, x_padded)

        # return self.__extract_valid(x_padded), (pad_t, pad_b, pad_l, pad_r)

        return self.__extract_valid(x)

    def extract(self, x: np.ndarray, patch_type: str = "mirror") -> list:
        """
        Extract patches from the input image.

        Args:
            x (ndarray): Input image (H, W, C)
            patch_type (str): 'valid' or 'mirror'

        Returns:
            list of ndarray: Extracted patches.
        """
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == "valid":
            return self.__extract_valid(x)
        elif patch_type == "mirror":
            return self.__extract_mirror(x)
        else:
            raise ValueError(f"Unknown patch_type: '{patch_type}' (expected 'valid' or 'mirror')")


if __name__ == "__main__":
    # Toy example
    img = np.full((1200, 1200, 3), 255, dtype=np.uint8)
    extractor = PatchExtractor((450, 450), (120, 120), debug=True, debug_delay=0.3)

    print("Extracting patches with mirror padding...")
    patches_mirror = extractor.extract(img, "mirror")
    print(f"Total patches (mirror): {len(patches_mirror)}")

    print("Extracting patches with valid sliding...")
    patches_valid = extractor.extract(img, "valid")
    print(f"Total patches (valid): {len(patches_valid)}")
