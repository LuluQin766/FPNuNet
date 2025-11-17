# misc/wsi_handler.py

from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
import openslide
import tifffile as tiff
import matplotlib.pyplot as plt


def show_image(image, title='image'):
    """
    Display an image using matplotlib.
    """
    plt.figure(1)
    plt.axis("off")
    if image.ndim == 2 or image.shape[-1] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(title)
    plt.show()


class FileHandler:
    """
    Abstract base class for WSI file handlers.
    """

    def __init__(self):
        self.metadata = {
            "available_mag": None,
            "base_mag": None,
            "vendor": None,
            "base_mpp": None,
            "base_shape": None,
        }

    def __load_metadata(self):
        raise NotImplementedError

    def get_full_img(self, read_mag=None, read_mpp=None):
        raise NotImplementedError

    def read_region(self, coords, size):
        raise NotImplementedError

    def get_dimensions(self, read_mag=None, read_mpp=None):
        """
        Return the image dimensions at the requested magnification or MPP.
        """
        if read_mpp is not None:
            scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = scale * self.metadata["base_mag"]
        scale = read_mag / self.metadata["base_mag"]
        return (self.metadata["base_shape"] * scale).astype(np.int32)

    def prepare_reading(self, read_mag=None, read_mpp=None, cache_path=None):
        """
        Optionally cache downsampled image for fast access.
        """
        read_lv, scale_factor = self._get_read_info(read_mag, read_mpp)
        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self.get_full_img(read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode='r')
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        raise NotImplementedError


class OpenSlideHandler(FileHandler):
    """
    Handler for OpenSlide-supported WSIs.
    """

    def __init__(self, file_path):
        super().__init__()
        try:
            self.file_ptr = openslide.OpenSlide(file_path)
        except openslide.OpenSlideUnsupportedFormatError as e:
            raise e
        self.metadata = self.__load_metadata()
        self.image_ptr = None
        self.read_lv = None

    def __load_metadata(self):
        props = self.file_ptr.properties
        try:
            base_mag = float(props[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except:
            base_mag = 40.0
        try:
            mpp = [
                float(props[openslide.PROPERTY_NAME_MPP_X]),
                float(props[openslide.PROPERTY_NAME_MPP_Y])
            ]
        except:
            mpp = [0.252065, 0.252065]
        try:
            vendor = props[openslide.PROPERTY_NAME_VENDOR]
        except:
            vendor = "unknown"

        downsample = self.file_ptr.level_downsamples
        mags = [base_mag / ds for ds in downsample]

        return OrderedDict({
            "available_mag": mags,
            "base_mag": base_mag,
            "vendor": vendor,
            "base_mpp": np.array(mpp),
            "base_shape": np.array(self.file_ptr.dimensions),
        })

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], "Non-square MPP not supported"
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]

        mag_list = np.array(self.metadata["available_mag"])
        if read_mag in mag_list:
            read_lv = mag_list.tolist().index(read_mag)
            return read_lv, None

        # Find nearest lower level and scale manually
        base_mag = self.metadata["base_mag"]
        if read_mag > base_mag:
            scale = read_mag / base_mag
            return 0, scale
        else:
            # choose the closest higher mag level available
            diffs = mag_list - read_mag
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                return 0, read_mag / base_mag
            best_mag = mag_list[np.argmin(diffs)]
            best_lv = mag_list.tolist().index(best_mag)
            return best_lv, read_mag / best_mag

    def get_full_img(self, read_mag=None, read_mpp=None):
        read_lv, scale_factor = self._get_read_info(read_mag, read_mpp)
        img = self.file_ptr.read_region((0, 0), read_lv, self.file_ptr.level_dimensions[read_lv])
        img = np.array(img)[..., :3]
        if scale_factor:
            interp = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp)
        return img

    def read_region(self, coords, size):
        if self.image_ptr is None:
            lv_0_shape = np.array(self.file_ptr.level_dimensions[0])
            lv_r_shape = np.array(self.file_ptr.level_dimensions[self.read_lv])
            scale = lv_0_shape[0] / lv_r_shape[0]
            coord_0 = tuple(np.round(np.array(coords) * scale).astype(int))
            region = self.file_ptr.read_region(coord_0, self.read_lv, size)
            return np.array(region)[..., :3]
        else:
            x, y = coords
            w, h = size
            return self.image_ptr[y:y+h, x:x+w]


class GenericTiffHandler(FileHandler):
    """
    Fallback handler for non-OpenSlide-compatible TIFF images.
    """

    def __init__(self, file_path):
        super().__init__()
        try:
            self.image = Image.open(file_path)
            self.image.load()
        except Exception:
            img_array = tiff.imread(file_path)
            self.image = Image.fromarray(img_array)
        self.metadata = self.__load_metadata()

    def __load_metadata(self):
        w, h = self.image.size
        return OrderedDict({
            "available_mag": [1.0],
            "base_mag": 1.0,
            "vendor": "generic",
            "base_mpp": np.array([0.252065, 0.252065]),
            "base_shape": np.array([h, w]),
        })

    def read_region(self, coords, size):
        x, y = coords
        w, h = size
        x_end = min(x + w, self.image.width)
        y_end = min(y + h, self.image.height)
        region = self.image.crop((x, y, x_end, y_end))
        return np.array(region)

    def get_full_img(self, read_mag=None, read_mpp=None):
        return np.array(self.image)


def get_file_handler(path: str, backend: str):
    """
    Get a file handler based on backend or file extension.

    Args:
        path (str): Path to WSI.
        backend (str): File extension or format type.

    Returns:
        FileHandler: An instance of OpenSlideHandler or GenericTiffHandler.
    """
    openslide_exts = [
        '.svs', '.tif', '.vms', '.vmu', '.ndpi',
        '.scn', '.mrxs', '.tiff', '.svslide', '.bif',
    ]
    path = path.lower()
    if any(path.endswith(ext) for ext in openslide_exts):
        try:
            return OpenSlideHandler(path)
        except openslide.OpenSlideUnsupportedFormatError:
            print(f"[Warning] OpenSlide could not load {path}, falling back to GenericTiffHandler.")
            return GenericTiffHandler(path)
    else:
        raise ValueError(f"Unknown or unsupported backend format: '{backend}'")


# Example usage
if __name__ == '__main__':
    wsi_path = "/root/public_data/CD47/huada_data/regist_tif/B04754F5_HE_regist.tif"
    wsi_ext = ".tif"
    handler = get_file_handler(wsi_path, backend=wsi_ext)

    full_img = handler.get_full_img(read_mag=10.0)
    print("Loaded WSI shape:", full_img.shape)
    show_image(full_img)