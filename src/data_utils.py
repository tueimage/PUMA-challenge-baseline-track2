import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Callable
from skimage.morphology import remove_small_objects, disk, dilation
import PIL
import pathlib
import cv2
from src.constants import LUT_MAGNIFICATION_MPP, LUT_MAGNIFICATION_X
from shutil import copy2, copytree
import os


def copy_img(im_path, cache_dir):
    """
    Helper function to copy WSI to cache directory

    Parameters
    ----------
    im_path : str
        path to the WSI
    cache_dir : str
        path to the cache directory

    Returns
    -------
    str
        path to the copied WSI
    """
    file, ext = os.path.splitext(im_path)
    if ext == ".mrxs":
        copy2(im_path, cache_dir)
        copytree(
            file, os.path.join(cache_dir, os.path.split(file)[-1]), dirs_exist_ok=True
        )
    else:
        copy2(im_path, cache_dir)
    return os.path.join(cache_dir, os.path.split(im_path)[-1])


def normalize_min_max(x: np.ndarray, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """
    Min max scaling for input array

    Parameters
    ----------
    x : np.ndarray
        input array
    mi : float or int
        minimum value
    ma : float or int
        maximum value
    clip : bool, optional
        clip values be between 0 and 1, False by default
    eps : float
        epsilon value to avoid division by zero
    dtype : type
        data type of the output array

    Returns
    -------
    np.ndarray
        normalized array
    """
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x


def center_crop(t, croph, cropw):
    """
    Center crop input tensor in last two axes to height and width
    """
    h, w = t.shape[-2:]
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[..., starth : starth + croph, startw : startw + cropw]


class NpyDataset(Dataset):
    def __init__(
        self,
        path,
        crop_size_px,
        padding_factor=0.5,
        remove_bg=True,
        ratio_object_thresh=5e-1,
        min_tiss=0.1,
    ):
        """
        Torch Dataset to load from NPY files.

        Parameters
        ----------
        path : str
            Path to the NPY file.
        crop_size_px : int
            Size of the extracted tiles in pixels. e.g 256 -> 256x256 tiles
        padding_factor : float, optional
            Padding value when creating reference grid. Distance between two consecutive crops as a proportion of the
            first listed crop size.
        remove_bg : bool, optional
            Remove background crops if their saturation value is above 5. Default value is True.
        ratio_object_thresh : float, optional
            Objects are removed if they are smaller than ratio*largest object
        min_tiss : float, optional
            Threshold value to consider a crop as tissue. Default value is 0.1.
        """
        self.path = path
        self.crop_size_px = crop_size_px
        self.padding_factor = padding_factor
        self.ratio_object_thresh = ratio_object_thresh
        self.min_tiss = min_tiss
        self.remove_bg = remove_bg
        self.store = np.load(path)
        if self.store.ndim == 3:
            self.store = self.store[np.newaxis, :]
        if self.store.dtype != np.uint8:
            print("converting input dtype to uint8")
            self.store = self.store.astype(np.uint8)
        self.orig_shape = self.store.shape
        self.store = np.pad(
            self.store,
            [
                (0, 0),
                (self.crop_size_px, self.crop_size_px),
                (self.crop_size_px, self.crop_size_px),
                (0, 0),
            ],
            "constant",
            constant_values=255,
        )
        self.msks, self.fg_amount = self._foreground_mask()

        self.grid = self._calc_grid()
        self.idx = self._create_idx()

        # TODO No idea what kind of exceptions could happen.
        # If you are having issues with this dataloader, create an issue.

    def _foreground_mask(self, h_tresh=5):
        # print("computing fg masks")
        ret = []
        fg_amount = []
        for im in self.store:
            msk = (
                cv2.blur(cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[..., 1], (50, 50))
                > h_tresh
            )
            comp, labl, size, cent = cv2.connectedComponentsWithStats(
                msk.astype(np.uint8) * 255
            )
            selec = size[1:, -1] / size[1:, -1].max() > self.ratio_object_thresh
            ids = np.arange(1, comp)[selec]
            fin_msk = np.isin(labl, ids)
            ret.append(fin_msk)
            fg_amount.append(np.mean(fin_msk))

        return ret, fg_amount

    def _calc_grid(self):
        _, h, w, _ = self.store.shape
        n_w = np.floor(
            (w - self.crop_size_px) / (self.crop_size_px * self.padding_factor)
        )
        n_h = np.floor(
            (h - self.crop_size_px) / (self.crop_size_px * self.padding_factor)
        )
        margin_w = (
            int(w - (self.padding_factor * n_w * self.crop_size_px + self.crop_size_px))
            // 2
        )
        margin_h = (
            int(h - (self.padding_factor * n_h * self.crop_size_px + self.crop_size_px))
            // 2
        )
        c_x = (
            np.arange(n_w + 1) * self.crop_size_px * self.padding_factor + margin_w
        ).astype(int)
        c_y = (
            np.arange(n_h + 1) * self.crop_size_px * self.padding_factor + margin_h
        ).astype(int)
        c_x, c_y = np.meshgrid(c_x, c_y)
        return np.array([c_y.flatten(), c_x.flatten()]).T

    def _create_idx(self):
        crd_list = []
        for i, msk in enumerate(self.msks):
            if self.remove_bg:
                valid_crd = [
                    np.mean(
                        msk[
                            crd[0] : crd[0] + self.crop_size_px,
                            crd[1] : crd[1] + self.crop_size_px,
                        ]
                    )
                    > self.min_tiss
                    for crd in self.grid
                ]
                crd_subset = self.grid[valid_crd, :]
                crd_list.append(
                    np.concatenate(
                        [np.repeat(i, crd_subset.shape[0]).reshape(-1, 1), crd_subset],
                        -1,
                    )
                )
            else:
                crd_list.append(
                    np.concatenate(
                        [np.repeat(i, self.grid.shape[0]).reshape(-1, 1), self.grid], -1
                    )
                )
        return np.vstack(crd_list)

    def __len__(self) -> int:
        return self.idx.shape[0]

    def __getitem__(self, idx):
        c, x, y = self.idx[idx]
        out_img = self.store[c, x : x + self.crop_size_px, y : y + self.crop_size_px]
        out_img = normalize_min_max(out_img, 0, 255)
        return out_img, (c, x, y)


class ImageDataset(NpyDataset):
    """
    Torch Dataset to load from NPY files.

    Parameters
    ----------
    path : str
        Path to the Image, needs to be supported by opencv
    crop_size_px : int
        Size of the extracted tiles in pixels. e.g 256 -> 256x256 tiles
    padding_factor : float, optional
        Padding value when creating reference grid. Distance between two consecutive crops as a proportion of the
        first listed crop size.
    remove_bg : bool, optional
        Remove background crops if their saturation value is above 5. Default value is True.
    ratio_object_thresh : float, optional
        Objects are removed if they are smaller than ratio*largest object
    min_tiss : float, optional
        Threshold value to consider a crop as tissue. Default value is 0.1.
    """

    def __init__(
        self,
        path,
        crop_size_px,
        padding_factor=0.5,
        remove_bg=True,
        ratio_object_thresh=5e-1,
        min_tiss=0.1,
    ):
        self.path = path
        self.crop_size_px = crop_size_px
        self.padding_factor = padding_factor
        self.ratio_object_thresh = ratio_object_thresh
        self.min_tiss = min_tiss
        self.remove_bg = remove_bg
        self.store = self._load_image()

        self.orig_shape = self.store.shape
        self.store = np.pad(
            self.store,
            [
                (0, 0),
                (self.crop_size_px, self.crop_size_px),
                (self.crop_size_px, self.crop_size_px),
                (0, 0),
            ],
            "constant",
            constant_values=255,
        )
        self.msks, self.fg_amount = self._foreground_mask()
        self.grid = self._calc_grid()
        self.idx = self._create_idx()

    def _load_image(self):
        img = cv2.imread(self.path)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("Image is neither RGBA nor RGB")
        return img[np.newaxis, ...]
