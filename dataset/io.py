from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import cv2
import torch


def ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC, uint8


def imread_gray_anydepth(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # HW, dtype: u8/u16/f32


def normalize_rgb01(img_u8_rgb: np.ndarray) -> np.ndarray:
    if img_u8_rgb.dtype != np.uint8:
        img_u8_rgb = img_u8_rgb.astype(np.uint8)
    return img_u8_rgb.astype(np.float32) / 255.0


def normalize_depth01(depth_any: np.ndarray) -> np.ndarray:
    d = depth_any.astype(np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return (d - dmin) / (dmax - dmin)


def binarize_mask(mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if mask.dtype != np.float32 and mask.dtype != np.float64:
        mask = mask.astype(np.float32)
    mx = float(mask.max()) if mask.size > 0 else 1.0
    denom = max(mx, 1.0)
    return ((mask / denom) >= thr).astype(np.float32)


# def resize(img: np.ndarray, size_hw: Optional[Tuple[int, int]], is_mask: bool) -> np.ndarray:
#     """(H,W) 目标；mask 用 NEAREST；图像/深度自适应插值；深度下采样忽略 0。"""
#     if size_hw is None:
#         return img
#     H0, W0 = img.shape[:2]
#     H, W = size_hw
#     if (H, W) == (H0, W0):
#         return img
#     if is_mask:
#         return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
#
#     down_h, down_w = (H < H0), (W < W0)
#     up_h, up_w = (H > H0), (W > W0)
#
#     is_single = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
#     is_depth_like = is_single and (img.dtype in (np.uint16, np.uint32, np.int32, np.float32, np.float64))
#     if is_depth_like and down_h and down_w and (img == 0).any():
#         m = (img != 0).astype(np.float32)
#         d = img.astype(np.float32)
#         num = cv2.resize(d * m, (W, H), interpolation=cv2.INTER_AREA)
#         den = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
#         out = np.where(den > 1e-6, num / den, 0)
#         if np.issubdtype(img.dtype, np.integer):
#             out = np.round(out).astype(img.dtype)
#         else:
#             out = out.astype(img.dtype)
#         return out
#
#     if down_h and down_w:
#         interp = cv2.INTER_AREA
#     elif up_h and up_w:
#         s = max(H / H0, W / W0)
#         interp = cv2.INTER_CUBIC if s > 1.5 else cv2.INTER_LINEAR
#     else:
#         interp = cv2.INTER_LINEAR
#     return cv2.resize(img, (W, H), interpolation=interp)

def to_tensor_chw(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 2:
        x = x[:, :, None]
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return torch.from_numpy(x).float()


def resize(img: np.ndarray, size_hw: Optional[Tuple[int, int]], is_mask: bool) -> np.ndarray:
    if size_hw is None:
        return img
    H0, W0 = img.shape[:2]
    H, W = size_hw
    if (H, W) == (H0, W0):
        return img

    if is_mask:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

    is_single = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
    is_depth_like = is_single and (img.dtype in (np.uint16, np.uint32, np.int32, np.float32, np.float64))

    def _pick_interp(scale: float):
        if scale < 1.0:
            return cv2.INTER_AREA
        elif scale > 1.5:
            return cv2.INTER_CUBIC
        elif scale > 1.0:
            return cv2.INTER_LINEAR
        else:
            return None

    def _resize_mask_aware(src: np.ndarray, h: int, w: int, interp: int) -> np.ndarray:
        src32 = src.astype(np.float32)
        m = np.isfinite(src32) & (src32 != 0)
        m = m.astype(np.float32)
        num = cv2.resize(src32 * m, (w, h), interpolation=interp)
        den = cv2.resize(m, (w, h), interpolation=interp)
        # out = np.where(den > 1e-6, num / den, 0.0)
        out = np.zeros_like(num, dtype=np.float32)
        mask = np.isfinite(den) & (den > 1e-6)
        np.divide(num, den, out=out, where=mask)
        if np.issubdtype(src.dtype, np.integer):
            out = np.round(out).astype(src.dtype)
        else:
            out = out.astype(src.dtype)
        return out

    scale_w = W / W0
    scale_h = H / H0
    iw = _pick_interp(scale_w) or cv2.INTER_LINEAR
    ih = _pick_interp(scale_h) or cv2.INTER_LINEAR

    if is_depth_like:
        tmp = img
        if W != W0:
            tmp = _resize_mask_aware(tmp, tmp.shape[0], W, iw)
        if H != H0:
            tmp = _resize_mask_aware(tmp, H, W, ih)
        return tmp

    anisotropic = (W != W0) and (H != H0) and ((scale_w < 1.0) ^ (scale_h < 1.0))

    if anisotropic:
        tmp = cv2.resize(img, (W, H0), interpolation=iw)
        out = cv2.resize(tmp, (W, H), interpolation=ih)
        return out

    if (scale_w < 1.0) and (scale_h < 1.0):
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    elif (scale_w > 1.0) and (scale_h > 1.0):
        s = max(scale_w, scale_h)
        interp = cv2.INTER_CUBIC if s > 1.5 else cv2.INTER_LINEAR
        return cv2.resize(img, (W, H), interpolation=interp)
    else:
        tmp = img
        if W != W0:
            tmp = cv2.resize(tmp, (W, tmp.shape[0]), interpolation=iw)
        if H != H0:
            tmp = cv2.resize(tmp, (W, H), interpolation=ih)
        return tmp
