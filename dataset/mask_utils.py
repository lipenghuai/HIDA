from __future__ import annotations
import numpy as np
import cv2

def clean_binary_mask(mask01: np.ndarray,
                      min_island: float = 0.0005,
                      min_hole: float = 0.0002,
                      kernel: int = 3,
                      iterations: int = 1,
                      keep_largest: bool = False) -> np.ndarray:
    assert mask01.ndim == 2
    m = (mask01 >= 0.5).astype(np.uint8)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K, iterations=iterations)

    H, W = m.shape
    area = H * W
    def as_px(th): return int(th * area) if th < 1.0 else int(th)

    min_island_px = as_px(min_island)
    min_hole_px   = as_px(min_hole)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if num > 1:
        if keep_largest:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_lab = 1 + int(areas.argmax())
            m = (lab == keep_lab).astype(np.uint8)
        else:
            m2 = np.zeros_like(m)
            for L in range(1, num):
                if stats[L, cv2.CC_STAT_AREA] >= min_island_px:
                    m2[lab == L] = 1
            m = m2
    inv = (1 - m).astype(np.uint8)
    num2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num2 > 1:
        inv2 = np.zeros_like(inv)
        for L in range(1, num2):
            if stats2[L, cv2.CC_STAT_AREA] >= min_hole_px:
                inv2[lab2 == L] = 1
        inv = inv2
        m = (1 - inv).astype(np.uint8)

    return m.astype(np.float32)
