# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..io import (
    ensure_exists, imread_rgb, imread_gray_anydepth,
    normalize_rgb01, normalize_depth01, binarize_mask,
    resize, to_tensor_chw
)
from ..mask_utils import clean_binary_mask

_NUMERIC_ID_RE = re.compile(r"^(\d+)_occlusion\.png$")


@dataclass
class Pix2GestaltOccluPaths:
    root: str
    dir_testpic: str = "occlusion"
    dir_vis_mask: str = "sam1_visiable"
    dir_depth: str = "depth"
    dir_sam_feats: str = "sam1_imageencoder"
    dir_whole_mask: str = "whole_mask"

    def path_img(self, key: str) -> str:
        return os.path.join(self.root, self.dir_testpic, f"{key}_occlusion.png")

    def path_vis_mask(self, key: str) -> str:
        return os.path.join(self.root, self.dir_vis_mask, f"{key}_sam_mask.png")

    def path_depth(self, key: str) -> str:
        return os.path.join(self.root, self.dir_depth, f"{key}_occlusion.png")

    def path_sam_feats(self, key: str) -> str:
        return os.path.join(self.root, self.dir_sam_feats, f"{key}_occlusion.pt")

    def path_whole_mask(self, key: str) -> str:
        return os.path.join(self.root, self.dir_whole_mask, f"{key}_whole_mask.png")


class Pix2GestaltOccluDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 input_size: Optional[Tuple[int, int]] = (256, 256),
                 require_whole_mask: bool = True,
                 strict_check: bool = True,
                 keys_subset: Optional[List[str]] = None,
                 dir_testpic: Optional[str] = None,
                 dir_vis_mask: Optional[str] = None,
                 dir_depth: Optional[str] = None,
                 dir_sam_feats: Optional[str] = None,
                 dir_whole_mask: Optional[str] = None,
                 clean_masks: bool = True,
                 clean_vis_mask: bool = False,
                 min_island: float = 0.0005,
                 min_hole: float = 0.0002,
                 morph_kernel: int = 3,
                 morph_iter: int = 1,
                 keep_largest: bool = False,
                 show_progress: bool = True,
                 scan_workers: int = 8):
        self.paths = Pix2GestaltOccluPaths(root=root_dir)
        if dir_testpic:   self.paths.dir_testpic = dir_testpic
        if dir_vis_mask:  self.paths.dir_vis_mask = dir_vis_mask
        if dir_depth:     self.paths.dir_depth = dir_depth
        if dir_sam_feats: self.paths.dir_sam_feats = dir_sam_feats
        if dir_whole_mask: self.paths.dir_whole_mask = dir_whole_mask

        self.input_size = input_size
        self.require_whole_mask = require_whole_mask
        self.strict_check = strict_check

        self.clean_masks = clean_masks
        self.clean_vis_mask = clean_vis_mask
        self.min_island = min_island
        self.min_hole = min_hole
        self.morph_kernel = morph_kernel
        self.morph_iter = morph_iter
        self.keep_largest = keep_largest

        rank0 = True
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank0 = (dist.get_rank() == 0)
        except Exception:
            pass
        self.show_progress = bool(show_progress and rank0 and (tqdm is not None))
        self.scan_workers = max(1, int(scan_workers))

        self.samples = self._scan_samples(keys_subset)
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Please check your folder structure and file names.")

    def _scan_samples(self, keys_subset: Optional[List[str]]) -> List[str]:
        dir_img = os.path.join(self.paths.root, self.paths.dir_testpic)
        ensure_exists(dir_img, "testpic folder")

        keys = []
        for fn in os.listdir(dir_img):
            m = _NUMERIC_ID_RE.match(fn)
            if m:
                keys.append(m.group(1))
        keys = sorted(set(keys), key=lambda x: int(x))

        if keys_subset is not None:
            filt = set(keys_subset)
            keys = [k for k in keys if k in filt]

        def check_one(k: str):
            need = [
                ("img", self.paths.path_img(k)),
                ("vis_mask", self.paths.path_vis_mask(k)),
                ("depth", self.paths.path_depth(k)),
                ("sam_feats", self.paths.path_sam_feats(k)),
            ]
            if self.require_whole_mask:
                need.append(("whole_mask", self.paths.path_whole_mask(k)))
            missing = [name for name, p in need if not os.path.exists(p)]
            return k, (len(missing) == 0), missing

        valid, missing_all = [], []
        use_tqdm = self.show_progress and (tqdm is not None)
        pbar = tqdm(total=len(keys), desc="Scanned Samples", unit="file", disable=not use_tqdm) if use_tqdm else None

        if self.scan_workers > 1 and len(keys) > 64:
            with ThreadPoolExecutor(max_workers=self.scan_workers) as ex:
                futures = {ex.submit(check_one, k): k for k in keys}
                for fut in as_completed(futures):
                    k, ok, miss = fut.result()
                    (valid if ok else missing_all).append((k, miss))
                    if pbar: pbar.update(1)
        else:
            for k in keys:
                k2, ok, miss = check_one(k)
                (valid if ok else missing_all).append((k2, miss))
                if pbar: pbar.update(1)

        if pbar: pbar.close()

        valid_keys = [k for k, _ in valid]
        if len(valid_keys) == 0 and len(missing_all) > 0 and self.strict_check:
            examples = [f"key={k} lack: {','.join(miss)}" for k, miss in missing_all[:20]]
            more = "" if len(missing_all) <= 20 else f"{len(missing_all) - 20}"
            raise FileNotFoundError("\n".join(examples) + ("\n" + more if more else ""))

        if self.strict_check and len(missing_all) > 0 and len(valid_keys) > 0:
            print(missing_all)
            print(f"[WARN] {missing_all[0][0]} -> {missing_all[0][1]} ( {len(missing_all)}")

        return sorted(valid_keys, key=lambda x: int(x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.samples[idx]
        p_img = self.paths.path_img(key)
        p_vis = self.paths.path_vis_mask(key)
        p_dep = self.paths.path_depth(key)
        p_pt = self.paths.path_sam_feats(key)
        p_wh = self.paths.path_whole_mask(key)

        # 读取
        img_rgb = imread_rgb(p_img)  # HWC u8
        vis_mask = imread_gray_anydepth(p_vis)  # HW
        depth = imread_gray_anydepth(p_dep)  # HW
        sam_dict = torch.load(p_pt, map_location="cpu", weights_only=True)
        if not isinstance(sam_dict, dict) or "feats" not in sam_dict:
            raise RuntimeError(f"{p_pt} must be a dict with key 'feats'")
        sam_feats = sam_dict["feats"].squeeze(0)
        if not torch.is_tensor(sam_feats):
            sam_feats = torch.tensor(sam_feats)
        sam_feats = sam_feats.float()  # (1,256,64,64)

        whole_mask = None
        if os.path.exists(p_wh):
            whole_mask = imread_gray_anydepth(p_wh)

        img_rgb = resize(img_rgb, self.input_size, is_mask=False)
        depth = resize(depth, self.input_size, is_mask=False)
        vis_mask = resize(vis_mask, self.input_size, is_mask=True)
        if whole_mask is not None:
            whole_mask = resize(whole_mask, self.input_size, is_mask=True)

        img_rgb = normalize_rgb01(img_rgb)  # [0,1]
        vis_mask = binarize_mask(vis_mask, 0.5)  # 0/1
        depth = normalize_depth01(depth)  # [0,1]
        if whole_mask is not None:
            whole_mask = binarize_mask(whole_mask, 0.5)

        if self.clean_masks and whole_mask is not None:
            whole_mask = clean_binary_mask(
                whole_mask,
                min_island=self.min_island, min_hole=self.min_hole,
                kernel=self.morph_kernel, iterations=self.morph_iter,
                keep_largest=self.keep_largest
            )
        if self.clean_vis_mask:
            vis_mask = clean_binary_mask(
                vis_mask,
                min_island=self.min_island * 0.5, min_hole=self.min_hole,
                kernel=self.morph_kernel, iterations=self.morph_iter,
                keep_largest=False
            )

        five_ch = np.concatenate([img_rgb, vis_mask[:, :, None], depth[:, :, None]], axis=2)  # H,W,5
        x5 = to_tensor_chw(five_ch)  # 5,H,W
        y_whole = to_tensor_chw(whole_mask) if whole_mask is not None else None  # 1,H,W

        return {
            "key": key,
            "x5": x5,
            "sam_feats": sam_feats,  # (1,256,64,64)
            "y_whole": y_whole,  # (1,H,W) or None
            "paths": {
                "img": p_img, "vis_mask": p_vis, "depth": p_dep, "sam_feats": p_pt, "whole_mask": p_wh
            }
        }
