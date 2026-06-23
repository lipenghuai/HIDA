# -*- coding: utf-8 -*-

import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import gc  # 主动回收内存

DIR_BASE = Path(r"/2024219001/data/D2SA/test")
DIR_ORI = DIR_BASE / "occlusion"
DIR_VIS = DIR_BASE / "visible_object_mask"
# output
DIR_OUT = DIR_BASE / "sam1_visiable"
DIR_EMB = DIR_BASE / "sam1_imageencoder"

SAM_CHECKPOINT = Path(
    r"./pth/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"

MARGIN_PCT = 0.00

# ====== 文件名规则 ======
PAT_VISIBLE = re.compile(r"^(\d+)_visible_mask\.png$", re.IGNORECASE)

# ====== 导入 SAM ======
try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise ImportError("未找到 segment-anything 库，请先在其仓库目录下执行 `pip install -e .` 或正常安装。") from e


def mask_to_bbox(mask: np.ndarray):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None

    ys, xs = np.where(fg > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return [x0, y0, x1, y1]


def load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"err：{path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def save_mask_png(mask_bool: np.ndarray, out_path: Path):
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), mask_u8)
    if not ok:
        raise IOError(f"err：{out_path}")


def main():
    if not DIR_ORI.exists():
        raise FileNotFoundError(f"not exist: {DIR_ORI}")
    if not DIR_VIS.exists():
        raise FileNotFoundError(f"not exist: {DIR_VIS}")
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    DIR_EMB.mkdir(parents=True, exist_ok=True)
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"not exist: {SAM_CHECKPOINT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device：{device}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    vis_iter = (p for p in DIR_VIS.iterdir() if p.is_file() and PAT_VISIBLE.match(p.name))

    total, ok_cnt, miss_cnt, empty_cnt, err_cnt = 0, 0, 0, 0, 0

    for vis_path in vis_iter:
        total += 1
        m = PAT_VISIBLE.match(vis_path.name)
        _id = m.group(1) if m else None
        if _id is None:
            continue

        ori_path = DIR_ORI / f"{_id}_occlusion.png"
        out_path = DIR_OUT / f"{_id}_sam_mask.png"
        emb_path = DIR_EMB / f"{_id}_occlusion.pt"

        try:
            if not ori_path.exists():
                miss_cnt += 1
                if (total % 1000) == 0:
                    print(f"[WARN]  {total} Lack -> {ori_path}")
                continue

            vis_mask = cv2.imread(str(vis_path), cv2.IMREAD_UNCHANGED)
            if vis_mask is None:
                miss_cnt += 1
                if (total % 1000) == 0:
                    print(f"[WARN] {total} err -> {vis_path}")
                continue

            bbox = mask_to_bbox(vis_mask)
            if bbox is None:
                empty_cnt += 1
                continue

            image_rgb = load_image_rgb(ori_path)
            predictor.set_image(image_rgb)

            with torch.no_grad():
                image_encoder = predictor.get_image_embedding()  # (1, C, H', W')
                image_encoder = image_encoder.detach().cpu().contiguous()
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "feats": image_encoder,
                "shape": tuple(image_encoder.shape),
                "source_image": str(ori_path),
                "preproc": {
                    "resize": 1024,
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375]
                }
            }, emb_path)
            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0
            margin = int(round(MARGIN_PCT * min(w, h)))
            img_h, img_w = image_rgb.shape[:2]
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(img_w - 1, x1 + margin)
            y1 = min(img_h - 1, y1 + margin)

            xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)

            boxes_t = predictor.transform.apply_boxes_torch(
                torch.tensor(xyxy[None, :], device=device),
                image_rgb.shape[:2]  # (H, W)
            )

            with torch.no_grad():
                masks_t, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_t,
                    multimask_output=False
                )

            sam_mask = masks_t[0, 0].detach().cpu().numpy().astype(bool)

            save_mask_png(sam_mask, out_path)
            ok_cnt += 1

        except RuntimeError as re_err:
            err_cnt += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            err_cnt += 1
            traceback.print_exc(file=sys.stdout)
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (total % 1000) == 0:
            print(
                f"[INFO] Processed {total} images, successful {ok_cnt}, original image missing {miss_cnt}, null mask {empty_cnt}, error {err_cnt}")
    print("\n======== All processing completed ========")
    print(f"Total: {total}")
    print(f"Success: {ok_cnt}")
    print(f"Original image missing or failed to read: {miss_cnt}")
    print(f"Mask empty: {empty_cnt}")
    print(f"Exception: {err_cnt}")
    print(f"Mask output directory: {DIR_OUT}")
    print(f"Embedding output directory: {DIR_EMB}")


if __name__ == "__main__":
    main()
