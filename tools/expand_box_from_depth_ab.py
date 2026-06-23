# -*- coding: utf-8 -*-
import os
import re
import cv2
import time
import argparse
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

GPU_IDS = [0]  # [0], [0,1], [0,1,2,3]

# ============ 原始数据目录 ============
DIR_RGB = r"/2024219001/data/D2SA/traintemp/occlusion"
DIR_VIS = r"/2024219001/data/D2SA/traintemp/visible_object_mask"
DIR_WHOLE = r"/2024219001/data/D2SA/traintemp/whole_mask"

# ============ 新的输出根目录（目录结构与上面三个一致） ============
OUT_ROOT = r"/2024219001/data/D2SA/train"
OUT_RGB_DIR = os.path.join(OUT_ROOT, "occlusion")
OUT_VIS_DIR = os.path.join(OUT_ROOT, "visible_object_mask")
OUT_WHOLE_DIR = os.path.join(OUT_ROOT, "whole_mask")

# ============ Depth Anything V2 ============
import torch
from depth_anything_v2.dpt import DepthAnythingV2  # 确保可 import


@dataclass(frozen=True)
class DA2Config:
    encoder: str = "vitl"  # vits/vitb/vitl/vitg
    weight_path: str = r"/2024219001/model/Depth-Anything-V2-main/pth/depth_anything_v2_vitl.pth"
    input_size: int = 518
    device: Optional[str] = None
    arch: Dict[str, Dict] = field(default_factory=lambda: {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    })


_MODEL_CACHE: Dict[Tuple, DepthAnythingV2] = {}


def resolve_device(gpu_id: Optional[int]) -> str:
    if torch.cuda.is_available() and gpu_id is not None:
        n = torch.cuda.device_count()
        if 0 <= gpu_id < n:
            return f"cuda:{gpu_id}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_da2_model_from_config(cfg: DA2Config) -> DepthAnythingV2:
    device = cfg.device or "cpu"
    key = (cfg.encoder, os.path.abspath(cfg.weight_path), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = DepthAnythingV2(**cfg.arch[cfg.encoder])
    state = torch.load(cfg.weight_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


@torch.inference_mode()
def predict_depth_u8(model: DepthAnythingV2, bgr: np.ndarray, input_size: int) -> np.ndarray:
    depth = model.infer_image(bgr, input_size)  # 返回 float32
    x = depth.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn) * 255.0
    return x.clip(0, 255).astype(np.uint8)


def _clip_xyxy(x0, y0, x1, y1, W, H):
    x0 = int(max(0, min(x0, W - 1)))
    y0 = int(max(0, min(y0, H - 1)))
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    if x1 <= x0:
        if x0 < W - 1:
            x1 = x0 + 1
        else:
            x0 = max(0, W - 2);
            x1 = x0 + 1
    if y1 <= y0:
        if y0 < H - 1:
            y1 = y0 + 1
        else:
            y0 = max(0, H - 2);
            y1 = y0 + 1
    return x0, y0, x1, y1


def _xywh_to_xyxy(x, y, w, h): return x, y, x + w, y + h


def _xyxy_to_xywh(x0, y0, x1, y1): return x0, y0, x1 - x0, y1 - y0


def _median_safe(arr):
    arr = np.asarray(arr).astype(np.float32)
    if arr.size == 0: return np.nan
    return float(np.nanmedian(arr))


def _band_inside(mask, band_px=5):
    band_px = max(1, int(band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (band_px, band_px))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    band = (mask.astype(np.uint8) - eroded).astype(bool)
    return band


def _auto_polarity(d8, mask, edge_band_px=3):
    dil = cv2.dilate(mask.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (edge_band_px, edge_band_px)), 1)
    outer = (dil.astype(bool) & (~mask.astype(bool)))
    inner = _band_inside(mask, band_px=edge_band_px)
    med_in = _median_safe(d8[inner])
    med_out = _median_safe(d8[outer])
    return bool(med_out > med_in)  # True: d8 越大越近


def _direction_scores(d8, mask, box_xywh, near_if_greater, depth_margin=5, edge_band_px=5):
    H, W = d8.shape[:2]
    x, y, w, h = map(int, box_xywh)
    x0, y0, x1, y1 = _xywh_to_xyxy(x, y, w, h)
    x0, y0, x1, y1 = _clip_xyxy(x0, y0, x1, y1, W, H)

    inside_band = _band_inside(mask, band_px=edge_band_px)
    target_med = _median_safe(d8[inside_band])
    if np.isnan(target_med):
        target_med = _median_safe(d8[mask > 0])

    allow_left = int(min(w, x0 - 0))
    allow_right = int(min(w, (W - 1) - x1))
    allow_top = int(min(h, y0 - 0))
    allow_bot = int(min(h, (H - 1) - y1))

    def gather(side):
        if side == 'left' and allow_left > 0:
            xs0, xs1 = int(x0 - allow_left), int(x0)
            ys0, ys1 = int(y0), int(y1)
            if xs1 <= xs0 or ys1 <= ys0: return None, None
            stripe = np.zeros_like(mask, dtype=bool)
            stripe[ys0:ys1, xs0:xs1] = True
            stripe[mask] = False
            xx = np.arange(xs0, xs1, dtype=np.float32)
            w_line = 1.0 - (x0 - xx) / float(max(1, allow_left))
            wgt = np.zeros_like(d8, dtype=np.float32)
            wgt[ys0:ys1, xs0:xs1] = np.tile(w_line[None, :], (ys1 - ys0, 1))
            return stripe, wgt
        if side == 'right' and allow_right > 0:
            xs0, xs1 = int(x1), int(x1 + allow_right)
            ys0, ys1 = int(y0), int(y1)
            if xs1 <= xs0 or ys1 <= ys0: return None, None
            stripe = np.zeros_like(mask, dtype=bool)
            stripe[ys0:ys1, xs0:xs1] = True
            stripe[mask] = False
            xx = np.arange(xs0, xs1, dtype=np.float32)
            w_line = 1.0 - (xx - x1) / float(max(1, allow_right))
            wgt = np.zeros_like(d8, dtype=np.float32)
            wgt[ys0:ys1, xs0:xs1] = np.tile(w_line[None, :], (ys1 - ys0, 1))
            return stripe, wgt
        if side == 'top' and allow_top > 0:
            xs0, xs1 = int(x0), int(x1)
            ys0, ys1 = int(y0 - allow_top), int(y0)
            if xs1 <= xs0 or ys1 <= ys0: return None, None
            stripe = np.zeros_like(mask, dtype=bool)
            stripe[ys0:ys1, xs0:xs1] = True
            stripe[mask] = False
            yy = np.arange(ys0, ys1, dtype=np.float32)
            w_col = 1.0 - (y0 - yy) / float(max(1, allow_top))
            wgt = np.zeros_like(d8, dtype=np.float32)
            wgt[ys0:ys1, xs0:xs1] = np.tile(w_col[:, None], (1, xs1 - xs0))
            return stripe, wgt
        if side == 'bottom' and allow_bot > 0:
            xs0, xs1 = int(x0), int(x1)
            ys0, ys1 = int(y1), int(y1 + allow_bot)
            if xs1 <= xs0 or ys1 <= ys0: return None, None
            stripe = np.zeros_like(mask, dtype=bool)
            stripe[ys0:ys1, xs0:xs1] = True
            stripe[mask] = False
            yy = np.arange(ys0, ys1, dtype=np.float32)
            w_col = 1.0 - (yy - y1) / float(max(1, allow_bot))
            wgt = np.zeros_like(d8, dtype=np.float32)
            wgt[ys0:ys1, xs0:xs1] = np.tile(w_col[:, None], (1, xs1 - xs0))
            return stripe, wgt
        return None, None

    scores = {}
    allows = {'left': allow_left, 'right': allow_right, 'top': allow_top, 'bottom': allow_bot}
    for side in ['left', 'right', 'top', 'bottom']:
        stripe, wgt = gather(side)
        if stripe is None:
            scores[side] = 0.0
            continue
        vals = d8[stripe]
        w = wgt[stripe]
        if vals.size == 0:
            scores[side] = 0.0
            continue
        if near_if_greater:
            occl = (vals >= (target_med + depth_margin)).astype(np.float32)
        else:
            occl = (vals <= (target_med - depth_margin)).astype(np.float32)
        score = float((occl * w).sum() / (w.sum() + 1e-6))
        scores[side] = max(0.0, min(1.0, score))
    return scores, allows


def factors_from_weights(raw_a, raw_b, a, b):
    s = float(raw_a + raw_b)
    p = 0.5 if s <= 1e-8 else float(raw_a / s)
    f_a = b + (a - b) * p
    f_b = a + b - f_a
    f_a = min(max(f_a, b), a)
    f_b = min(max(f_b, b), a)
    return f_a, f_b


def allocate_axis_ab_no_redistribute(L, allow_a, allow_b, weight_a, weight_b, a, b):
    f_a, f_b = factors_from_weights(weight_a + 1e-6, weight_b + 1e-6, a, b)
    want_a = L * f_a
    want_b = L * f_b
    exp_a = min(float(allow_a), float(want_a))
    exp_b = min(float(allow_b), float(want_b))
    return exp_a, exp_b, (f_a, f_b), (want_a, want_b)


# ============ 工具 ============
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


def atomic_imwrite(dst_path: str, img: np.ndarray) -> bool:
    root, ext = os.path.splitext(dst_path)
    ext = ext.lower() if ext else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok: return False
    tmp = dst_path + ".part"
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, dst_path)
        return True
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass


def list_ids() -> List[str]:
    ids = []
    pat = re.compile(r"^(\d+)_occlusion\.png$", re.IGNORECASE)
    with os.scandir(DIR_RGB) as it:
        for e in it:
            if not e.is_file(): continue
            m = pat.match(e.name)
            if m:
                i = m.group(1)
                vis = os.path.join(DIR_VIS, f"{i}_visible_mask.png")
                whole = os.path.join(DIR_WHOLE, f"{i}_whole_mask.png")
                if os.path.exists(vis) and os.path.exists(whole):
                    ids.append(i)
    ids.sort(key=lambda x: int(x))
    return ids


def load_gray_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"err: {path}")
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _clip_xyxy_to_img(xyxy, W, H):
    x0, y0, x1, y1 = xyxy
    return _clip_xyxy(x0, y0, x1, y1, W, H)


def crop_xyxy(img: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _clip_xyxy(x0, y0, x1, y1, W, H)
    return img[y0:y1, x0:x1].copy()


def compute_full_box_from_vis_and_depth(
        d8: np.ndarray, vis_mask: np.ndarray, vis_xywh: Tuple[int, int, int, int],
        a_h: float, b_h: float, a_v: float, b_v: float,
        edge_band: int = 5, depth_margin: float = 5.0, near_if_greater: Optional[bool] = None
) -> Tuple[int, int, int, int]:
    if near_if_greater is None:
        near_if_greater = _auto_polarity(d8, vis_mask, edge_band_px=max(2, edge_band // 2))
    scores, allows = _direction_scores(
        d8, vis_mask, vis_xywh, near_if_greater=near_if_greater,
        depth_margin=depth_margin, edge_band_px=edge_band
    )
    x, y, w, h = map(int, vis_xywh)
    exp_L, exp_R, _, _ = allocate_axis_ab_no_redistribute(
        L=float(w), allow_a=allows['left'], allow_b=allows['right'],
        weight_a=scores['left'], weight_b=scores['right'], a=a_h, b=b_h
    )
    exp_T, exp_B, _, _ = allocate_axis_ab_no_redistribute(
        L=float(h), allow_a=allows['top'], allow_b=allows['bottom'],
        weight_a=scores['top'], weight_b=scores['bottom'], a=a_v, b=b_v
    )
    fx0, fy0 = int(round(x - exp_L)), int(round(y - exp_T))
    fx1, fy1 = int(round(x + w + exp_R)), int(round(y + h + exp_B))
    return fx0, fy0, fx1, fy1


# ============ 核心处理：单样本 ============
def process_one(
        id_str: str, model: DepthAnythingV2, input_size: int, device_str: str,
        a: float, b: float, a_vert: float, b_vert: float,
        edge_band: int, depth_margin: float,
        near_if_greater: Optional[bool],
        skip_existing: bool = True
) -> bool:
    # 输出路径（目录结构与原始一致）
    rgb_out = os.path.join(OUT_RGB_DIR, f"{id_str}_occlusion.png")
    vis_out = os.path.join(OUT_VIS_DIR, f"{id_str}_visible_mask.png")
    whole_out = os.path.join(OUT_WHOLE_DIR, f"{id_str}_whole_mask.png")

    if skip_existing and os.path.exists(rgb_out) and os.path.exists(vis_out) and os.path.exists(whole_out):
        return True

    # 读取输入
    rgb_path = os.path.join(DIR_RGB, f"{id_str}_occlusion.png")
    vis_path = os.path.join(DIR_VIS, f"{id_str}_visible_mask.png")
    whole_path = os.path.join(DIR_WHOLE, f"{id_str}_whole_mask.png")

    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None: return False
    vis = load_gray_mask(vis_path)
    whole = load_gray_mask(whole_path)

    # 尺寸对齐
    H, W = img.shape[:2]
    if vis.shape[:2] != (H, W):
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_NEAREST)
    if whole.shape[:2] != (H, W):
        whole = cv2.resize(whole, (W, H), interpolation=cv2.INTER_NEAREST)

    vx, vy, vw, vh = bbox_from_mask(vis)

    d8 = predict_depth_u8(model, img, input_size)

    a_h, b_h = float(a), float(b)
    a_v = float(a_vert) if a_vert is not None else a_h
    b_v = float(b_vert) if b_vert is not None else b_h
    if a_h < b_h: a_h, b_h = b_h, a_h
    if a_v < b_v: a_v, b_v = b_v, a_v

    fx0, fy0, fx1, fy1 = compute_full_box_from_vis_and_depth(
        d8, (vis > 0).astype(np.uint8), (vx, vy, vw, vh),
        a_h, b_h, a_v, b_v,
        edge_band=edge_band, depth_margin=depth_margin,
        near_if_greater=near_if_greater
    )
    fx0, fy0, fx1, fy1 = _clip_xyxy(fx0, fy0, fx1, fy1, W, H)

    for d in (OUT_RGB_DIR, OUT_VIS_DIR, OUT_WHOLE_DIR):
        os.makedirs(d, exist_ok=True)

    ok1 = atomic_imwrite(rgb_out, img[fy0:fy1, fx0:fx1])
    ok2 = atomic_imwrite(vis_out, vis[fy0:fy1, fx0:fx1])
    ok3 = atomic_imwrite(whole_out, whole[fy0:fy1, fx0:fx1])
    return bool(ok1 and ok2 and ok3)


def worker_main(shard_ids: List[str], gpu_id: Optional[int], args_ns, progress_q: mp.Queue):
    device = resolve_device(gpu_id)
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))

    cfg = DA2Config(
        encoder=args_ns.da2_encoder,
        weight_path=args_ns.da2_weight,
        input_size=args_ns.input_size,
        device=device
    )
    model = load_da2_model_from_config(cfg)

    total = len(shard_ids)
    ok_cnt, fail_cnt = 0, 0
    last_report = time.time()
    for idx, id_str in enumerate(shard_ids, 1):
        ok = False
        try:
            ok = process_one(
                id_str=id_str, model=model, input_size=args_ns.input_size, device_str=device,
                a=args_ns.a, b=args_ns.b, a_vert=args_ns.a_vert, b_vert=args_ns.b_vert,
                edge_band=args_ns.edge_band, depth_margin=args_ns.depth_margin,
                near_if_greater=args_ns.near_if_greater,
                skip_existing=not args_ns.overwrite
            )
        except Exception:
            ok = False
        ok_cnt += int(ok)
        fail_cnt += int(not ok)

        now = time.time()
        if (now - last_report) > 0.2 or idx == total:
            progress_q.put(("progress", gpu_id, idx, total, ok_cnt, fail_cnt))
            last_report = now

    progress_q.put(("done", gpu_id, total, ok_cnt, fail_cnt))


# ============ 主进程进度聚合 ============
def run_with_progress(all_ids: List[str], args_ns):
    n_gpus = len(GPU_IDS)
    if n_gpus <= 0:
        raise RuntimeError("If GPU_IDS is empty: Please set the list of GPUs you want to use at the top of the script.")

    shards = [[] for _ in range(n_gpus)]
    for i, sid in enumerate(all_ids):
        shards[i % n_gpus].append(sid)

    total = len(all_ids)
    per_gpu_idx = {gid: 0 for gid in GPU_IDS}
    per_gpu_ok = {gid: 0 for gid in GPU_IDS}
    per_gpu_fail = {gid: 0 for gid in GPU_IDS}

    mp.set_start_method("spawn", force=True)
    q = mp.Queue(maxsize=1024)
    procs = []
    for k, gid in enumerate(GPU_IDS):
        if len(shards[k]) == 0:
            continue
        p = mp.Process(target=worker_main, args=(shards[k], gid, args_ns, q), daemon=True)
        p.start()
        procs.append(p)

    last_print = time.time()
    done_workers = 0
    while done_workers < len(procs):
        try:
            msg = q.get(timeout=0.5)
        except Exception:
            msg = None
        if msg:
            tag = msg[0]
            if tag == "progress":
                _, gid, idx, shard_total, ok_cnt, fail_cnt = msg
                per_gpu_idx[gid] = idx
                per_gpu_ok[gid] = ok_cnt
                per_gpu_fail[gid] = fail_cnt
            elif tag == "done":
                _, gid, shard_total, ok_cnt, fail_cnt = msg
                per_gpu_idx[gid] = shard_total
                per_gpu_ok[gid] = ok_cnt
                per_gpu_fail[gid] = fail_cnt
                done_workers += 1

        now = time.time()
        if (now - last_print) > 0.5 or done_workers == len(procs):
            done = sum(per_gpu_idx.values())
            ok_total = sum(per_gpu_ok.values())
            fail_total = sum(per_gpu_fail.values())
            pct = 100.0 * done / max(1, total)
            print(f"[Progress] {done}/{total} ({pct:.2f}%)  ok={ok_total}  fail={fail_total}", end="\r")
            last_print = now

    for p in procs:
        p.join(timeout=0.2)
    print()
    print(f"[Summary] total={total}, ok={sum(per_gpu_ok.values())}, fail={sum(per_gpu_fail.values())}")


def parse_near_if_greater(val: str) -> Optional[bool]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("auto", "none", "null", ""):
        return None
    if s in ("true", "t", "1", "yes", "y", "near_greater", "greater"):
        return True
    if s in ("false", "f", "0", "no", "n", "near_smaller", "smaller", "less"):
        return False
    raise ValueError(f"--near_if_greater 不支持的值: {val}. 允许: auto/true/false")


def main():
    ap = argparse.ArgumentParser()
    # DA2
    ap.add_argument('--da2_weight', type=str,
                    default=r"/2024219001/model/Depth-Anything-V2-main/pth/depth_anything_v2_vitl.pth")
    ap.add_argument('--da2_encoder', type=str, default="vitl", choices=['vits', 'vitb', 'vitl', 'vitg'])
    ap.add_argument('--input_size', type=int, default=518)

    ap.add_argument('--a', type=float, default=0.6)
    ap.add_argument('--b', type=float, default=0.2, help='Minimum expansion factor on one side (lateral)')
    ap.add_argument('--a_vert', type=float, default=None)
    ap.add_argument('--b_vert', type=float, default=None)
    ap.add_argument('--edge_band', type=int, default=5)
    ap.add_argument('--depth_margin', type=float, default=5.0)

    ap.add_argument('--near_if_greater', type=str, default='true')

    ap.add_argument('--single_id', type=str, default=None)
    ap.add_argument('--overwrite', action='store_true')

    args = ap.parse_args()
    args.near_if_greater = parse_near_if_greater(args.near_if_greater)

    if args.single_id is not None:
        device = resolve_device(GPU_IDS[0] if len(GPU_IDS) > 0 else None)
        cfg = DA2Config(encoder=args.da2_encoder, weight_path=args.da2_weight, input_size=args.input_size,
                        device=device)
        model = load_da2_model_from_config(cfg)

        for d in (OUT_RGB_DIR, OUT_VIS_DIR, OUT_WHOLE_DIR):
            os.makedirs(d, exist_ok=True)

        ok = process_one(
            id_str=args.single_id, model=model, input_size=args.input_size, device_str=device,
            a=args.a, b=args.b, a_vert=args.a_vert, b_vert=args.b_vert,
            edge_band=args.edge_band, depth_margin=args.depth_margin,
            near_if_greater=args.near_if_greater,
            skip_existing=not args.overwrite
        )
        print("[Single] DONE.", "OK" if ok else "FAILED")
        return

    ids = list_ids()
    if len(ids) == 0:
        print("No sample with {ID}_occlusion.png (and all three paths) was found under testpic/.")
        return

    for d in (OUT_RGB_DIR, OUT_VIS_DIR, OUT_WHOLE_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"Processing {len(ids)}; GPU: {GPU_IDS}；near_if_greater={args.near_if_greater}")
    run_with_progress(ids, args)


if __name__ == "__main__":
    main()
