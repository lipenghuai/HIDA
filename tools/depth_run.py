import os
import sys
import time
import json
import signal
import traceback
import gc
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

# ==== 路径与配置 ====
SRC_DIR = r"/2024219001/data/D2SA/test/occlusion"
DST_DIR = r"/2024219001/data/D2SA/test/depth"
DIR = "/2024219001/data/temp"
ENCODER = "vitl"
WEIGHT_PATH = r"/2024219001/model/Depth-Anything-V2-main/pth/depth_anything_v2_vitl.pth"

PRIMARY_INPUT_SIZE = 518
FALLBACK_INPUT_SIZES = [448, 384, 320, 256]

GRAYSCALE = True
CMAP_NAME = "Spectral_r"
SKIP_IF_EXISTS = True

SORT_FILENAMES = False
COUNT_FILES = False

PRINT_EVERY = 500
EMPTY_CACHE_EVERY = 2000
GC_COLLECT_EVERY = 5000
ERROR_LOG = os.path.join(DIR, "errors.tsv")
PROGRESS_LOG = os.path.join(DIR, "progress.txt")

SHARD_IDX = int(os.getenv("SHARD_IDX", "0"))
SHARD_CNT = int(os.getenv("SHARD_CNT", "1"))
assert SHARD_CNT >= 1 and 0 <= SHARD_IDX < SHARD_CNT, "Invalid SHARD_IDX/SHARD_CNT"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

cfg = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn) * 255.0
    return x.clip(0, 255).astype(np.uint8)


def is_image_file(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS


def iter_image_files(root: str):
    if SORT_FILENAMES:
        names = [e.name for e in os.scandir(root) if e.is_file() and is_image_file(e.name)]
        names.sort()
        for n in names:
            yield n
    else:
        for e in os.scandir(root):
            if e.is_file() and is_image_file(e.name):
                yield e.name


def atomic_imwrite(dst_path: str, img: np.ndarray) -> bool:
    root, ext = os.path.splitext(dst_path)
    ext = ext.lower()
    if ext not in IMG_EXTS:
        pass

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False

    tmp_path = dst_path + ".part"
    try:
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp_path, dst_path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def log_error(idx: int, total: int, fname: str, err: Exception):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    tb = traceback.format_exc()
    line = f"{ts}\t[{idx}/{total}]\t{fname}\t{repr(err)}\t{tb.strip()}\n"
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def log_progress(done: int, skipped: int, total_seen: int):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    msg = f"{ts} done={done} skipped={skipped} seen={total_seen}\n"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg)
        f.flush()


def safe_infer(model: DepthAnythingV2, img_bgr: np.ndarray, device: str):
    try_sizes = [PRIMARY_INPUT_SIZE] + [s for s in FALLBACK_INPUT_SIZES if s != PRIMARY_INPUT_SIZE]

    for sz in try_sizes:
        try:
            with torch.inference_mode():
                return model.infer_image(img_bgr, sz)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            time.sleep(0.1)
            continue
        except Exception:
            raise

    try:
        model_cpu = model.to("cpu")
        with torch.inference_mode():
            out = model_cpu.infer_image(img_bgr, try_sizes[-1])
        model.to(device)
        return out
    except Exception:
        try:
            model.to(device)
        except Exception:
            pass
        raise


def install_signal_handlers():
    def handle(sig, frame):
        print(f"\n{sig}, Exit safely (processed files will not be duplicated).")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, handle)


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    install_signal_handlers()

    torch.set_grad_enabled(False)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    if COUNT_FILES:
        try:
            total_files = sum(1 for _ in iter_image_files(SRC_DIR))
        except Exception:
            total_files = -1
    else:
        total_files = -1

    print(f"device: {DEVICE}")
    print(f"load {ENCODER} ：{WEIGHT_PATH}")
    model = DepthAnythingV2(**cfg[ENCODER])
    try:
        try:
            state = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)  # PyTorch>=2.0
        except TypeError:
            state = torch.load(WEIGHT_PATH, map_location="cpu")
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"err: {e}")
        return

    model = model.to(DEVICE).eval()
    try:
        dummy = np.zeros((PRIMARY_INPUT_SIZE, PRIMARY_INPUT_SIZE, 3), dtype=np.uint8)
        with torch.inference_mode():
            _ = model.infer_image(dummy, PRIMARY_INPUT_SIZE)
        del dummy
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    done = 0
    skipped = 0
    seen = 0
    t0 = time.time()

    for fname in iter_image_files(SRC_DIR):
        seen += 1

        if SHARD_CNT > 1 and (seen % SHARD_CNT) != (SHARD_IDX % SHARD_CNT):
            continue

        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)

        if SKIP_IF_EXISTS and os.path.exists(dst_path):
            skipped += 1
            if seen % PRINT_EVERY == 0:
                print(f"[{seen}/{total_files if total_files >= 0 else '?'}] jump: {fname}")
            continue

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            log_error(seen, total_files, fname, RuntimeError("imread failed"))
            if seen % PRINT_EVERY == 0:
                print(f"[{seen}/{total_files if total_files >= 0 else '?'}] Read failed, error logged: {fname}")
            continue

        try:
            depth = safe_infer(model, img, DEVICE)  # HxW float32
            d8 = to_u8(depth)

            if GRAYSCALE:
                ok = atomic_imwrite(dst_path, d8)  # 2D uint8
            else:
                try:
                    import matplotlib
                    import matplotlib.cm as cm
                except ImportError as e:
                    raise RuntimeError(
                        "Pseudo-color visualization requires matplotlib to be installed; please set GRAYSCALE=True or install matplotlib.") from e
                cmap = cm.get_cmap(CMAP_NAME)
                rgba = (cmap(d8.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)  # HxWx4
                rgb = rgba[..., :3]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok = atomic_imwrite(dst_path, bgr)

            if not ok:
                raise RuntimeError(f"imwrite failed: {dst_path}")

            done += 1

        except Exception as e:
            log_error(seen, total_files, fname, e)

        # 维护
        if DEVICE == "cuda" and (seen % EMPTY_CACHE_EVERY == 0):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if seen % GC_COLLECT_EVERY == 0:
            gc.collect()

        if seen % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(
                f"[{seen}/{total_files if total_files >= 0 else '?'}]1: {done} 2: {skipped}; speed≈ {rate: .2f} img/s")
            log_progress(done, skipped, seen)

    print(f"process {done}, jump {skipped}, {seen}。")
    print(f"save to: {DST_DIR}")
    log_progress(done, skipped, seen)


if __name__ == "__main__":
    main()
