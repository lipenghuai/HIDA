# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import build_dataset, build_loader
from models import build_model


def tensor_to_mask_u8(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    prob = torch.sigmoid(logits).detach().cpu().float()[0]  # (H,W)
    m = (prob >= thr).to(torch.uint8).numpy() * 255
    return m


@torch.no_grad()
def miou_mdice_batch(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    pred = (torch.sigmoid(logits) >= thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter / (union + 1e-6))

    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = (num / den)
    return iou.mean().item(), dice.mean().item()


def build_models_for_test(args, ckpt_args, device):
    feats_mode = args.feats or ckpt_args.get("feats", "concat")
    enc_size = args.enc_size or ckpt_args.get("enc_size", "base")
    dec_size = args.dec_size or ckpt_args.get("dec_size", "base")
    input_size = (args.size if args.size > 0 else ckpt_args.get("size", 256))
    input_size = (input_size, input_size)

    need_encoder = (feats_mode != "sam")
    enc = None
    if need_encoder:
        enc = build_model(
            "oatr_encoder",
            input_size=input_size,
            model_size=enc_size,
            out_channels=256,
            num_experts=2, router_hidden=16,
            oatr_alpha=4.0, oatr_beta=1.0, lb_weight=1e-3,
            extract_layers=(2, 4, 6),
        ).to(device)
        enc.eval()

    dec_in_ch = 512 if feats_mode == "concat" else 256
    dec = build_model("attn_decoder", in_ch=dec_in_ch, dec_model_size=dec_size).to(device)
    dec.eval()

    return enc, dec, feats_mode, input_size


def _extract_keys(batch):
    if "keys" in batch: return list(batch["keys"])
    if "key" in batch:  return list(batch["key"])
    b = batch["x5"].shape[0]
    return [f"{i:06d}" for i in range(b)]


def _get_ref_img_path(ds, batch_paths_item: dict | None, key: str) -> str:
    # 优先从 collate 出来的 paths 里拿
    if isinstance(batch_paths_item, dict) and "img" in batch_paths_item:
        return batch_paths_item["img"]
    # 回退：用数据集路径生成器
    if hasattr(ds, "paths"):
        return ds.paths.path_img(key)
    raise RuntimeError("Unable to determine the original image path used for size restoration.")


def main():
    ap = argparse.ArgumentParser()
    # 数据与模型
    ap.add_argument("--root", type=str,
                    default=r"/2024219001/data/kins/test",
                    help="数据根目录")
    ap.add_argument("--ckpt", type=str, default=r"/2024219001/lip/amodelddp/runs/exp_ddp0_ll_e2_r16/epoch_003.pt")
    ap.add_argument("--size", type=int, default=-1)
    ap.add_argument("--feats", type=str, default="concat", choices=["", "vit", "sam", "concat"],
                    help="留空沿用 ckpt；否则覆盖为 vit/sam/concat")
    ap.add_argument("--enc_size", type=str, default="large")
    ap.add_argument("--dec_size", type=str, default="large", help="Decoder size, leave blank and use ckpt.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--out_dir", type=str, default=r"/2024219001/data/kins/pred_masks")
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--save_prob", action="store_true")

    ap.add_argument("--save_mask", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    test_size = (args.size if args.size > 0 else ckpt_args.get("size", 256))
    ds = build_dataset(
        "pix2gestalt_occlu",
        root_dir=args.root,
        input_size=(test_size, test_size),
        require_whole_mask=True,
        strict_check=True,
        show_progress=True,
    )
    dl = build_loader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 构建模型并加载权重
    enc, dec, feats_mode, _ = build_models_for_test(args, ckpt_args, device)
    if "decoder" in ckpt:
        dec.load_state_dict(ckpt["decoder"], strict=True)
    else:
        raise RuntimeError("The 'decoder' weights were not found in ckpt. Please train the output ckpt using train.py.")
    if enc is not None and "encoder" in ckpt:
        enc.load_state_dict(ckpt["encoder"], strict=False)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    n_img = 0
    iou_sum, dice_sum = 0.0, 0.0

    pbar = tqdm(dl, desc="Testing", ncols=100)

    for batch in pbar:
        keys = _extract_keys(batch)
        x5 = batch["x5"].to(device)  # (B,5,h,w)
        y_gt = batch["y_whole"]  # (B,1,h,w) 或 None
        sam_feats = batch["sam_feats"].to(device).squeeze(1)  # (B,256,64,64)

        # 准备解码器输入特征
        if feats_mode == "sam":
            feats = sam_feats
        elif feats_mode == "vit":
            if enc is None:
                raise RuntimeError("feats=vit need encoder。")
            feats = enc(x5)
        else:  # concat
            if enc is None:
                raise RuntimeError("feats=concat need encoder。")
            feats = torch.cat([enc(x5), sam_feats], dim=1)  # (B,512,64,64)

        logits, iou_scores = dec(feats)  # (B,1,256,256), (B,1)

        for i, key in enumerate(keys):
            batch_paths_item = batch.get("paths", [None] * len(keys))[i] if "paths" in batch else None
            ref_img = _get_ref_img_path(ds, batch_paths_item, key)
            with Image.open(ref_img) as im_ref:
                W0, H0 = im_ref.size

            mask_u8_small = tensor_to_mask_u8(logits[i], thr=args.th)  # (Hs,Ws)

            mask_big = Image.fromarray(mask_u8_small).resize((W0, H0), resample=Image.NEAREST)

            if args.save_mask:
                mask_big.save(Path(args.out_dir, f"{key}_pred.png"))
            if args.save_prob:
                prob_small = torch.sigmoid(logits[i]).detach().cpu()[None, None]  # (1,1,Hs,Ws)
                prob_big = F.interpolate(
                    prob_small, size=(H0, W0), mode="bilinear", align_corners=False
                )[0, 0]
                np.save(
                    Path(args.out_dir, f"{key}_prob.npy"),
                    prob_big.numpy().astype(np.float32),
                )

        if y_gt is not None:
            y = y_gt.to(device)
            miou, mdice = miou_mdice_batch(logits, y, thr=args.th)
            iou_sum += miou * x5.size(0)
            dice_sum += mdice * x5.size(0)

            den = n_img + x5.size(0)
            cur_miou = iou_sum / max(den, 1)
            cur_mdice = dice_sum / max(den, 1)

            pbar.set_postfix(mIoU=f"{cur_miou:.4f}", mDice=f"{cur_mdice:.4f}")

        n_img += x5.size(0)

    if n_img > 0 and ds[0]["y_whole"] is not None:
        print(f"[Eval] mIoU={iou_sum / n_img:.6f}  mDice={dice_sum / n_img:.6f}")
    if args.save_mask:
        print(f"已保存到: {args.out_dir}")


if __name__ == "__main__":
    main()
