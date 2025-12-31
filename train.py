# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, argparse, random, socket, platform
from typing import Tuple
from contextlib import closing

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

from dataset import build_dataset
from models import build_model
from engine import train_one_epoch, validate


# ---------------- util ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@torch.no_grad()
def all_reduce_mean_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor([x], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def build_datasets_and_loaders(*, root: str, size: int, val_ratio: float,
                               batch_size: int, num_workers: int,
                               distributed: bool, seed: int):
    ds = build_dataset(
        "pix2gestalt_occlu",
        root_dir=root,
        input_size=(size, size),
        require_whole_mask=True,
        strict_check=True,
        show_progress=is_main_process(),
    )
    n_all = len(ds)
    n_val = max(1, int(n_all * val_ratio))
    n_train = n_all - n_val

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_all, generator=g).tolist()
    idx_train, idx_val = perm[:n_train], perm[n_train:]
    ds_train, ds_val = Subset(ds, idx_train), Subset(ds, idx_val)

    if distributed:
        train_sampler = DistributedSampler(ds_train, shuffle=True)
        val_sampler = DistributedSampler(ds_val, shuffle=False)
        train_shuffle = False
        val_shuffle = False
    else:
        train_sampler = RandomSampler(ds_train)
        val_sampler = SequentialSampler(ds_val)
        train_shuffle = False
        val_shuffle = False

    common_kwargs = dict(
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        drop_last=True,
        **common_kwargs,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, batch_size // 2),
        sampler=val_sampler,
        shuffle=val_shuffle,
        drop_last=False,
        **common_kwargs,
    )
    return dl_train, dl_val, train_sampler, val_sampler


def save_ckpt(save_dir: str, epoch: int, best_metric: float,
              decoder: torch.nn.Module, encoder: torch.nn.Module | None,
              args: dict, name: str, ddp_wrapped: bool):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    dec = decoder.module if ddp_wrapped else decoder
    enc = encoder.module if (ddp_wrapped and encoder is not None) else encoder
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "decoder": dec.state_dict(),
        "args": args,
    }
    if enc is not None:
        payload["encoder"] = enc.state_dict()
    torch.save(payload, path)
    if is_main_process():
        print(f"✔: {path}")


def build_models_and_optim(args, device):
    need_encoder = (args.feats != "sam")
    enc = None
    if need_encoder:
        enc = build_model(
            "oatr_encoder",
            input_size=(args.size, args.size),
            model_size=args.enc_size,
            out_channels=256,
            num_experts=8,
            router_hidden=64,
            oatr_alpha=4.0,
            oatr_beta=1.0,
            lb_weight=1e-3,
        ).to(device)
        if args.freeze_encoder:
            for p in enc.parameters():
                p.requires_grad = False
            if is_main_process():
                print("⚙️ encoder，train decoder。")

    dec_in_ch = 512 if args.feats == "concat" else 256
    dec = build_model("attn_decoder", in_ch=dec_in_ch, dec_model_size=args.dec_size).to(device)

    params = [p for p in dec.parameters() if p.requires_grad]
    if need_encoder:
        params += [p for p in enc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return enc, dec, optimizer, scheduler


def resume_if_needed(args, enc, dec, need_encoder: bool):
    start_epoch = 0
    best_dice = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        dec.load_state_dict(ckpt["decoder"])
        if need_encoder and "encoder" in ckpt and enc is not None:
            enc.load_state_dict(ckpt["encoder"])
        start_epoch = ckpt.get("epoch", 0)
        best_dice = ckpt.get("best_metric", -1.0)
        if is_main_process():
            print(f"🔄 已从 {args.resume} 恢复: epoch={start_epoch}, best_dice={best_dice:.4f}")
    return start_epoch, best_dice


# --------------- main loop ---------------
def train_loop(rank: int, world_size: int, args):
    # init dist
    if args.distributed:
        backend = "gloo" if platform.system().lower().startswith("win") else "nccl"
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{args.port}",
            world_size=world_size,
            rank=rank,
        )
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if is_main_process():
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    dl_train, dl_val, train_sampler, _ = build_datasets_and_loaders(
        root=args.root,
        size=args.size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        seed=args.seed,
    )

    enc, dec, optimizer, scheduler = build_models_and_optim(args, device)
    need_encoder = (args.feats != "sam")
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # DDP wrap —— 关键：find_unused_parameters=True
    ddp_wrapped = False
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if need_encoder and enc is not None:
            enc = DDP(
                enc.to(device),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                static_graph=False,
            )
        dec = DDP(
            dec.to(device),
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
            static_graph=False,
        )
        ddp_wrapped = True
    else:
        if need_encoder and enc is not None:
            enc = enc.to(device)
        dec = dec.to(device)

    start_epoch, best_dice = resume_if_needed(args, enc, dec, need_encoder)

    for ep in range(start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(ep)

        t0 = time.time()
        train_loss = train_one_epoch(
            encoder=enc if need_encoder else None,
            decoder=dec,
            loader=dl_train,
            optimizer=optimizer,
            scaler=scaler,
            device=str(device),
            use_amp=use_amp,
            feats_mode=args.feats,
            aux_weight=args.aux_weight,
            grad_clip=args.grad_clip,
            epoch=ep + 1,
            epochs=args.epochs,
            log_interval=args.print_freq,
            is_main=is_main_process(),
        )
        train_loss = all_reduce_mean_scalar(train_loss, device)

        val_loss, mIoU, mDice = validate(
            encoder=enc if need_encoder else None,
            decoder=dec,
            loader=dl_val,
            device=str(device),
            use_amp=use_amp,
            feats_mode=args.feats,
        )
        val_loss = all_reduce_mean_scalar(val_loss, device)
        mIoU = all_reduce_mean_scalar(mIoU, device)
        mDice = all_reduce_mean_scalar(mDice, device)

        scheduler.step()
        dt = time.time() - t0

        if is_main_process():
            print(
                f"[Epoch {ep + 1:03d}/{args.epochs}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"mIoU={mIoU:.4f}  mDice={mDice:.4f}  time={dt:.1f}s"
            )
            save_ckpt(
                args.out,
                ep + 1,
                best_dice,
                dec,
                enc if need_encoder else None,
                vars(args),
                name=f"epoch_{ep + 1:03d}",
                ddp_wrapped=ddp_wrapped,
            )
            save_ckpt(
                args.out,
                ep + 1,
                best_dice,
                dec,
                enc if need_encoder else None,
                vars(args),
                name="last",
                ddp_wrapped=ddp_wrapped,
            )
            if mDice > best_dice:
                best_dice = mDice
                save_ckpt(
                    args.out,
                    ep + 1,
                    best_dice,
                    dec,
                    enc if need_encoder else None,
                    vars(args),
                    name="best",
                    ddp_wrapped=ddp_wrapped,
                )
                print(f"⭐ New best mDice={best_dice:.4f}")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if is_main_process():
        print("🎉 训练完成。")


def parse_args():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--root", type=str,
                    default=r"/2024219001/data/pix2gestalt/train")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.008)
    ap.add_argument("--num_workers", type=int, default=8)
    # 特征/模型
    ap.add_argument("--feats", type=str, default="concat", choices=["vit", "sam", "concat"])
    ap.add_argument("--enc_size", type=str, default="base")
    ap.add_argument("--dec_size", type=str, default="large")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--aux_weight", type=float, default=1)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--port", type=int, default=0)

    ap.add_argument("--out", type=str, default="runs/exp_ddp0_bl_e8_r64")
    ap.add_argument("--resume", type=str, default="")

    ap.add_argument("--print_freq", type=int, default=10)

    args = ap.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_visible = len([x for x in visible.split(",") if x.strip() != ""])
    args.distributed = (torch.cuda.is_available() and num_visible > 1)
    if args.distributed and (args.port is None or args.port == 0):
        args.port = find_free_port()
    elif not args.distributed:
        args.port = 0
    return args, max(1, num_visible)


def main():
    args, world_size = parse_args()
    if not torch.cuda.is_available():
        print("cpu used")
        train_loop(rank=0, world_size=1, args=args)
        return
    if not args.distributed:
        train_loop(rank=0, world_size=1, args=args)
    else:
        mp.spawn(train_loop, nprocs=world_size, args=(world_size, args), join=True)


if __name__ == "__main__":
    main()
