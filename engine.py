# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + self.eps
        den = (p.pow(2) + target.pow(2)).sum(dim=(1, 2, 3)) + self.eps
        return 1.0 - (num / den).mean()


@torch.no_grad()
def _miou_mean(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return float((inter / (union + 1e-6)).mean().cpu())


@torch.no_grad()
def _mdice_mean(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= thr).float()
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return float((num / den).mean().cpu())


def _select_feats(
        feats_mode: str,
        x5: torch.Tensor,
        batch: Dict[str, Any],
        encoder: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    aux_loss = torch.tensor(0.0, device=x5.device)
    sam_feats = batch["sam_feats"].to(x5.device).squeeze(1)  # (B,256,64,64)

    if feats_mode == "sam":
        return sam_feats, aux_loss

    assert encoder is not None, "feats_mode=vit/concat encoder"
    vit_feats = encoder(x5)  # (B,256,64,64)
    if hasattr(encoder, "get_aux_loss"):
        aux_loss = encoder.get_aux_loss()

    if feats_mode == "vit":
        return vit_feats, aux_loss
    else:
        return torch.cat([vit_feats, sam_feats], dim=1), aux_loss  # (B,512,64,64)


def train_one_epoch(
        *,
        encoder: torch.nn.Module | None,
        decoder: torch.nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: str = "cuda",
        use_amp: bool = True,
        feats_mode: str = "concat",
        aux_weight: float = 1.0,
        grad_clip: float | None = 1.0,
        # 为了打印 epoch 内进度
        epoch: int | None = None,
        epochs: int | None = None,
        log_interval: int = 10,
        is_main: bool = True,
) -> float:
    if feats_mode in ("vit", "concat"):
        assert encoder is not None
        encoder.train()
    decoder.train()

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    loss_sum, n = 0.0, 0
    num_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        x5 = batch["x5"].to(device, non_blocking=True)  # (B,5,H,W)
        y = batch["y_whole"].to(device, non_blocking=True)  # (B,1,H,W)

        optimizer.zero_grad(set_to_none=True)

        # ---------- forward ----------
        with autocast(enabled=use_amp):
            feats, aux = _select_feats(feats_mode, x5, batch, encoder)
            mask_logits, _ = decoder(feats)  # (B,1,H,W)
            loss_main = 0.5 * bce(mask_logits, y) + 0.5 * dice(mask_logits, y)
            loss = loss_main + aux_weight * aux

        # ---------- NaN/Inf ----------
        is_finite = torch.isfinite(loss)
        local_bad = torch.zeros(1, device=x5.device)
        if not bool(is_finite):
            local_bad[0] = 1.0

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.SUM)

        global_bad = local_bad.item() > 0.5

        if global_bad:
            if is_main:
                if epoch is not None and epochs is not None:
                    prefix = f"[Epoch {epoch:03d}/{epochs}]"
                else:
                    prefix = "[Train]"

                msg = (
                    f"{prefix} step {step:04d}/{num_batches:04d} "
                    f"[NaN_BATCH] 检测到 NaN/Inf，跳过该 batch"
                )

                if "key" in batch:
                    try:
                        idx = batch["key"]
                        if isinstance(idx, torch.Tensor):
                            idx = idx.detach().cpu().tolist()
                        msg += f" indices={idx}"
                    except Exception:
                        pass

                try:
                    msg += (
                        f" loss_main={float(loss_main.detach().cpu().item()):.4f} "
                        f"aux={float(aux.detach().cpu().item()):.4f}"
                    )
                except Exception:
                    pass

                print(msg, flush=True)

            continue

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                if encoder is not None and feats_mode in ("vit", "concat"):
                    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                if encoder is not None and feats_mode in ("vit", "concat"):
                    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            optimizer.step()

        loss_sum += float(loss.item())
        n += 1

        if is_main and (step % log_interval == 0 or step == num_batches):
            avg_loss = loss_sum / max(n, 1)
            if epoch is not None and epochs is not None:
                print(
                    f"[Epoch {epoch:03d}/{epochs}] "
                    f"step {step:04d}/{num_batches:04d} "
                    f"({step / num_batches * 100:5.1f}%)  "
                    f"train_loss={avg_loss:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[Train] step {step:04d}/{num_batches:04d} "
                    f"({step / num_batches * 100:5.1f}%)  "
                    f"train_loss={avg_loss:.4f}",
                    flush=True,
                )

    return loss_sum / max(n, 1)


@torch.no_grad()
def validate(
        *,
        encoder: torch.nn.Module | None,
        decoder: torch.nn.Module,
        loader,
        device: str = "cuda",
        use_amp: bool = True,
        feats_mode: str = "concat",
) -> tuple[float, float, float]:
    """return (val_loss, mIoU, mDice)"""
    if feats_mode in ("vit", "concat") and encoder is not None:
        encoder.eval()
    decoder.eval()

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    loss_sum, n = 0.0, 0
    iou_list, dice_list = [], []
    for batch in loader:
        x5 = batch["x5"].to(device, non_blocking=True)
        y = batch["y_whole"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            feats, aux = _select_feats(feats_mode, x5, batch, encoder)
            logits, _ = decoder(feats)
            loss = 0.5 * bce(logits, y) + 0.5 * dice(logits, y) + 0.0 * aux

        loss_sum += float(loss.item())
        n += 1
        iou_list.append(_miou_mean(logits, y))
        dice_list.append(_mdice_mean(logits, y))

    val_loss = loss_sum / max(n, 1)
    mIoU = float(np.mean(iou_list)) if iou_list else 0.0
    mDice = float(np.mean(dice_list)) if dice_list else 0.0
    return val_loss, mIoU, mDice
