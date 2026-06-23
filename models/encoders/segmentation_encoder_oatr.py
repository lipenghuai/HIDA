# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from vit_pytorch import ViT
except ImportError:
    raise ImportError("pip install vit-pytorch")


# --------------------------
# tools
# --------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _binary_edge(mask: torch.Tensor) -> torch.Tensor:
    maxed = F.max_pool2d(mask, 3, stride=1, padding=1)
    mined = 1.0 - F.max_pool2d(1.0 - mask, 3, stride=1, padding=1)
    edge = (maxed - mined).clamp(0, 1)
    return (edge > 0).to(mask.dtype)

import torch
import torch.nn as nn

class ResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expert_dw_kernel, act):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                expert_dw_kernel,
                padding=expert_dw_kernel // 2,
                groups=in_channels,
                bias=False,
            ),
            LayerNorm2d(in_channels),
            act(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.proj = None

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        return out

def _distance_like(edge: torch.Tensor) -> torch.Tensor:
    B, _, H, W = edge.shape
    if _HAS_CV2:
        outs = []
        for b in range(B):
            e = edge[b, 0].detach().float().cpu().numpy()
            inv = (1.0 - e).astype('float32')
            dt = cv2.distanceTransform((inv * 255).astype('uint8'), cv2.DIST_L2, 3).astype('float32')
            if dt.max() > 0:
                dt = dt / (dt.max() + 1e-6)
            outs.append(torch.from_numpy(dt)[None, None])
        return torch.cat(outs, dim=0).to(edge.device, edge.dtype)
    else:
        inv = 1.0 - edge
        acc = torch.zeros_like(inv)
        for k in (3, 7, 15, 31):
            acc = acc + F.avg_pool2d(inv, k, stride=1, padding=k // 2)
        acc = acc / 4.0
        m, M = acc.amin(dim=(2, 3), keepdim=True), acc.amax(dim=(2, 3), keepdim=True)
        return (acc - m) / (M - m + 1e-6)


class OcclusionMoEBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_experts: int = 4,
            router_hidden: int = 16,
            router_with_mask: bool = True,
            expert_dw_kernel: int = 3,
            expert_act: str = "gelu",
            lb_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.lb_weight = lb_weight

        act = nn.GELU if expert_act == "gelu" else nn.ReLU

        self.experts = nn.ModuleList([
            # nn.Sequential(
            #     nn.Conv2d(in_channels, in_channels, expert_dw_kernel, padding=expert_dw_kernel // 2,
            #               groups=in_channels, bias=False),
            #     LayerNorm2d(in_channels),
            #     act(),
            #     nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # )
            ResidualExpert(
                in_channels=in_channels,
                out_channels=out_channels,
                expert_dw_kernel=expert_dw_kernel,
                act=act,
            )
            for _ in range(num_experts)
        ])

        occ_in_ch = 2 + (1 if router_with_mask else 0)  # grad, dist, (mask)
        self.pre_reduce = nn.Conv2d(in_channels, router_hidden, 1, bias=False)
        self.router = nn.Sequential(
            nn.Conv2d(router_hidden + occ_in_ch, router_hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(router_hidden, num_experts, 1, bias=True),
        )

        self.last_lb_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, occ_feats: torch.Tensor) -> torch.Tensor:
        z = self.pre_reduce(x)  # (B,router_hidden,H,W)
        logits = self.router(torch.cat([z, occ_feats], 1))  # (B,K,H,W)
        gates = logits.softmax(dim=1)  # (B,K,H,W)

        mean_pi = gates.mean(dim=(0, 2, 3))  # (K,)
        lb = (mean_pi * (mean_pi.clamp_min(1e-8).log())).sum()  # 近似 -H
        self.last_lb_loss = self.lb_weight * lb

        y_stack = torch.stack([e(x) for e in self.experts], dim=1)  # (B,K,C,H,W)
        y = (gates.unsqueeze(2) * y_stack).sum(dim=1)  # (B,C,H,W)
        return y


class ViTSegmentationEncoder(nn.Module):
    _MODEL_SIZES = {
        "small": dict(dim=256, depth=4, heads=8,  mlp_dim=1024),
        "base":  dict(dim=512, depth=6, heads=8,  mlp_dim=2048),
        "large": dict(dim=768, depth=8, heads=12, mlp_dim=3072),
        "xlarge": dict(dim=1024, depth=12, heads=16, mlp_dim=4096),
    }

    def __init__(
            self,
            input_size: Tuple[int, int] = (256, 256),
            model_size: str = "tiny",
            out_channels: int = 256,
            token_grid: Tuple[int, int] = (64, 64),
            vit_patch: int = 16,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            extract_layers: Iterable[int] | None = None,  # 1-based
            num_experts: int = 4,
            router_hidden: int = 16,
            oatr_alpha: float = 4.0,
            oatr_beta: float = 1.0,
            lb_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if model_size not in self._MODEL_SIZES:
            raise ValueError(f"model_size={model_size} {list(self._MODEL_SIZES.keys())}")
        self.input_size = tuple(input_size)
        self.out_channels = out_channels
        self.token_grid = tuple(token_grid)
        self.oatr_alpha = oatr_alpha
        self.oatr_beta = oatr_beta

        H, W = self.input_size
        if H % vit_patch != 0 or W % vit_patch != 0:
            raise ValueError(f"input_size={self.input_size} 需能被 vit_patch={vit_patch} 整除。")
        self.patch_size = (vit_patch, vit_patch)

        cfg = self._MODEL_SIZES[model_size]
        dim = cfg["dim"]
        depth = cfg["depth"]
        heads = cfg["heads"]
        mlp_dim = cfg["mlp_dim"]

        self.vit = ViT(
            image_size=self.input_size,
            patch_size=self.patch_size,
            num_classes=out_channels,
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
            channels=5, dropout=dropout, emb_dropout=emb_dropout, pool="cls",
        )

        if extract_layers is None:
            extract_layers = (max(1, depth - 3), depth - 1, depth)
        self.extract_layers: List[int] = sorted(list(extract_layers))

        self.per_layer_block = nn.ModuleList([
            OcclusionMoEBlock(in_channels=dim, out_channels=out_channels,
                              num_experts=num_experts, router_hidden=router_hidden, lb_weight=lb_weight)
            for _ in self.extract_layers
        ])

        self.fuse_logits = nn.Parameter(torch.zeros(len(self.extract_layers)))

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

        self._cached_lb_loss: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _check_input(self, x: torch.Tensor) -> None:
        b, c, h, w = x.shape
        if (h, w) != self.input_size:
            raise ValueError(f"输入 {(h, w)} 与初始化的 input_size={self.input_size} 不一致。")
        if c != 5:
            raise ValueError(f"期望输入通道=5 (RGB+mask+depth)，但得到 {c}。")

    def _tokens_to_map(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        B, N, C = tokens.shape
        gh, gw = grid_hw
        assert N == gh * gw, f"Token 数与网格不匹配: N={N}, gh*gw={gh * gw}"
        return tokens.transpose(1, 2).contiguous().view(B, C, gh, gw)

    def _build_occ_feats(self, depth: torch.Tensor, mask: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        d_grad = _sobel_grad(depth)
        d_grad = d_grad / (d_grad.amax(dim=(2, 3), keepdim=True) + 1e-6)
        mask_bin = (mask > 0.5).float()
        edge = _binary_edge(mask_bin)
        dist = _distance_like(edge)
        gh, gw = grid_hw
        d_grad_ds = F.interpolate(d_grad, size=(gh, gw), mode="bilinear", align_corners=False)
        dist_ds = F.interpolate(dist, size=(gh, gw), mode="bilinear", align_corners=False)
        mask_ds = F.interpolate(mask_bin, size=(gh, gw), mode="nearest")
        return torch.cat([d_grad_ds, dist_ds, mask_ds], dim=1)  # (B,3,gh,gw)

    def get_aux_loss(self) -> torch.Tensor:
        if self._cached_lb_loss is None:
            return torch.tensor(0., device=next(self.parameters()).device)
        return self._cached_lb_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        B, _, H, W = x.shape

        mask = x[:, 3:4]
        depth = x[:, 4:5]

        vit = self.vit
        gh, gw = (H // self.patch_size[0], W // self.patch_size[1])  # 例如 (16,16)
        occ_raw = self._build_occ_feats(depth, mask, (gh, gw))  # (B,3,gh,gw)

        tok = vit.to_patch_embedding(x)  # (B,N,dim)
        B, N, dim = tok.shape

        cls_tokens = vit.cls_token.expand(B, -1, -1)
        tok = torch.cat((cls_tokens, tok), dim=1)
        tok = tok + vit.pos_embedding[:, :(N + 1)]
        tok = vit.dropout(tok)

        feats_2d: List[torch.Tensor] = []
        lb_losses: List[torch.Tensor] = []
        layer_idx = 0

        for item in vit.transformer.layers:
            if isinstance(item, (list, tuple, nn.ModuleList)) and len(item) == 2:
                attn, ff = item[0], item[1]
                tok = attn(tok)
                tok = ff(tok)
                layer_idx += 1
            else:
                tok = item(tok)
                layer_idx += 0.5
                if abs(round(layer_idx) - layer_idx) < 1e-6:
                    layer_idx = int(round(layer_idx))

            if layer_idx in self.extract_layers:
                tokens_wo_cls = tok[:, 1:, :]
                fmap = self._tokens_to_map(tokens_wo_cls, (gh, gw))  # (B,dim,gh,gw)

                depth_grad = occ_raw[:, 0:1]
                dist_edge = occ_raw[:, 1:2]
                mask_ds = occ_raw[:, 2:3]
                occ_signal = torch.cat([
                    torch.sigmoid(self.oatr_alpha * depth_grad + self.oatr_beta * dist_edge),
                    dist_edge,
                    mask_ds
                ], dim=1)  # (B,3,gh,gw)

                blk = self.per_layer_block[self.extract_layers.index(layer_idx)]
                y = blk(fmap, occ_signal)  # (B,out_ch,gh,gw)
                feats_2d.append(y)
                if blk.last_lb_loss is not None:
                    lb_losses.append(blk.last_lb_loss)

        assert len(feats_2d) == len(self.extract_layers), \
            f"抽到 {len(feats_2d)} 个特征，但 extract_layers={self.extract_layers}"

        fuse_w = self.fuse_logits.softmax(dim=0)  # (L,)
        fused = torch.stack(feats_2d, dim=0)  # (L,B,C,gh,gw)
        fused = (fuse_w.view(-1, 1, 1, 1, 1) * fused).sum(dim=0)  # (B,C,gh,gw)

        # 上采样到目标 token_grid（默认 64×64），再做 3×3 混合
        fused_up = F.interpolate(fused, size=self.token_grid, mode="bilinear", align_corners=False)
        out = self.out_proj(fused_up)  # (B,C,token_grid)

        self._cached_lb_loss = torch.stack(lb_losses).sum() if len(lb_losses) else \
            torch.tensor(0., device=out.device)
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = ViTSegmentationEncoder(
        input_size=(256, 256),
        model_size="base",  # 轻量配置
        out_channels=256,  # 256
        token_grid=(64, 64),  # 最终输出固定 64×64
        vit_patch=16,  # 16×16 token 网格，Attention 仅 256 tokens
        extract_layers=(4, 5, 6),  # tiny(depth=6)，抽最后三层
        num_experts=4,
        router_hidden=16,
        oatr_alpha=4.0, oatr_beta=1.0,
        lb_weight=1e-3,
    ).to(device).eval()

    x = torch.randn(1, 5, 256, 256, device=device)
    with torch.no_grad():
        y = enc(x)
    print("Encoder out:", tuple(y.shape), " AuxLoss:", float(enc.get_aux_loss()))


    # 端到端最小训练步
    class Head(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.head = nn.Conv2d(in_ch, 1, 1)

        def forward(self, z): return self.head(z)


    head = Head(enc.out_channels).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    enc.train()
    head.train()
    x = torch.randn(2, 5, 256, 256, device=device)
    y_gt = (torch.rand(2, 1, 64, 64, device=device) > 0.5).float()
    pp = enc(x)
    logit = head(enc(x))
    loss = crit(logit, y_gt) + enc.get_aux_loss()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("Train step ok. loss=", float(loss))
