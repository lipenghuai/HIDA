from typing import Tuple
import torch
import torch.nn as nn


def get_cfg(model_size: str):
    # return dict[model cfg]
    model_size = model_size.lower()
    table = {
        "tiny": dict(embed_dim=256, depth=2, num_heads=8, mlp_ratio=8, num_mask_tokens=1),
        "small": dict(embed_dim=256, depth=3, num_heads=8, mlp_ratio=8, num_mask_tokens=1),
        "base": dict(embed_dim=256, depth=4, num_heads=8, mlp_ratio=8, num_mask_tokens=1),
        "large": dict(embed_dim=256, depth=6, num_heads=8, mlp_ratio=8, num_mask_tokens=1),
    }
    if model_size not in table:
        raise ValueError(f"Unknown model_size={model_size}, choose from {list(table.keys())}")
    return table[model_size]


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=(2, 3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(2, 3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def sine_pos_encoding_2d(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("pos-enc dim must be multiple of 4")
    device = device or "cpu"
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    outs = []
    for t in (x, y):
        t = t.reshape(-1, 1) * omega.reshape(1, -1)
        outs.extend([torch.sin(t), torch.cos(t)])
    return torch.cat(outs, dim=1)  # (H*W, dim)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, downsample_rate: int = 1, kv_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        inner = dim // downsample_rate
        kv_dim = kv_dim if kv_dim is not None else dim

        self.q_proj = nn.Linear(dim, inner, bias=True)
        self.k_proj = nn.Linear(kv_dim, inner, bias=True)
        self.v_proj = nn.Linear(kv_dim, inner, bias=True)
        self.out = nn.Linear(inner, dim, bias=True)

    def forward(self, q, k, v):
        B, Nq, Cq = q.shape
        Nk = k.shape[1]

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        h = self.num_heads
        q = q.view(B, Nq, h, -1).transpose(1, 2)  # (B,h,Nq,dim/h)
        k = k.view(B, Nk, h, -1).transpose(1, 2)
        v = v.view(B, Nk, h, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B,h,Nq,dim/h)

        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.out(out)


# ----------------------------
# Two-Way Transformer
# ----------------------------
class TwoWayAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads)

        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn_token = Attention(dim, num_heads, kv_dim=dim)

        self.norm_q3 = nn.LayerNorm(dim)
        self.mlp_q = MLP(dim, dim * mlp_ratio)

        self.norm_kv1 = nn.LayerNorm(dim)
        self.cross_attn_image = Attention(dim, num_heads, kv_dim=dim)

    def forward(self, tokens, img, img_pe):
        # 1) token self-attn
        q = tokens
        tokens = tokens + self.self_attn(self.norm_q1(q), self.norm_q1(q), self.norm_q1(q))
        # 2) token <- image（keys 加上 PE）
        tokens = tokens + self.cross_attn_token(self.norm_q2(tokens), self.norm_q2(img + img_pe), self.norm_q2(img))
        # 3) token MLP
        tokens = tokens + self.mlp_q(self.norm_q3(tokens))
        # 4) image <- token
        img = img + self.cross_attn_image(self.norm_kv1(img + img_pe), self.norm_kv1(tokens), self.norm_kv1(tokens))
        return tokens, img


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TwoWayAttentionBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_attn_norm = nn.LayerNorm(dim)
        self.final_attn = Attention(dim, num_heads)

    def forward(self, tokens, img, img_pe):
        for blk in self.blocks:
            tokens, img = blk(tokens, img, img_pe)
        tokens = tokens + self.final_attn(
            self.final_attn_norm(tokens),
            self.final_attn_norm(img + img_pe),
            self.final_attn_norm(img),
        )
        return tokens, img


class AttnMaskDecoder(nn.Module):
    def __init__(self, in_ch: int = 256, model_size: str = "base"):
        super().__init__()
        cfg = get_cfg(model_size)
        dim = cfg["embed_dim"]
        depth = cfg["depth"]
        heads = cfg["num_heads"]
        mlp_ratio = cfg["mlp_ratio"]
        self.num_mask_tokens = cfg["num_mask_tokens"]
        self.in_proj = nn.Conv2d(in_ch, dim, 1, bias=True)

        self.register_buffer("_pe_hw", torch.zeros(2, dtype=torch.long), persistent=False)
        self.register_buffer("_pe_cache", torch.zeros(1, 1, dim), persistent=False)

        self.iou_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_tokens = nn.Parameter(torch.zeros(1, self.num_mask_tokens, dim))
        nn.init.trunc_normal_(self.iou_token, std=0.02)
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)

        self.transformer = TwoWayTransformer(depth=depth, dim=dim, num_heads=heads, mlp_ratio=mlp_ratio)

        self.output_upscaling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, dim // 2, 3, padding=1), LayerNorm2d(dim // 2), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1), LayerNorm2d(dim // 4), nn.GELU(),
        )

        up_ch = dim // 4
        self.hypernets = nn.ModuleList([MLP(dim, dim * 2) for _ in range(self.num_mask_tokens)])  # 输出 dim
        self.hyper_out_proj = nn.Linear(dim, up_ch)  # 注意：这里输入维度是 dim（修复形状不匹配）

        self.iou_head = nn.Sequential(
            MLP(dim, dim * 4),
            nn.Linear(dim, self.num_mask_tokens)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _pos_encoding(self, h: int, w: int, dim: int, device) -> torch.Tensor:
        if self._pe_hw[0].item() == h and self._pe_hw[1].item() == w and self._pe_cache.numel() == h * w * dim:
            return self._pe_cache.view(1, h * w, dim)
        pe = sine_pos_encoding_2d(h, w, dim, device=device).unsqueeze(0)
        self._pe_hw = torch.tensor([h, w], device=device)
        self._pe_cache = pe
        return pe

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # , torch.Tensor
        """
        feats: (B, C_in, 64, 64)
        return: (mask_logits, iou_scores)
          mask_logits: (B, 1, 256, 256)
          iou_scores : (B, 1)
        """
        B, _, H, W = feats.shape
        assert H == 64 and W == 64, "expect 64x64 input feature"

        x = self.in_proj(feats)  # (B, dim, 64, 64)
        img = x.flatten(2).transpose(1, 2)  # (B, 4096, dim)
        img_pe = self._pos_encoding(64, 64, x.shape[1], x.device).expand(B, -1, -1)

        # tokens: [IoU] + [Mask]
        iou_tok = self.iou_token.expand(B, 1, -1)
        mask_toks = self.mask_tokens.expand(B, self.num_mask_tokens, -1)
        tokens = torch.cat([iou_tok, mask_toks], dim=1)  # (B, 1+M, dim)

        tokens, img = self.transformer(tokens, img, img_pe)

        iou_token_out = tokens[:, 0:1, :]  # (B,1,dim)
        mask_tokens_out = tokens[:, 1:, :]  # (B,M,dim)

        up = self.output_upscaling(x)  # (B, up_ch=dim//4, 256, 256)
        B, Cup, Hup, Wup = up.shape
        up_flat = up.view(B, Cup, Hup * Wup)  # (B, C_up, HW)

        weights = []
        for m_idx in range(mask_tokens_out.shape[1]):
            w = self.hyper_out_proj(self.hypernets[m_idx](mask_tokens_out[:, m_idx, :]))  # (B, C_up)
            weights.append(w)
        weights = torch.stack(weights, dim=1)  # (B, M, C_up)

        # (B, M, HW) = (B, M, C_up) × (B, C_up, HW)
        masks = torch.einsum("bmc, bch -> bmh", weights, up_flat)
        masks = masks.view(B, -1, Hup, Wup)  # (B, M, 256, 256)

        # IoU
        iou_scores = self.iou_head(iou_token_out.squeeze(1))  # (B, M)

        return masks[:, :1, :, :], iou_scores[:, :1]
        # return masks[:, :1, :, :]
