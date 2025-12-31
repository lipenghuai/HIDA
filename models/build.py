from __future__ import annotations
from typing import Tuple, Iterable, Optional, Dict, Any

import torch
from torch import nn

from .encoders import ViTSegmentationEncoder
from .decoders import AttnMaskDecoder
from .seg_model import OATRSegModel


_SUPPORTED = {
    "oatr_attn",
    "oatr_encoder",
    "attn_decoder",
}


def list_models():
    return sorted(list(_SUPPORTED))


def _build_encoder(
    *,
    input_size: Tuple[int, int] = (256, 256),
    model_size: str = "base",
    out_channels: int = 256,
    token_grid: Tuple[int, int] = (64, 64),
    vit_patch: int = 16,
    dropout: float = 0.1,
    emb_dropout: float = 0.1,
    extract_layers: Optional[Iterable[int]] = (1, 2, 3, 4, 5, 6, 7, 8),
    num_experts: int = 4,
    router_hidden: int = 16,
    oatr_alpha: float = 4.0,
    oatr_beta: float = 1.0,
    lb_weight: float = 1e-3,
) -> ViTSegmentationEncoder:
    enc = ViTSegmentationEncoder(
        input_size=input_size,
        model_size=model_size,
        out_channels=out_channels,
        token_grid=token_grid,
        vit_patch=vit_patch,
        dropout=dropout,
        emb_dropout=emb_dropout,
        extract_layers=extract_layers,
        num_experts=num_experts,
        router_hidden=router_hidden,
        oatr_alpha=oatr_alpha,
        oatr_beta=oatr_beta,
        lb_weight=lb_weight,
    )
    return enc


def _build_decoder(
    *,
    in_ch: int = 256,
    model_size: str = "base",
) -> AttnMaskDecoder:
    dec = AttnMaskDecoder(in_ch=in_ch, model_size=model_size)
    return dec


def build_model(name: str,
                **kwargs) -> nn.Module:
    name = name.lower()
    if name not in _SUPPORTED:
        raise ValueError(f"Unknown model name: {name}. Supported: {sorted(list(_SUPPORTED))}")

    enc_kwargs = {k: kwargs[k] for k in list(kwargs.keys())
                  if k in {
                      "input_size", "model_size", "out_channels", "token_grid", "vit_patch",
                      "dropout", "emb_dropout", "extract_layers",
                      "num_experts", "router_hidden", "oatr_alpha", "oatr_beta", "lb_weight"
                  }}
    dec_kwargs = {
        "in_ch": kwargs.get("in_ch", 512),
        "model_size": kwargs.get("dec_model_size", "base"),
    }
    wrap_kwargs = {
        "upsample_to_input": kwargs.get("upsample_to_input", False)
    }

    if name == "oatr_encoder":
        return _build_encoder(**enc_kwargs)
    if name == "attn_decoder":
        return _build_decoder(**dec_kwargs)

    # name == "oatr_attn"
    encoder = _build_encoder(**enc_kwargs)
    decoder = _build_decoder(**dec_kwargs)
    return OATRSegModel(encoder, decoder, **wrap_kwargs)
