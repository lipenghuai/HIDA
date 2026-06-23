from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import ViTSegmentationEncoder
from .decoders import AttnMaskDecoder


class OATRSegModel(nn.Module):
    def __init__(self,
                 encoder: ViTSegmentationEncoder,
                 decoder: AttnMaskDecoder,
                 upsample_to_input: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsample_to_input = upsample_to_input

    @torch.no_grad()
    def check_input_size(self, x: torch.Tensor) -> None:
        h, w = x.shape[-2:]
        if (h, w) != tuple(self.encoder.input_size):
            raise ValueError(
                f"input size {h, w} != encoder.input_size {self.encoder.input_size}. "
            )

    def forward(self, x: torch.Tensor):
        self.check_input_size(x)
        feats = self.encoder(x)  # (B, C=256, 64, 64) 默认
        mask_logits, iou_scores = self.decoder(feats)  # (B,1,256,256), (B,1)

        if self.upsample_to_input and mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(mask_logits, size=x.shape[-2:],
                                        mode="bilinear", align_corners=False)
        return mask_logits, iou_scores

    def get_aux_loss(self) -> torch.Tensor:
        if hasattr(self.encoder, "get_aux_loss"):
            return self.encoder.get_aux_loss()
        return torch.tensor(0., device=next(self.parameters()).device)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]
