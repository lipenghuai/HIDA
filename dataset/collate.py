from __future__ import annotations
from typing import List, Dict, Any
import torch

def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [b["key"] for b in batch]
    x5 = torch.stack([b["x5"] for b in batch], dim=0)                  # B,5,H,W
    sam_feats = torch.stack([b["sam_feats"] for b in batch], dim=0)    # B,1,256,64,64
    if all(b["y_whole"] is not None for b in batch):
        y_whole = torch.stack([b["y_whole"] for b in batch], dim=0)    # B,1,H,W
    else:
        y_whole = None
    paths = [b["paths"] for b in batch]
    return {"keys": keys, "x5": x5, "sam_feats": sam_feats, "y_whole": y_whole, "paths": paths}
