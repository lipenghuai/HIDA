from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Sequence
from torch.utils.data import DataLoader, random_split
import torch

from .collate import default_collate_fn
from .datasets import Pix2GestaltOccluDataset

_DATASETS = {
    "pix2gestalt_occlu": Pix2GestaltOccluDataset,
}

def list_datasets():
    return sorted(list(_DATASETS.keys()))

def build_dataset(name: str, **kwargs):
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list_datasets()}")
    return _DATASETS[name](**kwargs)

def build_loader(dataset,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 collate_fn=default_collate_fn):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last,
                      collate_fn=collate_fn)

def make_splits(dataset,
                val_ratio: float = 0.1,
                batch_size: int = 8,
                num_workers: int = 4,
                seed: int = 42):
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val = random_split(dataset, [n_train, n_val], generator=g)
    dl_train = build_loader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_val   = build_loader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers//2))
    return dl_train, dl_val
