"""LLaVA-Pretrain (LAION-CC-SBU-558K) dataset for stage-1 alignment.

Each sample -> a logical sequence [IMAGE_TOKEN] + caption_tokens + [EOT]; we train next-token, so
input_ids = logical[:-1], targets = logical[1:]. MoEVLM expands the IMAGE_TOKEN into 729 SigLIP2
features and carries the post-image target onto the last feature (see multimodal.build_inputs_embeds),
so the model learns to produce the caption conditioned on the image.
"""
from __future__ import annotations

import json
import os

import tiktoken
import torch
from PIL import Image
from torch.utils.data import Dataset

from multimodal import IMAGE_TOKEN, IGNORE_INDEX

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class LlavaPretrain(Dataset):
    def __init__(self, json_path: str, image_root: str, max_cap: int = 128):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.max_cap = max_cap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        # conversations: [{"from":"human","value":"<image>\n..."}, {"from":"gpt","value":"<caption>"}]
        caption = ex["conversations"][1]["value"].strip()
        cap_ids = ENC.encode_ordinary(" " + caption)[:self.max_cap] + [EOT]
        logical = [IMAGE_TOKEN] + cap_ids
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        try:
            img = Image.open(os.path.join(self.image_root, ex["image"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384))   # tolerate a missing/corrupt image
        return img, ids, tgt


def collate(batch):
    """Pad input_ids (with EOT) and targets (with IGNORE) to the batch max; return (PIL list, ids, tgt)."""
    imgs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return list(imgs), pad_ids, pad_tgt
