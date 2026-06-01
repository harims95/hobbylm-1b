"""Video = sampled frames through the SAME SigLIP2 image encoder + spatial pooling.

No new encoder and no new projector: each sampled frame is encoded by SigLIP2 (729 tokens), spatially
pooled to a small grid, and the per-frame token grids are concatenated into one <video> sequence that
goes through the image `mm_projector`. So a model that can describe images can describe videos.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vision import SiglipVision


class SiglipVideo:
    def __init__(self, siglip: SiglipVision, pool_hw: tuple[int, int] = (6, 6)):
        self.siglip = siglip                 # reuse the frozen image encoder
        self.pool_hw = pool_hw               # per-frame spatial pool target (6x6 = 36 tokens/frame)

    @torch.no_grad()
    def encode_frames(self, frames) -> torch.Tensor:
        """frames: list of PIL.Image (one video). Returns (1, n_frames*H*W, hidden) video tokens."""
        feats = self.siglip.encode(frames)            # (N, 729, C)
        N, T, C = feats.shape
        side = int(round(T ** 0.5))                   # 27 for SigLIP2-so400m-384 (27x27)
        f = feats[:, :side * side, :].reshape(N, side, side, C).permute(0, 3, 1, 2)  # (N,C,side,side)
        f = F.adaptive_avg_pool2d(f.float(), self.pool_hw).to(feats.dtype)           # (N,C,h,w)
        f = f.flatten(2).transpose(1, 2)              # (N, h*w, C)
        return f.reshape(1, -1, C)                    # (1, N*h*w, C) — one video's tokens


def sample_frames(path: str, n: int = 8):
    """Uniformly sample n frames from a video file -> list of PIL.Image (needs opencv)."""
    import cv2
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(0, total - 1), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames or [Image.new("RGB", (384, 384))]
