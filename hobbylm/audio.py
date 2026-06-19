"""Frozen CLAP audio encoder wrapper for the MoE-VLM (Phase 3, general audio).

Loads `laion/clap-htsat-unfused`, runs waveforms through the CLAP audio tower (HTSAT) under no_grad,
and returns a sequence of audio patch features (B, T, hidden) to be projected + spliced at <audio>
sentinels by MoEVLM (same mechanism as image patches). Frozen in every training stage. Lazy import.

CLAP expects 48 kHz mono. HTSAT's last_hidden_state is a 2D feature map (B, C, H, W); we flatten the
spatial grid into a token sequence (B, H*W, C).
"""
from __future__ import annotations

import torch
import torch.nn as nn

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000


class ClapAudio(nn.Module):
    def __init__(self, model_id: str = CLAP_ID, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        self.processor = ClapProcessor.from_pretrained(model_id)
        full = ClapModel.from_pretrained(model_id, torch_dtype=dtype)
        self.audio = full.audio_model.to(device).eval()
        for p in self.audio.parameters():
            p.requires_grad = False
        self.device = device
        self.dtype = dtype
        self.hidden = self.audio.config.hidden_size

    @torch.no_grad()
    def encode(self, waveforms, sr: int = CLAP_SR) -> torch.Tensor:
        """waveforms: list of 1-D float arrays (mono). Returns (B, T, hidden) audio token features."""
        inputs = self.processor(audios=waveforms, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else None) for k, v in inputs.items()}
        out = self.audio(**inputs).last_hidden_state
        if out.ndim == 4:                      # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2)
        return out
