"""Audio-caption dataset for stage-1 audio alignment (Clotho via HF datasets, parquet with embedded audio).

Mirrors the vision pretrain format: logical = [AUDIO_TOKEN] + caption + [EOT]; input=logical[:-1],
target=logical[1:]. MoEVLM expands AUDIO_TOKEN into CLAP features and carries the post-audio target onto
the last feature. Returns the raw waveform (CLAP's processor turns it into mel features in the train loop).
"""
from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import Dataset

from .multimodal import AUDIO_TOKEN, IGNORE_INDEX

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class ClothoAudio(Dataset):
    """Reads the HF parquet directly (pyarrow) and decodes embedded audio bytes with soundfile —
    no `datasets` Audio feature (which pulls torchcodec / breaks across pyarrow versions)."""
    def __init__(self, repo: str = "CLAPv2/Clotho", sr: int = 48000, max_cap: int = 64):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        train = [f for f in files if "train" in f.lower()] or files
        tabs = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset")) for f in sorted(train)]
        self.table = pa.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
        names = self.table.column_names
        self.audio_col = next(c for c in names if "audio" in c.lower())
        self.cap_col = next(c for c in names if "cap" in c.lower() or "text" in c.lower())
        self.ac = self.table.column(self.audio_col)
        self.cc = self.table.column(self.cap_col)
        self.sr = sr
        self.max_cap = max_cap
        print(f"[ClothoAudio] {repo} n={self.table.num_rows} audio_col={self.audio_col} cap_col={self.cap_col}",
              flush=True)

    def __len__(self):
        return self.table.num_rows

    def raw(self, i):
        """Return (waveform float32 @ self.sr, caption str) for inference/inspection."""
        import io
        import numpy as np
        import soundfile as sf
        a = self.ac[i].as_py()
        b = a["bytes"] if isinstance(a, dict) else a
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        cap = self.cc[i].as_py()
        if isinstance(cap, (list, tuple)):
            cap = cap[0] if cap else ""
        return np.ascontiguousarray(wav), str(cap)

    def __getitem__(self, i):
        import io
        import soundfile as sf
        import numpy as np
        a = self.ac[i].as_py()                        # {'bytes':..., 'path':...} or raw bytes
        b = a["bytes"] if isinstance(a, dict) else a
        try:
            wav, sr = sf.read(io.BytesIO(b), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)                 # mono
            if sr != self.sr:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception:
            wav = np.zeros(self.sr, dtype=np.float32)
        cap = self.cc[i].as_py()
        if isinstance(cap, (list, tuple)):
            cap = cap[0] if cap else ""
        cap_ids = ENC.encode_ordinary(" " + str(cap).strip())[:self.max_cap] + [EOT]
        logical = [AUDIO_TOKEN] + cap_ids
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        return torch.as_tensor(np.ascontiguousarray(wav), dtype=torch.float32), ids, tgt


def audio_collate(batch):
    """Returns (list of 1-D waveform tensors, padded input_ids, padded targets)."""
    wavs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return [w.numpy() for w in wavs], pad_ids, pad_tgt
