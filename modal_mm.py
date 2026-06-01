"""Modal harness for the multimodal MoE-VLM (image + audio). TinyLLaVA-style.

  # download the LLaVA-Pretrain (LAION-CC-SBU-558K) alignment data to a volume:
  python -m modal run modal_mm.py --action download

  # GPU smoke: real SigLIP2 + 500M_ctx2048 backbone + MoEVLM forward/backward on a synthetic image:
  python -m modal run modal_mm.py --action smoke
"""
import modal

# image with the vision/LLM training deps (separate from the lm-eval image)
vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "transformers>=4.50,<5", "pillow", "numpy",
                 "huggingface-hub", "accelerate", "sentencepiece", "tiktoken")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(".", "/root/moe-lab")
)

app = modal.App("moe-vlm", image=vlm_image)
data_vol = modal.Volume.from_name("llava-data", create_if_missing=True)   # LLaVA-Pretrain images + json
runs_vol = modal.Volume.from_name("fineweb10B")                           # holds /runs/500M_ctx2048/model.pt
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)     # SigLIP2 weights cache
HF = modal.Secret.from_name("huggingface")

BACKBONE = "/data/runs/500M_ctx2048/model.pt"


@app.function(image=vlm_image, volumes={"/llava": data_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def download():
    """Download LLaVA-Pretrain (LAION-CC-SBU-558K): captions json + ~558K images (images.zip ~24GB)."""
    import os, zipfile
    from huggingface_hub import hf_hub_download
    repo = "liuhaotian/LLaVA-Pretrain"
    os.makedirs("/llava", exist_ok=True)
    for fn in ["blip_laion_cc_sbu_558k.json"]:
        if not os.path.exists(f"/llava/{fn}"):
            hf_hub_download(repo, fn, repo_type="dataset", local_dir="/llava")
            print(f"got {fn}", flush=True)
    img_dir = "/llava/images"
    if not os.path.isdir(img_dir) or not os.listdir(img_dir):
        print("downloading images.zip (~24GB) ...", flush=True)
        zp = hf_hub_download(repo, "images.zip", repo_type="dataset", local_dir="/llava")
        print("unzipping ...", flush=True)
        os.makedirs(img_dir, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(img_dir)
        os.remove(zp)
        data_vol.commit()
    n = sum(len(fs) for _, _, fs in os.walk(img_dir))
    print(f"LLaVA-Pretrain ready: {n} image files under {img_dir}", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def smoke():
    """End-to-end vision path on a synthetic image: SigLIP2 -> project -> splice -> 500M_ctx2048 -> loss/grad."""
    import os, sys, numpy as np, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN, IGNORE_INDEX
    from generate import load_model

    dev = torch.device("cuda")
    torch.manual_seed(0)
    enc = SiglipVision(device=dev)
    img = Image.fromarray((np.random.rand(384, 384, 3) * 255).astype("uint8"))
    feats = enc.encode([img])                                   # (1, N, hidden)
    print(f"SigLIP2 hidden={enc.hidden}  features={tuple(feats.shape)}", flush=True)

    llm, cfg, vloss, step = load_model(BACKBONE, dev)
    print(f"backbone {BACKBONE}: d{cfg.d_model} L{cfg.n_layers} val={vloss} step={step}", flush=True)
    llm.train()
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)

    enc_tok = tiktoken.get_encoding("gpt2")
    cap = enc_tok.encode_ordinary("a photo of a cat sitting on a chair")
    ids = torch.tensor([[IMAGE_TOKEN] + cap], device=dev)       # <image> then caption
    tgt = torch.tensor([[IGNORE_INDEX] + cap], device=dev)      # predict caption only
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = vlm(ids, image_features=feats, targets=tgt)
    loss.backward()
    g = vlm.mm_projector.net[0].weight.grad
    merged_len = feats.shape[1] + len(cap)
    print(f"merged seq len={merged_len}  loss={loss.item():.4f}  ce={parts['ce'].item():.4f}", flush=True)
    print(f"projector grad finite={bool(torch.isfinite(g).all())}  "
          f"backbone frozen? (grad on embed={llm.embed.weight.grad is not None})", flush=True)
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    print("VISION SMOKE OK", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_stage1(max_steps: int = 2000, micro: int = 24, lr: float = 1e-3, warmup: int = 100,
                 save_name: str = "500M_vlm_stage1", log_every: int = 20):
    """Stage 1 (alignment): train ONLY the projector on LAION-CC-SBU-558K; LLM + SigLIP2 frozen."""
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from torch.utils.data import DataLoader
    from vision import SiglipVision
    from multimodal import MoEVLM
    from generate import load_model
    from vlm_data import LlavaPretrain, collate

    dev = torch.device("cuda")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    enc = SiglipVision(device=dev)
    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    vlm.set_llm_trainable(False)                      # stage 1: projector only
    n_proj = sum(p.numel() for p in vlm.projector_parameters())
    print(f"backbone d{cfg.d_model} val={vloss:.3f} | SigLIP2 hidden={enc.hidden} | projector params={n_proj/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(vlm.projector_parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    ds = LlavaPretrain("/llava/blip_laion_cc_sbu_558k.json", "/llava/images")
    dl = DataLoader(ds, batch_size=micro, shuffle=True, num_workers=8, collate_fn=collate,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    print(f"dataset={len(ds)} samples | micro={micro} | max_steps={max_steps}", flush=True)

    vlm.train()
    step, t0, run = 0, time.time(), 0.0
    done = False
    while not done:
        for imgs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = enc.encode_pixels(enc.preprocess(imgs))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = vlm(ids, image_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector_parameters(), 1.0)
            opt.step()
            run += loss.item()
            if step % log_every == 0:
                dt = (time.time() - t0) / (step + 1)
                print(f"step {step:5d} | loss {loss.item():.4f} | avg {run/(step+1):.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {dt*1000:.0f}ms/step", flush=True)
            step += 1
            if step >= max_steps:
                done = True
                break

    out = f"/data/runs/{save_name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"projector": vlm.mm_projector.state_dict(), "vision_dim": enc.hidden,
                "backbone": BACKBONE, "steps": step}, f"{out}/projector.pt")
    runs_vol.commit()
    print(f"saved projector -> {out}/projector.pt  (final loss {loss.item():.4f})", flush=True)
    return {"final_loss": loss.item(), "steps": step}


@app.local_entrypoint()
def main(action: str = "smoke", max_steps: int = 2000, micro: int = 24, lr: float = 1e-3):
    if action == "download":
        download.remote()
    elif action == "smoke":
        smoke.remote()
    elif action == "stage1":
        train_stage1.remote(max_steps=max_steps, micro=micro, lr=lr)
    else:
        raise SystemExit(f"unknown action {action!r} (use download|smoke|stage1)")
