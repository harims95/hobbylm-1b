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
    """Download LLaVA-Pretrain (LAION-CC-SBU-558K): captions json + images.zip (~24GB), kept as ONE file
    on the volume and STREAMED (random-access by name) at train time — no extracting 558K files."""
    import os, zipfile
    from huggingface_hub import hf_hub_download
    repo = "liuhaotian/LLaVA-Pretrain"
    os.makedirs("/llava", exist_ok=True)
    if not os.path.exists("/llava/blip_laion_cc_sbu_558k.json"):
        hf_hub_download(repo, "blip_laion_cc_sbu_558k.json", repo_type="dataset", local_dir="/llava")
        data_vol.commit()
        print("got captions json", flush=True)
    if not os.path.exists("/llava/images.zip"):
        print("downloading images.zip (~24GB) -> volume (no unzip)...", flush=True)
        hf_hub_download(repo, "images.zip", repo_type="dataset", local_dir="/llava")
        data_vol.commit()
    # sanity: open the zip and confirm the first json image is readable by name
    import json
    data = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    with zipfile.ZipFile("/llava/images.zip") as z:
        names = z.namelist()
        first = data[0]["image"]
        ok = first in z.NameToInfo
    sz = os.path.getsize("/llava/images.zip") / 1e9
    print(f"ready: images.zip ({sz:.1f}GB, {len(names)} entries) + {len(data)} captions | "
          f"first image '{first}' in zip: {ok}", flush=True)


@app.function(image=vlm_image, volumes={"/llava": data_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def download_sft():
    """Stage-2 data: LLaVA-Instruct-150K (the GPT-4 visual-instruction set) + COCO train2017 images
    (~18GB, streamed from the zip like stage 1). Single most effective subset of the 665K mix."""
    import os, json, zipfile, urllib.request
    from huggingface_hub import hf_hub_download
    if not os.path.exists("/llava/llava_instruct_150k.json"):
        hf_hub_download("liuhaotian/LLaVA-Instruct-150K", "llava_instruct_150k.json",
                        repo_type="dataset", local_dir="/llava")
        data_vol.commit()
        print("got llava_instruct_150k.json", flush=True)
    if not os.path.exists("/llava/train2017.zip"):
        print("downloading COCO train2017 (~18GB)...", flush=True)
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/train2017.zip", "/llava/train2017.zip")
        data_vol.commit()
    data = json.load(open("/llava/llava_instruct_150k.json"))
    with zipfile.ZipFile("/llava/train2017.zip") as z:
        names = z.NameToInfo
    hit = sum(1 for ex in data[:300] if f"train2017/{ex['image']}" in names)
    sz = os.path.getsize("/llava/train2017.zip") / 1e9
    print(f"ready: {len(data)} instructions | COCO train2017.zip {sz:.1f}GB ({len(names)} entries) | "
          f"hit-rate first 300: {hit}/300", flush=True)


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


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_stage1(max_steps: int = 1500, micro: int = 32, lr: float = 1e-3, save_name: str = "500M_vlm_stage1"):
    """Stage 1 (alignment) on 8x H100 via torchrun: projector only; LLM + SigLIP2 frozen."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_stage1.py",
           "--backbone", BACKBONE, "--json", "/llava/blip_laion_cc_sbu_558k.json",
           "--zip", "/llava/images.zip", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_stage2(max_steps: int = 1200, micro: int = 16, lr: float = 2e-5, save_name: str = "500M_vlm_stage2",
                 stage1_run: str = "500M_vlm_stage1"):
    """Stage 2 (SFT) on 8x H100 via torchrun: instruction-tune projector + LLM on LLaVA-Instruct-150K."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_stage2.py",
           "--backbone", BACKBONE, "--stage1", f"/data/runs/{stage1_run}/projector.pt",
           "--json", "/llava/llava_instruct_150k.json", "--zip", "/llava/train2017.zip", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def caption(stage1_run: str = "500M_vlm_stage1", n: int = 8, max_new: int = 32, prompt: str = "",
            stage2_run: str = ""):
    """Greedy-caption a few real LAION images with the stage-1 VLM; print predicted vs ground-truth."""
    import os, sys, io, json, zipfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = SiglipVision(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    if stage2_run:
        ck = torch.load(f"/data/runs/{stage2_run}/model.pt", map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(ck["model"])              # stage-2 finetuned LLM
        vlm.mm_projector.load_state_dict(ck["projector"])
        print(f"loaded stage-2 VLM from {stage2_run}", flush=True)
    else:
        ck = torch.load(f"/data/runs/{stage1_run}/projector.pt", map_location=dev, weights_only=False)
        vlm.mm_projector.load_state_dict(ck["projector"])
        print(f"loaded stage-1 projector (steps={ck.get('steps')})", flush=True)
    vlm.eval()

    tok = tiktoken.get_encoding("gpt2")
    data = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    z = zipfile.ZipFile("/llava/images.zip")
    pre = tok.encode_ordinary(prompt) if prompt else []

    @torch.no_grad()
    def gen(image):
        feats = enc.encode([image])
        ids = torch.tensor([[IMAGE_TOKEN] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, image_features=feats)
            outs = []
            for _ in range(max_new):
                logits, _ = vlm.llm(inputs_embeds=cur)
                lg = logits[:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                e = vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)
                cur = torch.cat([cur, e], dim=1)
        return tok.decode(outs)

    for k in range(n):
        i = (k * 69779) % len(data)                    # deterministic spread across the set
        ex = data[i]
        img = Image.open(io.BytesIO(z.read(ex["image"]))).convert("RGB")
        gt = ex["conversations"][1]["value"].strip().replace("\n", " ")[:90]
        print(f"\n[{ex['image']}]\n   GT:   {gt}\n   PRED: {gen(img).strip()}", flush=True)
    print("\nCAPTION DONE", flush=True)


@app.local_entrypoint()
def main(action: str = "smoke", max_steps: int = 1500, micro: int = 32, lr: float = 1e-3, n: int = 8,
         stage2_run: str = ""):
    if action == "download":
        download.remote()
    elif action == "download_sft":
        download_sft.remote()
    elif action == "smoke":
        smoke.remote()
    elif action == "stage1":
        train_stage1.remote(max_steps=max_steps, micro=micro, lr=lr)
    elif action == "stage2":
        train_stage2.remote(max_steps=max_steps, micro=micro, lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "caption":
        caption.remote(n=n, stage2_run=stage2_run)
    else:
        raise SystemExit(f"unknown action {action!r} "
                         "(use download|download_sft|smoke|stage1|stage2|caption)")
