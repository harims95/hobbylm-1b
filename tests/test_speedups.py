"""CPU correctness checks for the nanogpt-inspired throughput optimizations.

Run: python test_speedups.py

Validates (on CPU, bmm backend):
  1. fused_ce produces a loss numerically identical to the baseline CE path (+ finite grads).
  2. fp8_head builds, runs fwd/bwd (bf16 fallback on CPU), and matches the param-count bump.
  3. Polar Express orthogonalizer steps cleanly and stays finite.
The fp8 GEMM and grouped_mm speedups themselves are GPU-only and validated on Modal.
"""
import torch

from hobbylm.config import get_config, TrainConfig
from hobbylm.model import MoETransformer, count_params
from hobbylm.optim import build_optimizers, newton_schulz


def _tiny_cfg(**over):
    cfg = get_config("130M")
    cfg.expert_backend = "bmm"
    cfg.n_layers, cfg.n_experts, cfg.d_model = 4, 8, 128
    cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim = 4, 2, 32
    cfg.dense_ffn, cfg.expert_ffn = 256, 64
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    return MoETransformer(cfg)


def test_fused_ce_equivalence():
    print("[1] fused_ce numerical equivalence vs baseline")
    torch.manual_seed(7)
    idx = torch.randint(0, 50257, (2, 64))
    tgt = torch.randint(0, 50257, (2, 64))

    base = _build(_tiny_cfg(fused_ce=False), seed=0)
    fused = _build(_tiny_cfg(fused_ce=True, ce_chunk=37), seed=0)  # odd chunk to stress edges
    fused.load_state_dict(base.state_dict())  # identical weights

    lb, pb = base(idx, tgt)
    lf, pf = fused(idx, tgt)
    lb.backward(); lf.backward()

    dloss = abs(lb.item() - lf.item())
    # compare a shared gradient (embedding) between the two paths
    gb = base.embed.weight.grad
    gf = fused.embed.weight.grad
    dgrad = (gb - gf).abs().max().item()
    print(f"    baseline loss={lb.item():.6f}  fused loss={lf.item():.6f}  |dloss|={dloss:.2e}")
    print(f"    max |grad diff| on embed = {dgrad:.2e}")
    assert dloss < 1e-4, f"loss mismatch {dloss}"
    assert dgrad < 1e-3, f"grad mismatch {dgrad}"
    assert torch.isfinite(lf), "non-finite fused loss"
    print("    OK")


def test_fp8_head_builds():
    print("[2] fp8_head build + fwd/bwd (CPU bf16 fallback) + param bump")
    idx = torch.randint(0, 50257, (2, 32))
    tgt = torch.randint(0, 50257, (2, 32))
    tied = _build(_tiny_cfg(fp8_head=False), seed=1)
    untied = _build(_tiny_cfg(fp8_head=True), seed=1)
    ct, cu = count_params(tied), count_params(untied)
    extra = cu["total"] - ct["total"]
    expect = untied.cfg.vocab_size * untied.cfg.d_model
    print(f"    tied total={ct['total']/1e6:.3f}M  fp8/untied total={cu['total']/1e6:.3f}M  "
          f"(+{extra/1e6:.3f}M, expect +{expect/1e6:.3f}M)")
    assert extra == expect, f"unexpected param delta {extra} vs {expect}"
    loss, _ = untied(idx, tgt)
    loss.backward()
    g = untied.lm_head.weight.grad
    print(f"    fp8 head loss={loss.item():.4f}  head.grad finite={bool(torch.isfinite(g).all())}")
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    print("    OK")


def test_polar_express():
    print("[3] Polar Express orthogonalizer")
    torch.manual_seed(3)
    G = torch.randn(64, 48)
    o_ns = newton_schulz(G, steps=5, schedule="ns5").float()
    o_pe = newton_schulz(G, schedule="polar").float()
    # both should be ~orthogonal: singular values near 1
    s_ns = torch.linalg.svdvals(o_ns)
    s_pe = torch.linalg.svdvals(o_pe)
    print(f"    ns5  singular values in [{s_ns.min():.3f}, {s_ns.max():.3f}]")
    print(f"    polar singular values in [{s_pe.min():.3f}, {s_pe.max():.3f}]")
    assert torch.isfinite(o_pe).all()
    assert s_pe.min() > 0.7 and s_pe.max() < 1.3, "polar not orthogonalizing"

    # full optimizer step with polar schedule
    cfg = _tiny_cfg()
    model = _build(cfg, seed=2)
    tc = TrainConfig(orthogonalizer="polar")
    muon, adamw, _ = build_optimizers(model, tc)
    idx = torch.randint(0, 50257, (2, 32)); tgt = torch.randint(0, 50257, (2, 32))
    loss, _ = model(idx, tgt); loss.backward()
    muon.step(); adamw.step()
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    print(f"    post-step params finite={finite}")
    assert finite
    print("    OK")


if __name__ == "__main__":
    test_fused_ce_equivalence()
    test_fp8_head_builds()
    test_polar_express()
    print("\nALL CPU CHECKS PASSED")
