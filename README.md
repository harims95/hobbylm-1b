# HobbyLM-1B

A 1B-parameter sparse MoE language model project trained on a curated 100B-token mix.

This repository is built on Harish's HobbyLM codebase:
https://github.com/harishsg993010/HobbyLM

## Status

The 100B-token data recipe is built and verified. The Vast training pipeline is ready. The flagship 1B run is pending the final Vast shakedown and launch gate.

## Credits

- Base architecture, MoE trainer, Muon setup, and Rust inference engine: Harish (`harishsg993010`)
- Curated 100B data recipe, loader shuffle fix, Vast training pipeline, and 1B scaling: us
- Testing and validation: Prajan

## Validation

The 130M validation run on the curated mix scored about `43.3` average on the 7-task eval suite, versus the `42.97` FineWeb baseline, while using half the tokens. That result green-lit the data recipe for the 1B flagship.

## Data Recipe

Tokenizer: GPT-2, padded to vocab size `50,304`.

Main phase:

- `60%` FineWeb-Edu from byte-verified Karpathy GPT-2 shards
- `15%` DCLM
- `10%` code
- `10%` FineMath

Anneal phase:

- `5%` anneal slice

The Vast staging script downloads exactly `600` FineWeb-Edu shards, plus the `400` built shards from `harims95/hobbylm-mix100b-gpt2`, and stages the original FineWeb validation shard.

## Training Plan

Target hardware: `4x H100 SXM` on Vast as the available fallback host class. The scripts keep `nproc_per_node` configurable so the run can switch to `8x H100 SXM` if a suitable single-machine host appears.

Training is two-phase:

- Main phase on `edu + dclm + code + math`, using stratified shard interleaving.
- Anneal phase resuming from the main checkpoint on `anneal_*.bin`.

Core launch path:

```bash
python scripts/launch_vast_train.py
```

The launcher prints the exact `torchrun --standalone --nproc_per_node=4 training/train.py ...` command by default. Add `--run` on the Vast host to execute.

## Vast Pipeline

The Vast conversion is split into small, reviewable helpers:

- `scripts/vast_instance.ps1`: search/create/ssh/destroy Vast instances using the Vast API key from the environment only.
- `scripts/stage_vast_data.py`: stage the 100B data mix and validation shard from Hugging Face.
- `scripts/launch_vast_train.py`: print or run the plain `torchrun` training command.
- `scripts/backup_vast_checkpoints.py`: batch-upload checkpoints with `upload_folder()` to avoid Hugging Face commit-rate limits.

No secrets should be committed. Keep credentials in environment variables or ignored `.env` files only.
