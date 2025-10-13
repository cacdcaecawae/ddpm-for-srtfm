# Repository Guidelines

## Project Structure & Module Organization
The codebase is intentionally flat. Diffusion schedulers live in `ddpm.py`, `ddpm_simple.py`, and the newly added `ddim.py`. Networks are defined in `network.py`, which now contains the time-aware UNet and ConvNet variants. Super-resolution training is driven exclusively by `SR_train.py`, backed by `dataset.py`, `encoder.py`, `decoder.py`, `attention.py`, and utility helpers like `noise.py`. Configuration files now live in `configs/`, and evaluation utilities reside in `eval.py`. Generated assets land in `SR/`, `work_dirs/`, or a user-defined preview folder.

## Build, Test, and Development Commands
Super-resolution training is launched via `python SR_train.py --config configs/sr_train.json`. Adjust dataset roots and hyperparameters inside that JSON before running; avoid in-code edits. Evaluate checkpoints with `python eval.py --config configs/eval.json`, which writes metrics and per-image outputs under the configured directory. TensorBoard support remains available through `tensorboard --logdir runs`. The legacy MNIST demos have been removed—focus development on the SR pipeline.

## Coding Style & Naming Conventions
Follow PEP 8 (four-space indentation, snake_case for functions/variables). Keep config loaders and CLI entry points lightweight, leverage helpers from `dataset.py` and `ddpm_simple.py`, and prefer pure functions for shared logic. Provide concise type hints for public APIs (for example, `cfg: Dict[str, Any]`, `device: torch.device`) and add short comments when tensor shapes or units are not obvious. Maintain import order compatible with `isort`/`black`; no trailing whitespace and stick to ASCII unless the surrounding file already uses UTF-8 text.

## Testing Guidelines
A lightweight pytest suite now lives in `tests/`. Run `pytest` before merging to ensure scheduler smoke tests and shape checks pass. For end-to-end validation, execute `python eval.py --config configs/eval.json` and capture PSNR/SSIM from the generated CSV. When adding new utilities, include shape assertions or unit tests to avoid silent regressions.

## Commit & Pull Request Guidelines
Use imperative present-tense commit subjects around 60 characters (for example, `feat: add JSON configs for SR training`). Reference related experiments or issues in the body, and list commands executed (training, evaluation, pytest). Pull requests should summarise motivation, highlight key files touched, outline testing status, and attach relevant outputs (metrics table or preview image path) before requesting review.

## Data & Configuration Notes
Training assumes CUDA by default; set `"device": "cpu"` and `"num_workers": 0` in configs for CPU-only runs. Always duplicate model config templates with `.copy()` before modifying them to avoid shared-state bugs. Store checkpoints under `SR/` (tracked via JSON configs) and keep large binaries ignored. All imports are now local—there is no `dldemos` namespace requirement—so ensure scripts execute from the repository root or add the project directory to `PYTHONPATH` when launching notebooks.
