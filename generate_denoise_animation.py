import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from dataset import (PairedH5Dataset, PairedTransform, _build_transform,
                     set_image_shape)
from ddpm_simple import DDPM as SRDDPM
from ddim import DDIM
from eval import (build_model, denormalize, ensure_dir, load_config,
                  prepare_sampler, resolve_device, tensor_to_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reverse-denoising animation frames and GIF.")
    parser.add_argument("--config",
                        type=Path,
                        default=Path("configs/eval.json"),
                        help="Path to evaluation configuration JSON file.")
    parser.add_argument("--lr-image",
                        type=str,
                        help="Guidance sample identifier (filename for folders or index/name in HDF5).")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("SR/animations"),
                        help="Directory to place generated artifacts.")
    parser.add_argument("--gif-name",
                        type=str,
                        help="Optional GIF filename (defaults to lr image stem).")
    parser.add_argument("--fps",
                        type=int,
                        default=10,
                        help="Playback rate for the exported GIF.")
    parser.add_argument("--frame-stride",
                        type=int,
                        default=1,
                        help="Keep one frame every N diffusion steps.")
    parser.add_argument("--max-frames",
                        type=int,
                        help="Optional ceiling on saved frames (stride auto adjusts).")
    parser.add_argument("--seed",
                        type=int,
                        help="Random seed for noise sampling.")
    parser.add_argument("--sampler",
                        choices=["ddpm", "ddim"],
                        help="Override sampler type defined in config.")
    parser.add_argument("--skip-frame-dump",
                        action="store_true",
                        help="Only export GIF, skip individual frame PNGs.")
    parser.add_argument("--device",
                        type=str,
                        help="Override device string from config (e.g. cpu or cuda:0).")
    return parser.parse_args()


def select_lr_image_path(data_cfg: Dict, lr_name: Optional[str]) -> Path:
    lr_root = Path(data_cfg["lr_root"])
    if lr_name:
        lr_path = lr_root / lr_name
        if not lr_path.exists():
            raise FileNotFoundError(f"Could not find LR image '{lr_path}'.")
        return lr_path
    candidates = sorted(
        path for path in lr_root.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not candidates:
        raise FileNotFoundError(f"No images found under '{lr_root}'.")
    return candidates[0]


def resolve_h5_index(dataset: PairedH5Dataset, target: Optional[str]) -> int:
    if target is None:
        return 0
    try:
        index = int(target)
        if 0 <= index < len(dataset):
            return index
    except ValueError:
        pass
    for index in range(len(dataset)):
        _, _, name = dataset[index]
        if name == target:
            return index
    raise ValueError(f"Could not locate sample '{target}' in {dataset.h5_path}.")


def load_lr_tensor_from_config(cfg: Dict[str, Any],
                               lr_name: Optional[str],
                               device: torch.device) -> Tuple[torch.Tensor, str]:
    data_cfg = cfg["data"]
    image_size = data_cfg["image_size"]
    channels = data_cfg["channels"]
    set_image_shape(image_size, channels)

    value_range = data_cfg.get("value_range")
    if value_range is not None:
        value_range = tuple(value_range)

    if data_cfg.get("h5_path"):
        transform = PairedTransform(image_size, channels, augment=False)
        dataset = PairedH5Dataset(
            h5_path=data_cfg["h5_path"],
            lr_key=data_cfg.get("h5_lr_key", "lr"),
            hr_key=data_cfg.get("h5_hr_key", "hr"),
            transform=transform,
            value_range=value_range,
            lr_dataset_name=data_cfg.get("h5_lr_dataset"),
            hr_dataset_name=data_cfg.get("h5_hr_dataset"),
            transpose_lr=data_cfg.get("transpose_lr", False),
            transpose_hr=data_cfg.get("transpose_hr", False),
        )
        index = resolve_h5_index(dataset, lr_name)
        lr_tensor, _, sample_name = dataset[index]
        return lr_tensor.unsqueeze(0).to(device), sample_name

    lr_path = select_lr_image_path(data_cfg, lr_name)
    transform = _build_transform(image_size, channels)
    with Image.open(lr_path) as pil_image:
        lr_tensor = transform(pil_image).unsqueeze(0).to(device)
    return lr_tensor, lr_path.stem


def adjust_stride(total_steps: int,
                  stride: int,
                  max_frames: Optional[int]) -> int:
    stride = max(1, int(stride))
    if not max_frames:
        return stride
    max_frames = max(1, int(max_frames))
    stride = max(stride, (total_steps + max_frames - 1) // max_frames)
    return stride


def iterate_ddpm_frames(
        sampler: SRDDPM,
        net: torch.nn.Module,
        lr_tensor: torch.Tensor,
        sample_shape: Tuple[int, int, int, int],
        simple_var: bool,
        stride: int) -> Iterable[Tuple[int, torch.Tensor]]:
    with torch.inference_mode():
        x = torch.randn(sample_shape, device=lr_tensor.device)
        yield sampler.n_steps, x.detach().cpu()
        for step, t in enumerate(
                tqdm(range(sampler.n_steps - 1, -1, -1),
                     total=sampler.n_steps,
                     desc="DDPM reverse"),
                start=1):
            x = sampler.sample_backward_stepsr(x, t, net, lr_tensor, simple_var)
            if step % stride == 0 or t == 0:
                yield t, x.detach().cpu()


def iterate_ddim_frames(
        sampler: DDIM,
        net: torch.nn.Module,
        lr_tensor: torch.Tensor,
        sampler_params: Dict,
        sample_shape: Tuple[int, int, int, int],
        stride: int) -> Iterable[Tuple[int, torch.Tensor]]:
    with torch.inference_mode():
        ddim_steps = int(sampler_params.get("ddim_step", 50))
        eta = float(sampler_params.get("eta", 0.0))
        simple_var = bool(sampler_params.get("simple_var", False))
        if simple_var:
            eta = 1.0
        ts = torch.linspace(sampler.n_steps, 0, ddim_steps + 1,
                            dtype=torch.long,
                            device=lr_tensor.device)
        x = torch.randn(sample_shape, device=lr_tensor.device)
        yield int(ts[0].item()), x.detach().cpu()
        batch_size = x.shape[0]
        for step in tqdm(range(1, ddim_steps + 1),
                         total=ddim_steps,
                         desc="DDIM reverse"):
            cur_index = int((ts[step - 1] - 1).item())
            prev_index = int((ts[step] - 1).item())

            ab_cur = sampler.alpha_bars[cur_index]
            if prev_index >= 0:
                ab_prev = sampler.alpha_bars[prev_index]
            else:
                ab_prev = torch.tensor(1.0, device=lr_tensor.device)

            t_tensor = torch.full((batch_size, 1),
                                  cur_index,
                                  dtype=torch.long,
                                  device=lr_tensor.device)
            eps = net(x, t_tensor, lr_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term
            if step % stride == 0 or step == ddim_steps:
                yield prev_index, x.detach().cpu()


def tensors_to_images(frames: Iterable[Tuple[int, torch.Tensor]],
                      sample_index: int,
                      use_jet: bool,
                      threshold: Optional[float] = None,
                      compress_factor: Optional[float] = None,
                      tail_frames: int = 0) -> List[Tuple[int, Image.Image]]:
    frame_list = list(frames)
    total = len(frame_list)
    images: List[Tuple[int, Image.Image]] = []
    for idx, (t_step, tensor) in enumerate(frame_list):
        if sample_index >= tensor.shape[0]:
            raise IndexError(
                f"Sample index {sample_index} out of range for tensor batch "
                f"of size {tensor.shape[0]}.")
        original = denormalize(tensor[sample_index])
        frame_tensor = original
        if tail_frames > 0 and idx >= total - tail_frames:
            processed = original
            if threshold is not None:
                processed = torch.where(
                    processed < threshold,
                    torch.zeros_like(processed),
                    processed)
            if compress_factor is not None:
                processed = processed * compress_factor
            processed = processed.clamp(0.0, 1.0)
            if tail_frames == 1:
                blend = 1.0
            else:
                blend = float(idx - (total - tail_frames)) / float(tail_frames - 1)
            frame_tensor = torch.lerp(original, processed, blend)
        images.append((t_step, tensor_to_image(frame_tensor, apply_jet=use_jet)))
    return images


def save_animation(images: List[Tuple[int, Image.Image]],
                   output_dir: Path,
                   gif_name: str,
                   fps: int,
                   dump_frames: bool,
                   tail_hold: int = 10) -> Path:
    ensure_dir(output_dir)
    gif_path = output_dir / gif_name
    if dump_frames:
        frames_dir = output_dir / (gif_path.stem + "_frames")
        ensure_dir(frames_dir)
        for idx, (t_step, img) in enumerate(images):
            img.convert("RGB").save(frames_dir / f"{idx:04d}_t{t_step:04d}.png")
    duration = max(1, int(1000 / max(1, fps)))
    pil_frames = [img.convert("RGB") for _, img in images]
    if not pil_frames:
        raise RuntimeError("No frames generated for animation.")
    total_frames = pil_frames.copy()
    for _ in range(max(0, tail_hold)):
        total_frames.append(pil_frames[-1])
    pil_frames[0].save(gif_path,
                       save_all=True,
                       append_images=total_frames[1:],
                       duration=duration,
                       loop=0)
    return gif_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.sampler:
        cfg.setdefault("sampler", {})
        cfg["sampler"]["type"] = args.sampler

    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")

    # 必须在 build_model 之前设置图像尺寸
    data_cfg = cfg["data"]
    set_image_shape(data_cfg["image_size"], data_cfg["channels"])

    net = build_model(cfg, device)
    sampler_type, sampler, sampler_params = prepare_sampler(cfg, device)
    sampler_type = sampler_type.lower()
    print(f"Sampler: {sampler_type.upper()}")

    lr_tensor, lr_label = load_lr_tensor_from_config(cfg, args.lr_image, device)
    print(f"Guidance LR sample: {lr_label}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    else:
        print("Warning: No random seed specified, results may not be reproducible.")

    sample_shape = (1,
                    data_cfg["channels"],
                    data_cfg["image_size"],
                    data_cfg["image_size"])
    use_jet = data_cfg.get("channels", 3) == 1

    if sampler_type == "ddpm":
        simple_var = bool(sampler_params.get("simple_var", True))
        stride = adjust_stride(sampler.n_steps, args.frame_stride, args.max_frames)
        frames = iterate_ddpm_frames(sampler, net, lr_tensor, sample_shape,
                                     simple_var, stride)
    elif sampler_type == "ddim":
        stride = adjust_stride(int(sampler_params.get("ddim_step", 50)),
                               args.frame_stride, args.max_frames)
        frames = iterate_ddim_frames(sampler, net, lr_tensor, sampler_params,
                                     sample_shape, stride)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")

    frame_sequence = list(frames)
    threshold = cfg.get("sampler", {}).get("threshold", 0.95)
    compress_factor = 227.0 / 253.0

    images = tensors_to_images(frame_sequence,
                               sample_index=0,
                               use_jet=use_jet,
                               threshold=threshold,
                               compress_factor=compress_factor,
                               tail_frames=10)

    output_dir = args.output_dir
    safe_label = lr_label.replace(os.sep, "_").replace("/", "_")
    default_stem = Path(safe_label).stem or "lr_sample"
    gif_name = args.gif_name or f"{default_stem}_{sampler_type}.gif"
    gif_path = save_animation(images,
                              output_dir,
                              gif_name,
                              fps=args.fps,
                              dump_frames=not args.skip_frame_dump,
                              tail_hold=10)
    print(f"Saved animation to: {gif_path}")


if __name__ == "__main__":
    main()
    # python generate_denoise_animation.py --config configs/eval.json --lr-image 416_1.png --sampler ddim
