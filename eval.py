import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.functional import (peak_signal_noise_ratio,
                                     structural_similarity_index_measure)

from matplotlib import cm

from dataset import get_paired_dataloader, set_image_shape
from ddpm_simple import DDPM
from ddim import DDIM
from network import (build_network, convnet_big_cfg, convnet_medium_cfg,
                     convnet_small_cfg, unet_1_cfg, unet_res_cfg)
from encoder import VAE_Encoder
from decoder import VAE_Decoder

MODEL_REGISTRY: Dict[str, Tuple] = {
    "network": (build_network, {
        "convnet_small": convnet_small_cfg,
        "convnet_medium": convnet_medium_cfg,
        "convnet_big": convnet_big_cfg,
        "unet": unet_1_cfg,
        "unet_res": unet_res_cfg,
    }),
}

DEFAULT_CONFIG_PATH = Path("configs/eval.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    return get_paired_dataloader(
        batch_size=cfg["batch_size"],
        lr_root=cfg["lr_root"],
        hr_root=cfg["hr_root"],
        image_size=cfg["image_size"],
        channels=cfg["channels"],
        num_workers=cfg.get("num_workers", 4),
    )


def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    module_key = cfg["model"].get("module", "network")
    if module_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model module '{module_key}'. "
                       f"Available: {', '.join(MODEL_REGISTRY)}")
    builder, config_map = MODEL_REGISTRY[module_key]
    backbone = cfg["model"]["backbone"]
    if backbone not in config_map:
        raise KeyError(f"Unknown backbone '{backbone}'. "
                       f"Available: {', '.join(config_map)}")
    net_cfg = config_map[backbone].copy()
    diffusion_steps = cfg["model"]["diffusion_steps"]
    model = builder(net_cfg, diffusion_steps).to(device)

    checkpoint = torch.load(cfg["model"]["checkpoint_path"],
                            map_location=device)
    state_dict = checkpoint.get("ema_model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_vae(
        cfg: Dict[str, Any],
        device: torch.device) -> Tuple[Optional[VAE_Encoder], Optional[VAE_Decoder]]:
    vae_cfg = cfg.get("vae")
    if not vae_cfg or not vae_cfg.get("enabled", True):
        return None, None
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    checkpoint = torch.load(vae_cfg["checkpoint_path"], map_location=device)
    encoder_key = vae_cfg.get("encoder_key", "enc")
    decoder_key = vae_cfg.get("decoder_key", "dec")
    encoder.load_state_dict(checkpoint[encoder_key])
    decoder.load_state_dict(checkpoint[decoder_key])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def prepare_sampler(
        cfg: Dict[str, Any],
        device: torch.device) -> Tuple[str, Union[DDPM, DDIM], Dict[str, Any]]:
    sampler_cfg = cfg["sampler"]
    sampler_type = sampler_cfg.get("type", "ddim").lower()
    steps = cfg["model"]["diffusion_steps"]
    if sampler_type == "ddpm":
        sampler = DDPM(device, steps)
        params = {"simple_var": sampler_cfg.get("simple_var", True)}
    else:
        sampler_type = "ddim"
        sampler = DDIM(device, steps)
        params = {
            "ddim_step": sampler_cfg.get("ddim_steps", 50),
            "eta": sampler_cfg.get("eta", 0.0),
            "simple_var": sampler_cfg.get("simple_var", False),
        }
    return sampler_type, sampler, params


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(-1, 1).add(1).div(2)


def tensor_to_image(tensor: torch.Tensor,
                    apply_jet: bool = False) -> Image.Image:
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    if array.shape[2] == 1 and apply_jet:
        gray_np = array[:, :, 0]
        jet_colored = cm.jet(gray_np)
        jet_colored_rgb = (jet_colored[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(jet_colored_rgb)
    array = (array * 255.0).astype(np.uint8)
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode='L')
    return Image.fromarray(array)


def evaluate(cfg: Dict[str, Any], device: torch.device) -> None:
    data_cfg = cfg["data"]
    set_image_shape(data_cfg["image_size"], data_cfg["channels"])
    dataloader = create_dataloader(cfg["data"])
    net = build_model(cfg, device)
    # encoder, decoder = build_vae(cfg, device)
    sampler_type, sampler, sampler_params = prepare_sampler(cfg, device)
    use_jet = data_cfg.get("channels", 3) == 1

    output_root = Path(cfg["output"]["root"])
    images_dir = output_root / "images"
    ensure_dir(images_dir)
    ensure_dir(output_root)

    psnr_scores = []
    ssim_scores = []
    per_image_results = []

    with torch.inference_mode():
        for lr_images, hr_images, names in tqdm(dataloader, desc="Evaluating"):
            seed = 1234
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            batch_size = lr_images.size(0)
            sample_shape = (batch_size, *hr_images.shape[1:])

            sr_images = sampler.sample_backward_sr(sample_shape,
                                                   net,
                                                   lr_images,
                                                   device=device,
                                                   **sampler_params)

            # if encoder and decoder:
            #     noise_shape = (batch_size, 4, cfg["data"]["image_size"] // 8,
            #                    cfg["data"]["image_size"] // 8)
            #     noise = torch.randn(noise_shape, device=device)
            #     latents, _, _ = encoder(sr_images, noise)
            #     sr_images = decoder(latents)

            sr_for_metric = denormalize(sr_images)
            sr_for_metric = torch.where(sr_for_metric < cfg["sampler"].get("threshold", 0.95),
                                         torch.zeros_like(sr_for_metric),
                                         sr_for_metric)

            # 进行压缩：将数值范围从[0,1]压缩到[0, 227/253]
            sr_for_metric = sr_for_metric * (227.0 / 253.0)

            hr_for_metric = denormalize(hr_images)

            for idx, name in enumerate(names):
                pred = sr_for_metric[idx].unsqueeze(0)
                target = hr_for_metric[idx].unsqueeze(0)
                psnr = peak_signal_noise_ratio(pred, target)
                ssim = structural_similarity_index_measure(
                    pred, target)
                psnr_scores.append(psnr.item())
                ssim_scores.append(ssim.item())
                per_image_results.append({
                    "filename": name,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                })
                if cfg["output"].get("save_images", True):
                    tensor_to_image(sr_for_metric[idx],
                                    apply_jet=use_jet).save(images_dir / name)

    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    print(f"平均 PSNR: {avg_psnr:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f}")

    results_txt = output_root / "results.txt"
    with results_txt.open("w", encoding="utf-8") as handle:
        handle.write("Evaluation Summary\n")
        handle.write("===================\n")
        handle.write(f"Model: {cfg['model']['checkpoint_path']}\n")
        handle.write(f"Dataset: {cfg['data']['lr_root']}\n")
        handle.write(f"Sampler: {sampler_type.upper()}\n")
        handle.write(f"Average PSNR: {avg_psnr:.4f}\n")
        handle.write(f"Average SSIM: {avg_ssim:.4f}\n")

    results_csv = output_root / "results_per_image.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(per_image_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SR diffusion model.")
    parser.add_argument("--config",
                        type=Path,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to evaluation configuration JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")
    evaluate(cfg, device)


if __name__ == "__main__":
    main()
