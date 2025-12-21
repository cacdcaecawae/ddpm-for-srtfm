import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchmetrics.functional import peak_signal_noise_ratio
from tqdm import tqdm

from dataset import get_h5_dataloader
from ddpm_simple import DDPM
from ddim import DDIM
from network import (build_network, convnet_big_cfg, convnet_medium_cfg,
                     convnet_small_cfg, dit_base_cfg, unet_1_cfg,
                     unet_res_cfg)


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnet_small": convnet_small_cfg,
    "convnet_medium": convnet_medium_cfg,
    "convnet_big": convnet_big_cfg,
    "unet": unet_1_cfg,
    "unet_res": unet_res_cfg,
    "dit_base": dit_base_cfg,
}

DEFAULT_CONFIG_PATH = Path("configs/sr_train.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_replace(src: Path, dst: Path, retries: int = 5, delay: float = 0.1) -> None:
    """Work around Windows file locking when replacing checkpoints."""
    last_error: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    time.sleep(delay)
            time.sleep(delay)
    raise last_error if last_error else PermissionError(f"Failed to replace {dst}")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None:
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for key, value in model_sd.items():
        ema_sd[key].mul_(decay).add_(value, alpha=1 - decay)
    ema_model.load_state_dict(ema_sd)


def warmup_cosine(optimizer: torch.optim.Optimizer,
                  current_epoch: int,
                  max_epoch: int,
                  lr_min: float = 0.0,
                  lr_max: float = 1e-4,
                  warmup_epoch: int = 10) -> None:
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_jet(x: torch.Tensor,
           vmin: Optional[float] = None,
           vmax: Optional[float] = None,
           bins: int = 256) -> torch.Tensor:
    """Map a batch of 1-channel tensors to RGB using a jet colormap."""
    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        squeeze_back = True
    elif x.dim() == 3:
        x = x.unsqueeze(0)

    device = x.device
    if vmin is None:
        vmin = float(x.min().item())
    if vmax is None:
        vmax = float(x.max().item())
    if vmax == vmin:
        vmax = vmin + 1e-6

    x_norm = (x - vmin) / (vmax - vmin)
    x_norm = x_norm.clamp(0, 1)

    # 修复 matplotlib 弃用警告
    try:
        # matplotlib >= 3.7
        cmap = cm.get_cmap('jet')
        lut_np = cmap(np.linspace(0, 1, bins))[:, :3]
    except AttributeError:
        # matplotlib < 3.7
        lut_np = cm.get_cmap('jet', bins)(np.linspace(0, 1, bins))[:, :3]
    
    lut = torch.from_numpy(lut_np).to(device=device, dtype=torch.float32)

    # 处理单通道: [B, 1, H, W]
    idx = (x_norm * (bins - 1)).round().long()  # [B, 1, H, W]
    idx = idx.squeeze(1)  # [B, H, W]
    rgb = lut[idx]  # [B, H, W, 3]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

    if squeeze_back:
        rgb = rgb.squeeze(0)
    return rgb


def make_preview_grid(tensor: torch.Tensor, channels: int,
                      nrow: int) -> torch.Tensor:
    """
    创建预览网格图像
    
    Args:
        tensor: [B, C, H, W] - C=1 单通道用jet着色, C=3 直接使用
        channels: 通道数
        nrow: 网格行数
    """
    if channels == 1:
        # 单通道: 转为 RGB jet colormap
        tensor = to_jet(tensor, vmin=0.0, vmax=1.0)
    elif channels == 3:
        # 三通道: 只显示第一个通道 (I通道)
        tensor = to_jet(tensor[:, 0:1], vmin=0.0, vmax=1.0)
    return make_grid(tensor, nrow=nrow)


def save_grid_image(tensor: torch.Tensor, output_path: Path) -> None:
    array = (tensor.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    if array.shape[2] == 1:
        image = Image.fromarray(array[:, :, 0], mode='L')
    else:
        image = Image.fromarray(array, mode='RGB')
    image.save(output_path)


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    backbone_key = model_cfg["backbone"]
    if backbone_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown backbone '{backbone_key}'. "
                       f"Available: {', '.join(MODEL_CONFIGS)}")
    net_cfg = MODEL_CONFIGS[backbone_key].copy()
    n_steps = model_cfg["diffusion_steps"]
    
    # 获取输入通道数和图像尺寸
    in_channels = data_cfg["channels"]  # HR 的通道数
    image_size = data_cfg["image_size"]
    
    # LR 的通道数
    lr_channels = in_channels
    if data_cfg.get("use_tfm_channels", False):
        lr_channels = 3  # TFM 模式: I, X, Y 三通道
    
    net = build_network(net_cfg, n_steps, in_channels, image_size, lr_channels).to(device)
    return net


def maybe_load_checkpoint(net: nn.Module, cfg: Dict[str, Any],
                          device: torch.device) -> None:
    resume_path = cfg["model"].get("resume_checkpoint")
    if not resume_path:
        return
    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = checkpoint.get("ema_model_state_dict",
                                checkpoint.get("model_state_dict", checkpoint))
    net.load_state_dict(state_dict)
    print(f"Loaded weights from {resume_path}")


def get_image_shape_from_config(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """从配置中获取图像形状 (channels, height, width)"""
    data_cfg = cfg["data"]
    channels = data_cfg["channels"]
    image_size = data_cfg["image_size"]
    return (channels, image_size, image_size)


def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader:
    data_cfg = cfg["data"]
    
    # 构建 coord_range 参数
    coord_range = None
    if data_cfg.get("use_tfm_channels", False):
        coord_range_x = tuple(data_cfg["coord_range_x"]) if "coord_range_x" in data_cfg else (-1.0, 1.0)
        coord_range_y = tuple(data_cfg["coord_range_y"]) if "coord_range_y" in data_cfg else (-1.0, 1.0)
        coord_range = (coord_range_x, coord_range_y)
    
    return get_h5_dataloader(
        h5_path=data_cfg["h5_path"],
        batch_size=data_cfg["batch_size"],
        lr_key=data_cfg.get("h5_lr_key", "TFM"),
        hr_key=data_cfg.get("h5_hr_key", "hr"),
        lr_dataset_name=data_cfg.get("h5_lr_dataset"),
        hr_dataset_name=data_cfg.get("h5_hr_dataset"),
        transpose_lr=data_cfg.get("transpose_lr", False),
        transpose_hr=data_cfg.get("transpose_hr", False),
        use_tfm_channels=data_cfg.get("use_tfm_channels", False),
        coord_range=coord_range,
        augment=data_cfg.get("augment", False),
        h_flip_prob=data_cfg.get("h_flip_prob", 0.5),
        translate_prob=data_cfg.get("translate_prob", 0.5),
        max_translate_ratio=data_cfg.get("max_translate_ratio", 0.05),
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=True,
    )



def train(ddpm: DDPM,
          net: nn.Module,
          cfg: Dict[str, Any],
          device: torch.device,
          ckpt_path: Path,
          log_dir: Path) -> nn.Module:
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    logging_cfg = cfg["logging"]

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Start training, batch size: {data_cfg['batch_size']}, "
          f"epochs: {opt_cfg['epochs']}")

    dataloader = create_dataloader(cfg)
    net = net.to(device).train()
    ema_net = deepcopy(net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=opt_cfg["learning_rate"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.99))),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    best_loss = float('inf')
    best_loss_epoch = -1
    model_best_state_dict = None
    ema_model_best_state_dict = None

    best_psnr = -float('inf')
    best_psnr_epoch = -1
    model_best_state_dict_by_psnr = None
    ema_model_best_state_dict_by_psnr = None

    use_amp = bool(opt_cfg.get("amp", device.type == 'cuda'))
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    print(f"Using AMP: {use_amp}")

    epochs = opt_cfg["epochs"]
    warmup_epochs = opt_cfg.get("warmup_epochs", max(1, epochs // 10))
    lr_min = opt_cfg.get("lr_min", optimizer.param_groups[0]["lr"])
    lr_max = opt_cfg.get("lr_max", optimizer.param_groups[0]["lr"])
    preview_interval = logging_cfg.get("sample_interval", 10)
    preview_count = max(1, logging_cfg.get("num_preview_samples", 4))
    preview_nrow = int(preview_count**0.5)
    preview_nrow = max(1, preview_nrow)

    tic = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        warmup_cosine(optimizer,
                      epoch,
                      epochs - 1,
                      lr_min=lr_min,
                      lr_max=lr_max,
                      warmup_epoch=warmup_epochs)

        for lr_images, hr_images, _ in pbar:
            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)
            batch_size = hr_images.size(0)

            t = torch.randint(0,
                              ddpm.n_steps, (batch_size,),
                              device=device,
                              dtype=torch.long)
            eps = torch.randn_like(hr_images)
            x_t = ddpm.sample_forward(hr_images, t, eps)

            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.float16,
                                    enabled=use_amp
                                    and device.type == "cuda"):
                eps_theta = net(x_t, t, lr_images)
                loss = loss_fn(eps_theta, eps)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            ema_update(ema_net, net, decay=opt_cfg["ema_decay"])
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_psnr = None
        if (epoch + 0) % preview_interval == 0:
            net_was_training = net.training
            net.eval()
            ema_net.eval()
            with torch.inference_mode():
                preview_batch = min(preview_count, lr_images.size(0))
                lr_subset = lr_images[:preview_batch]
                img_shape = get_image_shape_from_config(cfg)
                img_net = ddpm.sample_backward_sr((preview_batch, *img_shape),
                                                  net,
                                                  lr_subset,
                                                  device=device,
                                                  simple_var=True).cpu()
                img_ema = ddpm.sample_backward_sr((preview_batch, *img_shape),
                                                  ema_net,
                                                  lr_subset,
                                                  device=device,
                                                  simple_var=True).cpu()
            if net_was_training:
                net.train()

            hr_subset = hr_images[:preview_batch].cpu()
            sr01 = ((img_ema + 1) / 2).clamp(0, 1)
            hr01 = ((hr_subset + 1) / 2).clamp(0, 1)

            psnr_scores = []
            for idx in range(sr01.size(0)):
                psnr_val = peak_signal_noise_ratio(sr01[idx].unsqueeze(0),
                                                   hr01[idx].unsqueeze(0),
                                                   data_range=1.0)
                psnr_scores.append(psnr_val.item())
            if psnr_scores:
                mean_psnr = float(sum(psnr_scores) / len(psnr_scores))

            lr01 = ((lr_subset.detach().cpu().clamp(-1, 1) + 1) / 2)
            hr01_display = ((hr_subset.clamp(-1, 1) + 1) / 2)
            net01 = ((img_net.detach().cpu().clamp(-1, 1) + 1) / 2)
            ema01 = ((img_ema.detach().cpu().clamp(-1, 1) + 1) / 2)
            channels = lr01.shape[1]
            writer.add_image(f'sample/epoch_{epoch + 1}_lr',
                             make_preview_grid(lr01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_hr',
                             make_preview_grid(hr01_display, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_net',
                             make_preview_grid(net01, channels, preview_nrow),
                             epoch + 1)
            writer.add_image(f'sample/epoch_{epoch + 1}_ema',
                             make_preview_grid(ema01, channels, preview_nrow),
                             epoch + 1)

            if mean_psnr is not None and mean_psnr > best_psnr:
                best_psnr = mean_psnr
                best_psnr_epoch = epoch + 1
                model_best_state_dict_by_psnr = deepcopy(net.state_dict())
                ema_model_best_state_dict_by_psnr = deepcopy(
                    ema_net.state_dict())

        avg_loss = total_loss / len(dataloader.dataset)
        writer.add_scalar('train/loss', avg_loss, epoch + 1)
        if mean_psnr is not None:
            writer.add_scalar('val/psnr_mean', mean_psnr, epoch + 1)

        toc = time.time()
        print(
            f"Epoch {epoch + 1}/{epochs} finished. "
            f"Average loss: {avg_loss:.6f}. "
            f"Elapsed: {(toc - tic):.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_loss_epoch = epoch + 1
            model_best_state_dict = deepcopy(net.state_dict())
            ema_model_best_state_dict = deepcopy(ema_net.state_dict())

            ckpt_best = {
                'epoch': best_loss_epoch,
                'ema_decay': opt_cfg["ema_decay"],
                'best_loss': best_loss,
                'model_best_state_dict': model_best_state_dict,
                'ema_model_best_state_dict': ema_model_best_state_dict,
            }
            best_path = ckpt_path.with_name(f"{ckpt_path.stem}_best.pth")
            tmp_best = best_path.with_suffix(best_path.suffix + ".tmp")
            torch.save(ckpt_best, tmp_best)
            safe_replace(tmp_best, best_path)

        ckpt = {
            'epoch': epoch + 1,
            'ema_decay': opt_cfg["ema_decay"],
            'model_state_dict': net.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
            'best_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'model_best_state_dict': model_best_state_dict,
            'ema_model_best_state_dict': ema_model_best_state_dict,
            'best_psnr': best_psnr,
            'best_psnr_epoch': best_psnr_epoch,
            'model_best_state_dict_by_psnr': model_best_state_dict_by_psnr,
            'ema_model_best_state_dict_by_psnr':
            ema_model_best_state_dict_by_psnr,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(ckpt, tmp_ckpt)
        safe_replace(tmp_ckpt, ckpt_path)

    writer.close()
    print("Done training!")
    return ema_net


def sample_imgs(ddpm: DDPM,
                net: nn.Module,
                lr_images: torch.Tensor,
                output_path: Path,
                device: torch.device,
                nrow: int,
                cfg: Dict[str, Any],
                simple_var: bool = True) -> None:
    net = net.to(device).eval()
    with torch.no_grad():
        img_shape = get_image_shape_from_config(cfg)
        shape = (lr_images.size(0), *img_shape)
        samples = ddpm.sample_backward_sr(shape,
                                          net,
                                          lr_images.to(device),
                                          device=device,
                                          simple_var=simple_var).cpu()
        samples = ((samples + 1) / 2).clamp(0, 1)
    grid = make_preview_grid(samples, img_shape[0], nrow)
    save_grid_image(grid, output_path)


def sample_imgs_ddim(ddim: DDIM,
                     net: nn.Module,
                     lr_images: torch.Tensor,
                     output_path: Path,
                     device: torch.device,
                     nrow: int,
                     cfg: Dict[str, Any],
                     ddim_step: int = 50,
                     eta: float = 0.0,
                     simple_var: bool = True) -> None:
    net = net.to(device).eval()
    with torch.no_grad():
        img_shape = get_image_shape_from_config(cfg)
        shape = (lr_images.size(0), *img_shape)
        samples = ddim.sample_backward_sr(shape,
                                          net,
                                          lr_images.to(device),
                                          device=device,
                                          simple_var=simple_var,
                                          ddim_step=ddim_step,
                                          eta=eta).cpu()
        samples = ((samples + 1) / 2).clamp(0, 1)
    grid = make_preview_grid(samples, img_shape[0], nrow)
    save_grid_image(grid, output_path)


def collect_preview_batch(cfg: Dict[str, Any],
                          device: torch.device,
                          count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dataloader = create_dataloader(cfg)
    lr_images, hr_images, _ = next(iter(dataloader))
    lr_images = lr_images[:count].to(device)
    hr_images = hr_images[:count].to(device)
    return lr_images, hr_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR diffusion model.")
    parser.add_argument("--config",
                        type=Path,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to training configuration JSON file.")
    parser.add_argument("--mode",
                        choices=["train", "sample"],
                        default="train",
                        help="Run training or sampling only.")
    parser.add_argument("--sampler",
                        choices=["ddpm", "ddim"],
                        default=None,
                        help="Sampler to use in sample mode.")
    parser.add_argument("--checkpoint",
                        type=Path,
                        default=None,
                        help="Override checkpoint path when sampling.")
    parser.add_argument("--ddim-steps",
                        type=int,
                        default=None,
                        help="DDIM steps when sampling.")
    parser.add_argument("--eta",
                        type=float,
                        default=None,
                        help="DDIM ETA when sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    if seed is not None:
        set_seed(seed)
    else:
        print("Seed disabled; results will vary between runs.")

    device = resolve_device(cfg.get("device"))
    print(f"Using device: {device}")

    net = build_model(cfg, device)
    maybe_load_checkpoint(net, cfg, device)

    n_steps = cfg["model"]["diffusion_steps"]
    ddpm = DDPM(device, n_steps)
    ddim = DDIM(device, n_steps)

    ckpt_dir = Path(cfg["model"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / cfg["model"]["checkpoint_name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_cfg = cfg["logging"]
    log_root = Path(log_cfg["tensorboard_root"])
    ensure_dir(log_root)
    log_dir = log_root / f"{timestamp}-{log_cfg.get('experiment_name', 'sr-train')}"
    ensure_dir(log_dir)

    if args.mode == "train":
        ema_model = train(ddpm, net, cfg, device, ckpt_path, log_dir)
        if log_cfg.get("save_preview_images", False):
            preview_dir = Path(log_cfg.get("preview_dir", "SR/previews"))
            ensure_dir(preview_dir)
            preview_count = max(1, log_cfg.get("num_preview_samples", 4))
            nrow = max(1, int(preview_count**0.5))
            lr_batch, _ = collect_preview_batch(cfg, device, preview_count)
            preview_path = preview_dir / f"{timestamp}_ema_ddpm.png"
            sample_imgs(ddpm,
                        ema_model,
                        lr_batch,
                        preview_path,
                        device,
                        nrow,
                        cfg,
                        simple_var=cfg["sampler"].get("simple_var", True))
    else:
        ckpt_to_use = args.checkpoint or cfg["model"]["checkpoint_path"]
        checkpoint = torch.load(ckpt_to_use, map_location=device)
        state_dict = checkpoint.get("ema_model_state_dict",
                                    checkpoint.get("model_state_dict",
                                                   checkpoint))
        net.load_state_dict(state_dict)
        net.eval()

        sampler_type = args.sampler or cfg["sampler"].get("type", "ddim")
        preview_count = max(1, cfg["logging"].get("num_preview_samples", 4))
        nrow = max(1, int(preview_count**0.5))
        lr_batch, _ = collect_preview_batch(cfg, device, preview_count)
        preview_dir = Path(cfg["logging"].get("preview_dir", "SR/previews"))
        ensure_dir(preview_dir)
        if sampler_type == "ddpm":
            output_path = preview_dir / f"{timestamp}_sample_ddpm.png"
            sample_imgs(ddpm,
                        net,
                        lr_batch,
                        output_path,
                        device,
                        nrow,
                        cfg,
                        simple_var=cfg["sampler"].get("simple_var", True))
        else:
            output_path = preview_dir / f"{timestamp}_sample_ddim.png"
            ddim_steps = args.ddim_steps or cfg["sampler"].get("ddim_steps", 50)
            eta = args.eta if args.eta is not None else cfg["sampler"].get(
                "eta", 0.0)
            sample_imgs_ddim(ddim,
                             net,
                             lr_batch,
                             output_path,
                             device,
                             nrow,
                             cfg,
                             ddim_step=ddim_steps,
                             eta=eta,
                             simple_var=cfg["sampler"].get(
                                 "simple_var", False))


if __name__ == "__main__":
    main()
