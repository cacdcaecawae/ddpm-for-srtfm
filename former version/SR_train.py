import time
import os
from copy import deepcopy
from PIL import Image
from matplotlib import cm
import math
import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
# 确保这里的函数名和您的 dataset.py 文件一致
from dldemos.ddpm.dataset import  get_img_shape,get_paired_dataloader,_build_transform
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure#后续新版需要在torchmetrics.image下调用
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddim.ddim import DDIM
from dldemos.ddpm.newnetwork import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

# --- 可配置参数 ---
batch_size = 64
n_epochs = 1000 # 建议至少训练100个epoch以上
ema_decay = 0.99 # EMA衰减率

ts = time.strftime("%Y%m%d-%H%M%S")            # 例如 20250902-153045
log_dir = os.path.join("runs", f"{ts}-test")  # runs/20250902-153045-test

# ---------- EMA 工具（参数+buffers，一次到位） ----------
@torch.no_grad()
def ema_update(ema_model, model, decay: float):
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        ema_sd[k].mul_(decay).add_(v, alpha=1 - decay)
    ema_model.load_state_dict(ema_sd)  # 回写

# ----------模拟余弦退火学习率----------
def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
from matplotlib import cm

def to_jet(x, vmin=None, vmax=None, bins=256):
    """
    x: [B,1,H,W] 或 [1,H,W] 或 [H,W]，数值型张量（可在 GPU）
    vmin/vmax: 颜色映射用到的最小/最大值（不传则按张量自身 min/max）
    bins: 色标等级（和生成 RGB 时一致，常用 256）
    return: 同设备的 [B,3,H,W]，数值范围 [0,1]
    """
    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        squeeze_back = True
    elif x.dim() == 3:  # [1,H,W] -> [1,1,H,W]
        x = x.unsqueeze(0)

    device = x.device
    if vmin is None:
        vmin = float(x.min().item())
    if vmax is None:
        vmax = float(x.max().item())
    if vmax == vmin:
        vmax = vmin + 1e-6  # 避免除零

    # 归一化到 [0,1]
    x_norm = (x - vmin) / (vmax - vmin)
    x_norm = x_norm.clamp(0, 1)

    # 生成 jet 查色表（CPU）并搬到目标设备
    lut_np = cm.get_cmap('jet', bins)(np.linspace(0, 1, bins))[:, :3]  # (bins,3)
    lut = torch.from_numpy(lut_np).to(device=device, dtype=torch.float32)

    # 量化为索引 0..bins-1
    idx = (x_norm * (bins - 1)).round().long()   # [B,1,H,W]

    # 查表映射到 RGB
    rgb = lut[idx.squeeze(1)]                    # [B,H,W,3]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()   # [B,3,H,W]，范围 [0,1]

    if squeeze_back:
        rgb = rgb.squeeze(0)  # [3,H,W]
    return rgb

# ---------- 训练 ----------
def train(ddpm: DDPM, net, device='cuda', ckpt_path='model.pth'):
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Start training, batch size: {batch_size}, epochs: {n_epochs}')
    n_steps = ddpm.n_steps
    dataloader = get_paired_dataloader(batch_size, lr_root='./data/train/TFM', hr_root='./data/train/hr', num_workers=4)
    net = net.to(device).train()
    ema_net = deepcopy(net).eval().requires_grad_(False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2)

    # ====== 最优权重跟踪 ======
    best_loss = float('inf'); best_loss_epoch = -1
    model_best_state_dict = None
    ema_model_best_state_dict = None

    # 如果你也想按PSNR存一份“最优”
    best_psnr = -float('inf'); best_psnr_epoch = -1
    model_best_state_dict_by_psnr = None
    ema_model_best_state_dict_by_psnr = None

    use_amp = (device.startswith('cuda') and torch.cuda.is_available())
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f'Using AMP: {use_amp}')

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {e+1}/{n_epochs}")
        warmup_cosine(optimizer, e, n_epochs-1, lr_min=1e-6, lr_max=5e-4, warmup_epoch=n_epochs//10)

        for lr_images, hr_images, _ in pbar:
            B = hr_images.shape[0]
            hr_images = hr_images.to(device, non_blocking=True)
            lr_images = lr_images.to(device, non_blocking=True)

            t = torch.randint(0, n_steps, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(hr_images)
            x_t = ddpm.sample_forward(hr_images, t, eps)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # ★ 建议：t 直接用 [B] 传入，不要 reshape(B,1)
                eps_theta = net(x_t, t, lr_images)
                loss = loss_fn(eps_theta, eps)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            ema_update(ema_net, net, decay=ema_decay)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ====== 可视化与PSNR（每10轮）======
        mean_psnr = None
        if (e + 10) % 10 == 0:
            net_was_training = net.training
            net.eval(); ema_net.eval()
            with torch.inference_mode():
                img  = ddpm.sample_backward_sr((4,*get_img_shape()), ema_net, lr_images[:4], device=device, simple_var=True).cpu()
                img1 = ddpm.sample_backward_sr((4,*get_img_shape()), net,     lr_images[:4], device=device, simple_var=True).cpu()
            if net_was_training: net.train()

            hr_img_batch = hr_images[:4].cpu()
            sr01 = ((img  + 1)/2).clamp(0,1)
            hr01 = ((hr_img_batch + 1)/2).clamp(0,1)

            psnr_scores = []
            for i in range(sr01.size(0)):
                psnr_val = peak_signal_noise_ratio(sr01[i].unsqueeze(0), hr01[i].unsqueeze(0), data_range=1.0)
                psnr_scores.append(psnr_val.item())
            if psnr_scores:
                mean_psnr = float(sum(psnr_scores)/len(psnr_scores))

            # TB 可视化（固定色尺）
            lr01  = ((lr_images[:4].detach().cpu().clamp(-1,1)+1)/2)
            ema01 = ((img.detach().cpu().clamp(-1,1)+1)/2)
            net01 = ((img1.detach().cpu().clamp(-1,1)+1)/2)
            grid0 = make_grid(to_jet(lr01,  vmin=0.0, vmax=1.0), nrow=4)
            grid1 = make_grid(to_jet(net01, vmin=0.0, vmax=1.0), nrow=4)
            grid2 = make_grid(to_jet(ema01, vmin=0.0, vmax=1.0), nrow=4)
            writer.add_image(f'sample/epoch_{e + 1}_lr',  grid0, e + 1)
            writer.add_image(f'sample/epoch_{e + 1}_net', grid1, e + 1)
            writer.add_image(f'sample/epoch_{e + 1}_ema', grid2, e + 1)

            # ====== 按 PSNR 记录一个“最优” （可选）======
            if mean_psnr is not None and mean_psnr > best_psnr:
                best_psnr = mean_psnr; best_psnr_epoch = e + 1
                model_best_state_dict_by_psnr = deepcopy(net.state_dict())
                ema_model_best_state_dict_by_psnr = deepcopy(ema_net.state_dict())

        avg_loss = total_loss / len(dataloader.dataset)
        writer.add_scalar('avg_Loss', avg_loss, e)
        if mean_psnr is not None:
            writer.add_scalar('val/psnr_mean4', mean_psnr, e)

        toc = time.time()
        print(f'Epoch {e+1}/{n_epochs} finished. Average loss: {avg_loss:.6f}. Elapsed: {(toc - tic):.2f}s')

        # ====== 按最小loss更新“最优” ======
        if avg_loss < best_loss:
            best_loss = avg_loss; best_loss_epoch = e + 1
            model_best_state_dict = deepcopy(net.state_dict())
            ema_model_best_state_dict = deepcopy(ema_net.state_dict())

            # 立刻另存一份 best 文件
            ckpt_best = {
                'epoch': best_loss_epoch,
                'ema_decay': ema_decay,
                'best_loss': best_loss,
                'model_best_state_dict': model_best_state_dict,
                'ema_model_best_state_dict': ema_model_best_state_dict,
            }
            best_path = ckpt_path.replace('.pth', '_best.pth')
            tmp_best = best_path + '.tmp'
            torch.save(ckpt_best, tmp_best)
            os.replace(tmp_best, best_path)

        # ====== 常规当前轮权重（顺带把最优一并写进去）======
        ckpt = {
            'epoch': e + 1,
            'ema_decay': ema_decay,
            'model_state_dict': net.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
            # 带上“当前已知的最优”
            'best_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'model_best_state_dict': model_best_state_dict,
            'ema_model_best_state_dict': ema_model_best_state_dict,
            # （可选）按 PSNR 的最优
            'best_psnr': best_psnr,
            'best_psnr_epoch': best_psnr_epoch,
            'model_best_state_dict_by_psnr': model_best_state_dict_by_psnr,
            'ema_model_best_state_dict_by_psnr': ema_model_best_state_dict_by_psnr,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        tmp = ckpt_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, ckpt_path)

    writer.close()
    print('Done training!')
    return ema_net

# ---------- 采样 ----------
def sample_imgs(ddpm, net, lr_images, output_path, n_sample=1, device='cuda', simple_var=True):
    assert int(n_sample**0.5) ** 2 == n_sample, "n_sample 必须是完全平方数（如 64/81/100）"
    net = net.to(device).eval()
    print(f'Sampling {n_sample} images to {output_path}...')
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())
        imgs = ddpm.sample_backward_sr(shape, net, lr_images, device=device, simple_var=simple_var).cpu()
        imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1) * 255
        grid = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))
        img_np = grid.numpy().astype(np.uint8)

        # 用 PIL 或者把 RGB->BGR 再用 cv2
        # from PIL import Image; Image.fromarray(img_np).save(output_path)
        cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print('Done sampling!')

def sample_imgs_ddim(ddim, net, output_path, n_sample=9, device='cuda', simple_var=True, ddim_step=100, eta=0.0):
    assert int(n_sample**0.5) ** 2 == n_sample, "n_sample 必须是完全平方数（如 64/81/100）"
    net = net.to(device).eval()
    print(f'Sampling {n_sample} images to {output_path}...')
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())
        imgs = ddim.sample_backward(shape, net, device=device, simple_var=simple_var, ddim_step=ddim_step, eta=eta).cpu()
        imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1) * 255
        grid = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))
        img_np = grid.numpy().astype(np.uint8)

        # 用 PIL 或者把 RGB->BGR 再用 cv2
        # from PIL import Image; Image.fromarray(img_np).save(output_path)
        cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print('Done sampling!')

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    # --- 模式开关：True 表示训练，False 表示只加载模型生成图片 ---
    TRAIN_MODEL = True
    # --- DDIM开关
    USE_DDIM = False
    # --- 路径和模型配置 ---
    ckpt_dir = 'SR/weight_ckpt'
    model_name = 'modelSR1015.pth'
    model_path = os.path.join(ckpt_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    n_steps = 1000
    config_id = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    config = configs[config_id].copy() # 使用 .copy() 避免修改原始配置
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    if TRAIN_MODEL:
        # --- 训练模式 ---
        # 训练模型，并返回 EMA 版本的模型
        # checkpoint = torch.load('./SR/weight_ckpt/modelSR1007.pth', map_location=device)
        # net.load_state_dict(checkpoint['ema_model_state_dict'])
        ema_model = train(ddpm, net, device=device, ckpt_path=model_path)
        # 使用返回的 EMA 模型生成最终样本
        # sample_imgs(ddpm, ema_model, 'work_dirs/diffusion_butterfly_final_ema.jpg', device=device)
    else:
        # --- 仅生成模式 ---
        if not os.path.exists(model_path):
            print(f"Checkpoint not found at {model_path}. Please train the model first.")
        else:
            if USE_DDIM:
                print("Using DDIM for sampling.")
                ddim = DDIM(device, n_steps)
                print(f"Loading checkpoint from {model_path}...")
                checkpoint = torch.load(model_path, map_location=device)
                net.load_state_dict(checkpoint['ema_model_state_dict'])
                save_path = os.path.join(ckpt_dir, f"{ts}-eval-ddim.jpg")
                sample_imgs_ddim(ddim, net, save_path, device=device, ddim_step=200, eta=0.0)
            else:
                print("Using DDPM for sampling.")
                print(f"Loading checkpoint from {model_path}...")
                checkpoint = torch.load(model_path, map_location=device)
                net.load_state_dict(checkpoint['ema_model_state_dict'])
                save_path = os.path.join("SR/eval", f"{ts}-eval-ddpm.jpg")
                image_processor = _build_transform(image_size=512, channels=3)
                lr_image = Image.open('./data/TFM/eval/lr/402.png').convert('RGB')
                transformed_tensor = image_processor(lr_image).to(device).unsqueeze(0)
                sample_imgs(ddpm, net, transformed_tensor, save_path, device=device)