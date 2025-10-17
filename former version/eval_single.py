import os
import csv
import matplotlib
from tqdm import tqdm
import torch
import matplotlib.cm as cm
from PIL import Image
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure#后续新版需要在torchmetrics.image下调用
from dldemos.ddpm.dataset import  get_img_shape,get_paired_dataloader,_build_transform
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddim.ddim import DDIM
from dldemos.ddpm.newnetwork import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)
from encoder import VAE_Encoder
from decoder import VAE_Decoder

# =================================================================================
# 2. 帮助函数：将Tensor转为可保存的PIL Image
# =================================================================================
def tensor_to_pil(tensor):
    """将范围在[-1, 1]或[0, 1]的Tensor转为PIL Image"""
    # 如果tensor在[-1, 1]范围，转换到[0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # clamp(0, 1)确保数值范围不会溢出
    tensor = tensor.clamp(0, 1)
    
    # 如果是3D tensor [C,H,W]，转换为 [H,W,C]
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 4:
        # 如果是4D tensor [B,C,H,W]，取第一个batch
        tensor = tensor[0].permute(1, 2, 0)
    
    # 转换回 [0, 255] 并转为uint8
    tensor_np = (tensor.cpu().numpy() * 255).astype('uint8')
    pil_image = Image.fromarray(tensor_np)
    return pil_image

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

    # 生成 jet 查色表（CPU）并搬到目标设备 - 修复deprecated警告
    try:
        # 新的matplotlib方式
        import matplotlib
        lut_np = matplotlib.colormaps['jet'](np.linspace(0, 1, bins))[:, :3]  # (bins,3)
    except (AttributeError, ImportError):
        # 回退到旧方式
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
def jet_palette_flat_256():
    cmap = matplotlib.colormaps.get_cmap('jet').resampled(256)
    lut = (cmap(np.linspace(0,1,256))[:,:3] * 255).round().astype(np.uint8)
    return lut.reshape(-1).tolist()  # 长度 256*3

def save_paletted_from_float(x01: np.ndarray, path: str, cap: int = 255):
    """
    x01: 2D numpy, 值域 0..1（若是 torch.Tensor 先 .detach().cpu().numpy()）
    cap: 最大索引（想限制在 227 就填 227）
    """
    x01 = np.clip(x01.astype(np.float32), 0.0, 1.0)
    idx = (x01 * cap).round().astype(np.uint8)        # 0..cap
    im  = Image.fromarray(idx, mode='P')              # P 模式（索引图）
    im.putpalette(jet_palette_flat_256())             # jet(256) 调色板
    im.save(path, format='PNG')

def apply_jet_colormap(tensor):
    """将tensor应用jet颜色映射，保持原始尺寸"""

    # 如果是彩色图像，转换为灰度图
    if tensor.shape[0] == 3:
        # 使用标准RGB到灰度的转换权重
        gray_tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    else:
        gray_tensor = tensor.squeeze(0)
    
    # 转换为numpy数组
    gray_np = gray_tensor.cpu().numpy()
    
    # 使用matplotlib的jet颜色映射（如果matplotlib可用）
    try:
        # 应用jet颜色映射
        jet_colored = cm.jet(gray_np)
        # 转换为PIL图像 (移除alpha通道)
        jet_colored_rgb = (jet_colored[:, :, :3] * 255).astype(np.uint8)
        pil_image = Image.fromarray(jet_colored_rgb)
    except ImportError:
        # 如果matplotlib不可用，直接保存为灰度图
        print("Warning: matplotlib not available, saving as grayscale")
        pil_image = Image.fromarray((gray_np * 255).astype('uint8')).convert('RGB')
    
    return pil_image

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]
# =================================================================================
# 3. 主评估函数 (修改后)
# =================================================================================
def evaluate(): # <--- 不再需要args参数
    # ===========================================================
    # --- 配置参数 (请在这里直接修改) ---
    class Config:
        model_path = './SR/auotodl_weight/modelSR1009_07.pth'
        test_data_path = './data/eval/'
        output_dir = 'SR/eval1007'  # 结果保存路径
        image_size = 400
        batch_size = 1
        device = 'cuda'
        sr_threshold = 0  # SR阈值，小于该值的像素将被设为0
    
    config = Config()
    # ===========================================================

    # --- 设置 ---
    device = torch.device(config.device if torch.cuda.is_available() else "cpu") # <--- 使用 config.device
    print(f"Using device: {device}")

    # --- 创建输出文件夹 ---
    images_output_dir = os.path.join(config.output_dir, "images") # <--- 使用 config.output_dir
    os.makedirs(images_output_dir, exist_ok=True)
    print(f"结果图片将保存至: {images_output_dir}")

    # --- 数据加载 ---
    test_dataloader = get_paired_dataloader(config.batch_size, lr_root=os.path.join(config.test_data_path, 'TFM'), hr_root=os.path.join(config.test_data_path, 'hr'), num_workers=4)

    # --- 模型加载 ---
    config_id = 4 # <--- 选择配置
    n_steps = 1000
    configg = configs[config_id].copy() # 使用 .copy() 避免修改原始配置
    net = build_network(configg, n_steps)
    enc = VAE_Encoder().to(device)
    dec = VAE_Decoder().to(device)
    ddpm = DDIM(device, n_steps) # or DDIM
    
    print(f"从 {config.model_path} 加载模型权重...")
    checkpoint = torch.load(config.model_path, map_location=device) # <--- 使用 config.model_path
    net.load_state_dict(checkpoint['ema_model_state_dict'])
    vaemodel_path = './SR/auotodl_weight/vae_best-1.pt' # <--- VAE模型路径
    vae_checkpoint = torch.load(vaemodel_path, map_location=device)
    enc.load_state_dict(vae_checkpoint['enc'])
    dec.load_state_dict(vae_checkpoint['dec'])
    net.eval()
    enc.eval()
    dec.eval()

    # --- 指标初始化 ---
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    individual_results = []
    # 循环外（一次性）
    g = torch.Generator(device=device).manual_seed(1234)
    fixed_noise_1 = torch.randn((1, 1, config.image_size, config.image_size), generator=g, device=device)
    # --- 开始评估循环 ---
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Evaluating")
        for lr_images, hr_images, img_names in pbar:
            seed = 1234
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # sec_noise = torch.randn(4, 3, config.image_size, config.image_size).to(device) # <--- 使用 config.image_size
            # noise = sec_noise[2:3]
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            batch_size = lr_images.shape[0]
            print('batch_size:', batch_size)
            shape = (batch_size, 1, config.image_size, config.image_size) # <--- 使用 config.image_size
            # shape = fixed_noise_1
            sr_images = ddpm.sample_backward_sr(shape, net, lr_images, device=device,simple_var=False,ddim_step=20, eta=1.0)
            # sr_images = ddpm.sample_backward_sr(shape,net,lr_images,device=device,simple_var=False)
            # noise = torch.randn(config.batch_size, 4, config.image_size//8, config.image_size//8, device=device)
            # z, _, _ = enc(sr_images, noise)
            # sr_images = dec(z)
            # 切记更换net结构
            sr_images_metric = ((sr_images + 1) / 2).clamp(0, 1)
            
            # 应用阈值处理：小于阈值的像素设为0
            sr_images_metric = torch.where(sr_images_metric < config.sr_threshold, 
                                         torch.zeros_like(sr_images_metric), 
                                         sr_images_metric)

            # 进行压缩：将数值范围从[0,1]压缩到[0, 227/253]
            sr_images_metric = sr_images_metric * (227.0 / 253.0)
            
            hr_images_metric = ((hr_images + 1) / 2).clamp(0, 1)

            psnr.update(sr_images_metric, hr_images_metric)
            ssim.update(sr_images_metric, hr_images_metric)
            
            for i in range(batch_size):
                sr_img = sr_images_metric[i].unsqueeze(0)
                hr_img = hr_images_metric[i].unsqueeze(0)
                psnr_val = peak_signal_noise_ratio(sr_img, hr_img)
                ssim_val = structural_similarity_index_measure(sr_img, hr_img)
                individual_results.append({
                    "filename": img_names[i],
                    "psnr": psnr_val.item(), # .item() 将单元素Tensor转为Python数字
                    "ssim": ssim_val.item()
                })
                # 使用jet颜色映射保存图片，保持原始尺寸
                # jet_pil = apply_jet_colormap(sr_images_metric[i])
                # # 转换为P模式（调色板模式）
                # jet_pil_p = jet_pil.convert('P')
                save_path = os.path.join(images_output_dir, img_names[i])
                # jet_pil_p.save(save_path)
                # 从标量直接保存 P 模式（推荐）
                x01 = sr_images_metric[i]
                save_paletted_from_float(x01.squeeze(0).detach().cpu().numpy(), save_path, cap=255)  # 或 227


    # --- 计算并保存最终结果 ---
    avg_psnr = psnr.compute()
    avg_ssim = ssim.compute()

    print("\n--- 评估完成 ---")
    print(f"平均 PSNR: {avg_psnr:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f}")

    results_path = os.path.join(config.output_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n")
        f.write(f"Model: {config.model_path}\n")
        f.write(f"Test Set: {config.test_data_path}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    print(f"评估结果数据已保存至: {results_path}")

    results_path_csv = os.path.join(config.output_dir, "results_per_image.csv")
    with open(results_path_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'psnr', 'ssim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(individual_results)
    print(f"每张图片的详细评估结果已保存至: {results_path_csv}")

# =================================================================================
# 4. 主程序入口 (修改后)
# =================================================================================
if __name__ == '__main__':
    evaluate() # <--- 直接调用函数，不再需要解析参数
