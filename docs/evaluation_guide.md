# 评估指南

`eval.py` 提供从 HDF5 数据到指标汇总的完整评估流程。本文介绍如何配置、运行以及解析输出来验证训练好的扩散模型。

## 1. 前置条件

1. **模型权重**  
   - 在训练结束后，选择合适的 checkpoint（通常是 `*_best.pth` 或 `*_best_psnr.pth` 中的 EMA 权重）。  
   - 将路径写入评估配置的 `model.checkpoint_path`。
2. **评估数据**  
   - 构建与训练相同结构的 `eval.h5`，包含 `TFM/编号` 与 `hr/编号`。  
   - 若使用 TFM 三通道，确保每个样本含有 `I`（或 `intensity`）、`X`、`Y`。
3. **准备配置**  
   - 复制 `configs/eval.json` 为新的配置文件，更新以下字段：
     - `data.h5_path`: 评估集位置。
     - `output.root`: 输出目录（建议按实验命名）。
     - `sampler.type`: `ddpm` 或 `ddim`；后者速度更快。
     - `sampler.ddim_steps`, `sampler.eta`: DDIM 采样步数与随机性。
     - `sampler.threshold`: 对低值像素的阈值裁剪（默认 0.05）。

## 2. 运行评估

```bash
python eval.py --config configs/eval_exp.json
```

脚本主要步骤：

1. **加载模型**：根据 `model.backbone` 构建网络，并从 checkpoint 读取权重（优先 `ema_model_state_dict`）。
2. **创建数据加载器**：`get_h5_dataloader` 在评估时关闭增强，保持样本顺序。
3. **准备采样器**：依据 `sampler.type` 选择 DDPM 或 DDIM，并读取可选参数（步数、`eta`、`simple_var`）。
4. **生成 SR 图像**：调用 `sample_backward_sr`，逐批次生成超分结果。
5. **后处理**：
   - `denormalize` 将输出映射回 `[0,1]`。
   - 将小于 `threshold` 的像素置零，抑制噪声。
   - 可选地将灰度图映射到 Jet 伪彩（`channels == 1` 时生效）。
6. **指标与写盘**：
   - 使用 `torchmetrics` 计算 PSNR / SSIM，并累计平均值。
   - 若 `output.save_images=true`，将结果保存为 PNG；同时写出 `results.txt` 与 `results_per_image.csv`。

## 3. 输出结构

```
SR/eval_results_xxx/
├── images/                 # 重建图像（可选）
│   ├── 000001.png
│   └── ...
├── results.txt             # 全局统计
├── results_per_image.csv   # 逐样本指标
└── config.json             # 可选：手动保存评估配置快照
```

`results.txt` 中包含模型路径、数据源、采样器类型、平均 PSNR / SSIM。CSV 则方便后续排序或与其他实验对比。

## 4. 常用参数调优

| 字段 | 作用 | 建议 |
| ---- | ---- | ---- |
| `sampler.type` | 控制采样算法 | DDIM 适合快速评估，DDPM 更具随机性 |
| `sampler.ddim_steps` | 推理步数 | 质量不足时提高到 50；快速预览可降到 10 |
| `sampler.eta` | DDIM 随机噪声系数 | 0 取确定性输出；>0 提升多样性 |
| `sampler.threshold` | 输出阈值 | 适合控制背景噪声，可在 0.01~0.1 间调节 |
| `output.save_images` | 是否保存图片 | 只关心数值指标时可设为 `false` |

## 5. 复现建议

- 固定随机种子：评估脚本内部对 `torch.manual_seed` / `torch.cuda.manual_seed_all` 设置常量 1234，如需实验随机性可自定义。
- 保留配置与环境信息：建议把使用的 `configs/eval_exp.json`、PyTorch 版本、GPU 型号写入《实验记录和结果》。
- 若启用 VAE：在配置的 `vae` 部分填入权重路径并设置 `enabled=true`，脚本会加载 `encoder.py` / `decoder.py` 中的模型进行潜空间重建。

## 6. 故障排查

| 问题 | 对策 |
| ---- | ---- |
| 找不到 checkpoint | 确认 `model.checkpoint_path` 指向实际文件，路径可使用相对或绝对形式 |
| 通道/尺寸不匹配 | 检查评估配置中的 `channels`、`image_size` 是否与训练一致；确保 HDF5 数据已归一化 |
| 输出全黑或全白 | 查看 `sampler.threshold` 是否过高；确认 `denormalize` 前张量值域是否在 [-1,1] |
| 运行缓慢 | 调低 `ddim_steps`、关闭图像保存、或使用 GPU 加速 |

评估完成后，将结果登记在《实验记录和结果》中，并根据需要更新《FAQ》整理新的经验。
