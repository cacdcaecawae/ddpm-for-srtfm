# 训练指南

本指南直接对应 `SR_train.py` 的实现，帮助你从配置到监控完整走通训练流程。

## 1. 准备步骤

1. **依赖与环境**
   - 安装 PyTorch ≥ 2.1（推荐 GPU 版本）及仓库 README 中列出的 Python 包。
   - 若使用 CUDA，请确认驱动与 `torch.cuda.is_available()` 正常。
2. **数据集**
   - 按照《数据准备说明》构建 `train.h5`，结构包含 `TFM/编号` 与 `hr/编号`。
   - 需要 TFM 三通道时，在 HDF5 样本中提供 `I`（或 `intensity`）、`X`、`Y` 数据集。
3. **配置文件**
   - 复制 `configs/sr_train.json` 至新文件（如 `configs/exp_xyz.json`），避免污染模板。
   - 重点字段：
     - `data.h5_path`: 训练集路径。
     - `data.use_tfm_channels`: 是否启用 I/X/Y 三通道。
     - `model.backbone`: 从 `convnet_small|medium|big`、`unet`、`unet_res` 中选择。
     - `optimization.learning_rate`, `lr_min`, `lr_max`, `warmup_epochs`: 控制余弦调度。

## 2. 启动训练

```bash
python SR_train.py --config configs/exp_xyz.json
```

执行流程概览：

1. **加载配置** → 构造 `H5PairedDataset` 和 DataLoader（支持增强、转置与坐标归一化）。
2. **构建模型** → `build_network` 根据骨干类型自动拼接 LR/HR 通道并注入时间嵌入。
3. **噪声预测训练**  
   - 使用 `ddpm_simple.DDPM.sample_forward` 为 HR 添加噪声得到 `x_t`。  
   - 网络预测噪声 `εθ(x_t, t, lr)`，与真实噪声 MSE 对齐。  
   - 支持 `torch.amp.autocast` + `GradScaler` 的混合精度训练。
4. **学习率调度** → `warmup_cosine` 按 `warmup_epochs` 线性预热，随后余弦退火到 `lr_min`。
5. **EMA** → `ema_update` 按 `optimization.ema_decay` 维护滑动平均模型，预览与评估更稳定。
6. **日志与检查点**  
   - TensorBoard 写入 Loss / PSNR，目录位于 `runs/<timestamp>-<experiment_name>`。  
   - 每轮保存常规 checkpoint；当平均 Loss 或 PSNR 创新时额外保存 `_best` 与 `_best_psnr`。  
   - 若 `logging.save_preview_images` 为 `true`，训练结束后会使用 EMA 模型在 `SR/previews/` 生成采样图。

## 3. 常用调参建议

| 目标 | 建议设置 | 说明 |
| ---- | -------- | ---- |
| 学习率 | `learning_rate`≈1e-4，`lr_max`≤5e-4 | 过高易引起发散，可结合 Batch Size 适度调节 |
| 预热/退火 | `warmup_epochs`≈10% 总轮数，`lr_min`=1e-6 | 预热结束后进入余弦衰减，收敛更平滑 |
| EMA | `ema_decay`=0.99~0.995 | 较大的衰减提升稳定性但响应更慢 |
| Mixed Precision | `optimization.amp=true`（默认） | 在 GPU 上显著节省显存与时间 |
| 数据增强 | 初期可关闭，模型稳定后开启翻转/平移 | 通过 `augment`、`h_flip_prob`、`translate_prob` 控制 |

## 4. 监控与断点恢复

- **TensorBoard**：`tensorboard --logdir runs` 查看训练曲线与采样网格。
- **检查点**：默认保存在 `model.checkpoint_dir`；`checkpoint_name` 决定主文件名。
- **恢复训练**：在配置中设置 `model.resume_checkpoint` 指向现有权重；脚本会加载 `model_state_dict` / `ema_model_state_dict` 继续训练。

## 5. 采样模式

使用同一脚本可直接生成采样图，无需单独运行评估：

```bash
python SR_train.py --config configs/exp_xyz.json --mode sample --sampler ddim --ddim-steps 20
```

此命令会加载配置中的 `checkpoint_path`，根据 `sampler` 选择 DDPM 或 DDIM，并把结果写入 `logging.preview_dir`。

## 6. 常见问题排查

| 现象 | 排查思路 |
| ---- | -------- |
| Loss/Fwd NaN | 检查数据是否存在 NaN；适当降低学习率；暂时关闭 AMP 验证 |
| 训练稳定但指标低 | 增加 `sampler.ddim_steps` 评估更多步数；调低 `sampler.threshold`；确认 HDF5 归一化一致 |
| 生成伪影或裁切异常 | 检查 `transpose_lr/hr` 设置、坐标范围是否匹配；确保 LR 与 HR 尺寸一致 |
| GPU 显存不足 | 减小 `batch_size`、使用 `convnet_small`、或关闭 `use_tfm_channels` |

## 7. 训练后续

1. 执行 `pytest` 确认基础测试通过。
2. 运行评估脚本（见《评估指南》）并记录指标到《实验记录和结果》。
3. 备份配置、日志与模型权重，方便后续复现或对比。
