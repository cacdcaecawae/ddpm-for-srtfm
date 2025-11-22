# 网络架构详细说明

本文基于仓库源码（`dataset.py`、`network.py`、`ddpm_simple.py`、`ddim.py`、`SR_train.py`、`eval.py`），介绍数据流到采样输出的关键组件，帮助你在理解结构的基础上扩展或调试。

## 1. 数据入口：`dataset.H5PairedDataset`

- **输入格式**：默认读取 HDF5 中 `TFM/样本名/data`（或 `I/X/Y` 子键）与 `hr/样本名/data`。支持复数强度拆分或 TFM 三通道。
- **归一化**：强度按样本 min-max 映射到 `[-1, 1]`，坐标（若存在 X/Y）按配置范围线性缩放。
- **增强**：水平翻转、平移等增强在训练时可选；评估脚本默认关闭。
- **输出**：返回 `(lr_tensor, hr_tensor, sample_name)`，张量形状统一为 `[C, H, W]`。

## 2. 时间条件网络：`network.py`

### 2.1 Positional Encoding

`PositionalEncoding(max_seq_len, d_model)` 基于正弦/余弦生成固定位置嵌入，供扩散步数 `t` 使用。扩散总步数由配置 `model.diffusion_steps` 指定。

### 2.2 ConvNet 系列

- `ConvNet` 由多层残差块堆叠，支持在每层或仅首层注入时间嵌入。
- 输入形状：`[B, C, H, W]`，输出同维度。
- 配置项：`convnet_small_cfg` / `convnet_medium_cfg` / `convnet_big_cfg` 通过 `intermediate_channels` 与 `pe_dim` 控制深度与宽度。
- 适用于轻量实验或资源受限场景。

### 2.3 UNet 系列

- `UNet` 在编码阶段将待恢复的 HR 噪声 `x_t` 与预处理好的 LR 图像在通道维拼接，再经多层下采样、跳跃连接和上采样输出。
- `UnetBlock` 采用 `LayerNorm + Conv + ReLU`，并通过线性层将时间嵌入加到特征图上。
- 配置：
  - `unet_1_cfg`：基础版，通道深度 `[10, 20, 40, 80]`。
  - `unet_res_cfg`：更深的残差版本，通道深度 `[16, 32, 64, 128, 256]`。
- 图像尺寸 (`image_size`) 与 HR 通道数 (`in_channels`) 通过 `build_network` 传入；LR 通道数可在配置中通过 `use_tfm_channels` 切换到 3 通道。

### 2.4 构建函数

`build_network(config, n_steps, in_channels, image_size, lr_channels)` 根据配置返回对应骨干，并自动注入扩散步数、图像尺寸和 LR 通道信息。`SR_train.py` 与 `eval.py` 均通过该函数实例化模型。

## 3. 扩散调度与采样

### 3.1 `ddpm_simple.DDPM`

- **前向过程**：`sample_forward(x, t, eps)` 在指定步数添加噪声，训练时用于生成 `x_t`。
- **反向过程（无条件）**：`sample_backward` 对随机噪声执行全步骤反演。
- **反向过程（条件 SR）**：`sample_backward_sr` 在每步调用网络 `net(x_t, t, lr_image)` 预测噪声，并组合 LR 条件生成 HR。
- 方差控制：`simple_var=True` 使用 β；否则按 DDPM 原公式计算。

### 3.2 `ddim.DDIM`

- `DDIM.sample_backward_sr` 与 DDPM 接口一致，但允许指定 `ddim_step`（采样步数）和 `eta`（随机性），实现快速或确定性推理。
- 评估脚本默认使用 DDIM，可通过配置切换到 DDPM。

## 4. 训练流程概览（`SR_train.py`）

1. **配置解析**：读取 JSON，构造数据加载器、模型与 DDPM 调度器。
2. **噪声预测目标**：对真值 HR 图像施加噪声得到 `x_t`，网络预测噪声 `εθ`，MSE 损失回归到真实噪声。
3. **优化策略**：
   - 优化器：AdamW，可调 `betas`、`weight_decay`。
   - 学习率：`warmup_cosine` 根据 `warmup_epochs`、`lr_min`、`lr_max` 进行预热及余弦退火。
   - AMP：默认在 CUDA 上开启 `torch.amp.autocast` 与 `GradScaler`。
   - EMA：`ema_update` 保持一份滑动平均副本，评估与采样优先使用 EMA 权重。
4. **日志与检查点**：TensorBoard 记录 loss/PSNR；每轮保存常规权重与最优 loss/PSNR 的快照；可选生成预览图。

## 5. 评估流程概览（`eval.py`）

1. **模型加载**：从配置指定的 checkpoint 读取（优先 `ema_model_state_dict`），并根据数据通道自动调整网络输入。
2. **采样器选择**：构建 DDPM 或 DDIM，读取阈值与步数等参数。
3. **逐样本推理**：使用 `sample_backward_sr` 生成 SR 结果，可选加权阈值剪裁与伪彩变换。
4. **指标与导出**：计算 PSNR / SSIM，保存到 `results.txt` / `results_per_image.csv`，并按需输出图像。

## 6. 组件协同关系

```
H5PairedDataset ─▶  DataLoader ─▶  SR_train.py
                               │          │
                               │          ├─ build_network → ConvNet / UNet
                               │          ├─ DDPM.sample_forward/backward_sr
                               │          └─ warmup_cosine / EMA / TensorBoard
                               │
                               └▶  eval.py ──▶ DDPM / DDIM 采样 → 指标 / 可视化
```

通过理解上述模块关系，可以自如地：

- 替换或新增网络骨干（新增配置后在 `MODEL_CONFIGS` 注册）。
- 集成自定义采样器（参考 `ddim.py` 实现并在配置中暴露参数）。
- 扩展数据预处理或增强逻辑（修改 `H5PairedDataset`，保持输出规范）。

更多实践建议请查阅《训练指南》《评估指南》，常见问题汇总于《FAQ》。
