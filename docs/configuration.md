# 配置文件详解

项目所有超参数通过 JSON 配置管理。本节详解 `configs/sr_train.json` 与 `configs/eval.json` 的字段含义与常用组合。

## 1. `sr_train.json`

```jsonc
{
  "seed": null,
  "device": "cuda",
  "data": { ... },
  "model": { ... },
  "optimization": { ... },
  "logging": { ... },
  "sampler": { ... }
}
```

### 1.1 通用设置

- `seed`：随机种子；设为整数可复现训练。
- `device`：`cuda` / `cpu`。CPU 模式建议同时将 `data.num_workers=0`。

### 1.2 数据块 `data`

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `batch_size` | int | 每个批次样本数 |
| `num_workers` | int | DataLoader 进程数 |
| `image_size` | int | HR 图像的边长 |
| `channels` | int | HR 通道数（通常为 1） |
| `h5_path` | str | HDF5 文件路径 |
| `h5_lr_key/hr_key` | str | LR/HR 分组名称 |
| `h5_lr_dataset` | str/null | 指定 dataset 名称，null 表示自动搜索 |
| `transpose_lr/hr` | bool | 是否在读取时转置 |
| `use_tfm_channels` | bool | 是否使用 TFM 三通道 |
| `coord_range_x/y` | [float, float] | 坐标归一化范围 |
| `augment` | bool | 是否启用数据增强 |
| `h_flip_prob` / `translate_prob` | float | 翻转/平移概率 |
| `max_translate_ratio` | float | 平移幅度占比 |

### 1.3 模型块 `model`

| 字段 | 默认 | 说明 |
| ---- | ---- | ---- |
| `module` | `"network"` | 模型定义所在模块 |
| `backbone` | `"unet_res"` | 可选：`unet`、`unet_res`、`convnet` 等 |
| `diffusion_steps` | 1000 | 扩散步数 |
| `checkpoint_dir` | `SR/weight_ckpt` | 训练权重保存目录 |
| `checkpoint_name` | `model023.pth` | 默认保存文件名 |

如需加载预训练权重，可将文件放入 `checkpoint_dir` 并保持同名。

### 1.4 优化器块 `optimization`

| 字段 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `epochs` | 1000 | 最大训练轮数 |
| `ema_decay` | 0.99 | EMA 衰减系数 |
| `learning_rate` | 1e-4 | 基础学习率 |
| `betas` | [0.9, 0.99] | Adam β 参数 |
| `weight_decay` | 0.01 | L2 正则 |
| `warmup_epochs` | 100 | 预热轮数（线性增长至 `lr_max`） |
| `lr_min` | 1e-6 | 余弦退火的下限 |
| `lr_max` | 5e-4 | 余弦退火的上限 |
| `amp` | true | 是否启用自动混合精度 |

额外可选字段：

- `resume_checkpoint`：继续训练时加载的权重路径。

### 1.5 日志块 `logging`

| 字段 | 说明 |
| ---- | ---- |
| `tensorboard_root` / `experiment_name` | TensorBoard 日志目录/实验名 |
| `sample_interval` | 每隔多少 step 保存预览图 |
| `num_preview_samples` | 采样图像数量 |
| `preview_seed` | 预览图随机种子 |
| `print_model_summary` | 是否打印模型结构 |
| `record_psnr` | 是否记录 PSNR 到 TensorBoard |
| `save_preview_images` | 是否写出预览图 |
| `preview_dir` | 预览图保存目录 |

### 1.6 采样块 `sampler`

| 字段 | 说明 |
| ---- | ---- |
| `type` | 训练阶段使用的采样器（通常 `ddpm`） |
| `simple_var` | 是否启用简化方差（对应 DDPM 简化公式） |

## 2. `eval.json`

结构与训练类似，但增加了 VAE 与输出配置：

```jsonc
{
  "device": "cuda",
  "data": { ... },
  "model": { ... },
  "vae": { ... },
  "sampler": { ... },
  "output": { ... }
}
```

### 2.1 模型块

- `checkpoint_path`：评估时加载的权重路径。
- 其余字段与训练一致。

### 2.2 VAE 块（可选）

| 字段 | 说明 |
| ---- | ---- |
| `enabled` | 是否启用 VAE 编解码 |
| `checkpoint_path` | VAE 权重路径 |
| `encoder_key` / `decoder_key` | state_dict 中的键名（支持只加载子模块） |

开启后，`eval.py` 会先将输入编码到 latent，再执行扩散超分，输出前再解码回像素空间。

### 2.3 采样块

| 字段 | 默认 | 说明 |
| ---- | ---- | ---- |
| `threshold` | 0.05 | 低于该阈值的像素将被置零 |
| `type` | `ddim` | 推理采样器 |
| `ddim_steps` | 20 | 推理步数 |
| `eta` | 1.0 | 随机噪声系数 |
| `simple_var` | false | 是否使用简化方差（与 DDPM 对齐时置 true） |

### 2.4 输出块

| 字段 | 说明 |
| ---- | ---- |
| `root` | 评估结果根目录 |
| `save_images` | 是否保存重建图像 |

## 3. 配置管理最佳实践

1. **复制模板**：每次实验前复制模板 JSON，命名为 `configs/exp_xxx.json`。
2. **版本控制**：将配置纳入 Git，配合 `SR/eval_results_*/config.json` 追踪实验。
3. **避免共享状态**：修改嵌套 dict 时使用 `.copy()`，避免 Python 引用导致的副作用。
4. **参数校验**：运行脚本前可使用 `python -m json.tool configs/xxx.json` 校验格式。
5. **记录日志**：在《实验记录和结果》中写明使用的配置文件名及关键改动。

合理利用配置，可以快速切换不同骨干、采样策略与数据设置，提升实验效率。
