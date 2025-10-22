# ddpm-sr-for-TFM

扩散模型用于超分辨率重建的实验代码。项目已经移除对第三方 `dldemos` 包的依赖，所有模块均位于当前仓库内，并通过 JSON 配置文件驱动。

## 环境准备

- Python 3.10+
- PyTorch ≥ 2.1（需包含 CUDA 支持时请安装 `torch` / `torchvision` 的 GPU 版本）
- 依赖库：`torchmetrics`、`opencv-python`、`tensorboard`、`einops`、`tqdm`

示例安装命令：

```bash
pip install torch torchvision torchmetrics opencv-python tensorboard einops tqdm
```

## 配置文件

所有超参数与路径均集中在 `configs/` 目录中：

- `configs/sr_train.json`：训练配置（数据路径、网络骨干、优化器、日志设置等）。
- `configs/eval.json`：评估配置（模型权重、测试集、采样器、VAE 权重以及输出目录）。

建议复制一份 JSON 后再修改，避免无意改动默认模板。

## 训练

```bash
python SR_train.py --config configs/sr_train.json
```

- 训练日志写入 `runs/`，可通过 `tensorboard --logdir runs` 查看。
- 预览图像可在配置中设置 `preview_dir`，默认保存到 `SR/previews/`。
- 训练使用 `ema` 权重自动保存到 `model.checkpoint_dir` 指定的位置。

## 评估

```bash
python eval.py --config configs/eval.json
```

- 支持 DDPM 与 DDIM 采样（通过配置文件的 `sampler.type` 控制）。
- 若提供 VAE 权重，评估时会先压缩到潜空间再解码，以提升重建质量。
- 评估结果将写入 `output.root` 目录，包括：
  - `images/`：每张图的重建结果。
  - `results.txt`：总体 PSNR / SSIM。
  - `results_per_image.csv`：逐图指标。

## 实用脚本

- SR_train.py：主训练入口，支持配置化训练与采样预览。
- noise.py：为单张图像添加 DDPM 前向噪声，可用于可视化退火过程。

## 测试

轻量级的 smoke test 位于 `tests/`，用于校验扩散调度器的基本行为：

```bash
pytest
```

新增功能时请优先补充测试（例如张量形状、采样流程等），并在提交前执行 `pytest` 以及一次评估脚本。

## 目录结构（节选）

```
configs/             配置文件
ddim.py              DDIM 采样器实现
ddpm.py              原始 DDPM 调度器
ddpm_simple.py       条件扩散工具（含 SR 支持）
network.py           基线 UNet/ConvNet
SR_train.py          单一训练入口
eval.py              指标评估入口
encoder.py / decoder.py    VAE 子模块
dataset.py           数据预处理与 DataLoader
tests/               单元测试
```

如需 GPU/CPU 切换或路径调整，可直接在 JSON 配置中修改对应字段，无需改动源码。欢迎根据业务需要扩展配置并补充测试用例。
