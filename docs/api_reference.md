# API 参考

汇总训练与评估中常用的类和函数，便于在代码层面查找入口。源码位置主要在 `dataset.py`、`network.py`、`ddpm_simple.py`、`ddim.py`、`SR_train.py`、`eval.py`。

## 1. 数据模块

### `dataset.H5PairedDataset`

```python
class H5PairedDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        lr_key: str = "TFM",
        hr_key: str = "hr",
        lr_dataset_name: Optional[str] = None,
        hr_dataset_name: Optional[str] = None,
        transpose_lr: bool = False,
        transpose_hr: bool = False,
        use_tfm_channels: bool = False,
        coord_range: Optional[tuple] = None,
        augment: bool = False,
        h_flip_prob: float = 0.5,
        translate_prob: float = 0.5,
        max_translate_ratio: float = 0.05,
    )
```

- 返回 `(lr_tensor, hr_tensor, sample_name)`，张量取值范围为 `[-1, 1]`，shape `[C, H, W]`。
- 自动完成强度、坐标归一化以及可选的数据增强。

### `dataset.get_h5_dataloader`

```python
def get_h5_dataloader(..., batch_size: int, shuffle: bool = True, num_workers: int = 4, ...) -> DataLoader
```

封装了 `DataLoader` 创建逻辑，训练与评估脚本均通过此函数实例化迭代器。

## 2. 网络骨干

### `network.ConvNet`

```python
class ConvNet(nn.Module):
    def __init__(
        self,
        n_steps: int,
        in_channels: int = 1,
        intermediate_channels: Optional[list[int]] = None,
        pe_dim: int = 10,
        insert_t_to_all_layers: bool = False,
    )
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...
```

- 多层残差块结构，可选择只在首层或每层注入时间嵌入。
- 适合资源受限或快速实验场景。

### `network.UNet`

```python
class UNet(nn.Module):
    def __init__(
        self,
        n_steps: int,
        in_channels: int = 1,
        image_size: int = 101,
        lr_channels: Optional[int] = None,
        channels: Optional[list[int]] = None,
        pe_dim: int = 128,
        residual: bool = False,
    )
    def forward(self, x: torch.Tensor, t: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        ...
```

- 编码端将 HR 噪声 `x_t` 与 LR 条件拼接，解码端通过跳跃连接恢复细节。
- `residual=True` 时在块内启用残差支路。

### `network.build_network`

```python
def build_network(config: dict, n_steps: int, in_channels: int, image_size: int, lr_channels: Optional[int] = None) -> nn.Module
```

根据配置字典实例化 ConvNet 或 UNet，并自动填充扩散步数、通道数与图像尺寸。

## 3. 扩散调度

### `ddpm_simple.DDPM`

```python
class DDPM:
    def __init__(self, device: torch.device, n_steps: int, min_beta: float = 1e-4, max_beta: float = 0.02)
    def sample_forward(self, x, t, eps=None)
    def sample_backward(self, img_shape, net, device, simple_var=True)
    def sample_backward_sr(self, img_shape, net, lr_image, device, simple_var=True)
```

- `sample_forward`：训练阶段向 HR 图像添加噪声。
- `sample_backward_sr`：给定 LR 条件、逐步反演得到超分结果。
- 方差由 `simple_var` 控制，可选原始 DDPM 公式或简化版。

### `ddim.DDIM`

```python
class DDIM:
    def __init__(self, device: torch.device, n_steps: int)
    def sample_backward_sr(self, img_shape, net, lr_image, device, simple_var=True, ddim_step=50, eta=0.0)
```

- 接口与 DDPM 相同，但允许通过 `ddim_step` 指定推理步数、`eta` 控制随机性。
- 评估脚本默认使用 DDIM，以降低采样成本。

## 4. 训练脚本（`SR_train.py`）

核心函数：

```python
def load_config(path: Path) -> Dict[str, Any]
def ensure_dir(path: Path) -> None
def set_seed(seed: Optional[int]) -> None
def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module
def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader
def train(ddpm: DDPM, net: nn.Module, cfg: Dict[str, Any], device: torch.device, ckpt_path: Path, log_dir: Path) -> nn.Module
def sample_imgs(...); def sample_imgs_ddim(...)
```

- `train`：执行完整训练循环（噪声采样、前向、MSE 损失、AMP、EMA、日志、checkpoint）。
- `sample_imgs*`：在训练或独立采样模式下，用指定采样器生成图像网格。

## 5. 评估脚本（`eval.py`）

常用接口：

```python
def load_config(path: Path) -> Dict[str, Any]
def resolve_device(preferred: Optional[str]) -> torch.device
def create_dataloader(cfg: Dict[str, Any]) -> torch.utils.data.DataLoader
def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module
def prepare_sampler(cfg: Dict[str, Any], device: torch.device) -> Tuple[str, Union[DDPM, DDIM], Dict[str, Any]]
def evaluate(cfg: Dict[str, Any], device: torch.device) -> None
```

- `build_model`：根据配置加载权重（优先使用 `ema_model_state_dict`）。
- `prepare_sampler`：返回采样器类型、实例以及调用所需参数。
- `evaluate`：封装推理、后处理（阈值裁剪、伪彩）、指标统计与结果保存。

## 6. 工具模块

- `noise.py:add_noise(image: np.ndarray, beta: float, steps: int)`：在 CPU 上模拟 DDPM 前向过程，主要用于可视化。
- `encoder.py` / `decoder.py`：若启用配置中的 VAE，会被评估脚本加载处理潜空间。
- `attention.py`：包含可复用的注意力层与 MLP，供自定义网络扩展。

## 7. 简易示例

```python
from pathlib import Path
import torch

from dataset import get_h5_dataloader
from network import build_network, unet_res_cfg
from ddpm_simple import DDPM

cfg = {
    "data": {
        "h5_path": "data/train.h5",
        "batch_size": 4,
        "image_size": 101,
        "channels": 1,
        "use_tfm_channels": False,
    },
    "model": {
        "backbone": "unet_res",
        "diffusion_steps": 1000,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = get_h5_dataloader(**cfg["data"], shuffle=True)
net = build_network(unet_res_cfg, cfg["model"]["diffusion_steps"],
                    in_channels=cfg["data"]["channels"],
                    image_size=cfg["data"]["image_size"]).to(device)
ddpm = DDPM(device, cfg["model"]["diffusion_steps"])

lr_batch, hr_batch, names = next(iter(dataloader))
lr_batch = lr_batch.to(device)
hr_batch = hr_batch.to(device)
t = torch.randint(0, ddpm.n_steps, (hr_batch.size(0),), device=device)
x_t = ddpm.sample_forward(hr_batch, t)
pred_noise = net(x_t, t, lr_batch)
```

如需进一步了解张量尺寸或网络结构，可使用 `print(net)`、`torchinfo.summary` 或直接阅读相应源码。
