# 数据准备说明

扩散重建流程依赖 HDF5 格式的 LR/HR 配对数据。本节结合 `dataset.H5PairedDataset` 的实现，说明如何构建标准数据集。

## 1. HDF5 目录结构

默认结构（可在配置中修改）：

```
train.h5 / eval.h5
├── TFM/                   # 低分辨率 (LR) 观测
│   ├── 000001/data
│   ├── 000002/data
│   └── ...
└── hr/                    # 高分辨率 (HR) Ground Truth
    ├── 000001/data
    ├── 000002/data
    └── ...
```

- `TFM` 与 `hr` 分组下的样本编号需一一对应。
- `data` 可以是 `h5py.Dataset` 或嵌套分组；若名称非 `data`，可在配置中设置 `h5_lr_dataset` / `h5_hr_dataset`。
- 支持三种模式：
  1. **实部/虚部**：将复数场拆分为两个通道，堆叠成 `(2, H, W)`。
  2. **TFM 三通道**：Intensity / X / Y 三个子通道，`use_tfm_channels=true`。
  3. **单通道强度**：对原始幅值归一化后得到 `(1, H, W)`。

## 2. 数据转换示例

以下示例展示如何将 NPY 数据转为 HDF5：

```python
import h5py
import numpy as np

samples = range(1, 7201)

with h5py.File("data/train.h5", "w") as f:
    lr_group = f.create_group("TFM")
    hr_group = f.create_group("hr")

    for idx in samples:
        name = f"{idx:06d}"
        lr_data = np.load(f"./npy/lr/{name}.npy")       # shape (H, W) or (C, H, W)
        hr_data = np.load(f"./npy/hr/{name}.npy")       # shape (H, W)

        g_lr = lr_group.create_group(name)
        g_lr.create_dataset("data", data=lr_data, compression="gzip")

        g_hr = hr_group.create_group(name)
        g_hr.create_dataset("data", data=hr_data, compression="gzip")
```

若数据源自 MATLAB，需要注意列优先与行优先差异：`transpose_lr` / `transpose_hr` 可在读取时自动转置。

## 3. 归一化与坐标

默认归一化逻辑：

1. **强度归一化**：Min-Max → `[0, 1]` → 映射到 `[-1, 1]`，使网络输出范围稳定。
2. **坐标归一化**：若存储了反演的 XY 坐标，可使用配置项 `coord_range_x/y` 指定真实范围，再线性映射到 `[-1, 1]`。

所有归一化逻辑已封装在 `H5PairedDataset` 中，除非需要特殊处理，无需自行实现。

## 4. 数据增强

在配置中开启：

- `augment`: true/false
- `h_flip_prob`: 水平翻转概率
- `translate_prob`: 平移概率
- `max_translate_ratio`: 平移像素占图像尺寸的比例

增强仅作用于训练集；评估时建议关闭，以保证指标稳定。

## 5. 数据划分建议

常见实践：

- 在训练初期保留部分样本作为验证集，监控 `train/loss` 与 `val/psnr_mean`。
- 当模型趋于稳定时，可合并数据进行全量训练，但需定期运行评估脚本防止过拟合。

## 6. 常见问题与排查

| 问题 | 解决方案 |
| ---- | -------- |
| `KeyError: HDF5 文件缺少分组` | 检查 HDF5 是否严格遵循 TFM/HR 架构；使用 `h5dump` 或自写脚本验证 |
| `ValueError: dataset not unique` | 当同一编号下存在多个 dataset 时，请通过配置指定 `h5_lr_dataset` / `h5_hr_dataset` |
| 读取后图像旋转/镜像 | 尝试切换 `transpose_lr/hr` 或在生成 HDF5 时手动转置 |
| 归一化结果全 0 | 确认源数据非常量；若是整数类型，转为 float 再归一化 |
| GPU 推理异常 | 检查 `use_tfm_channels` 与模型输入通道数是否一致；必要时在配置中设置 `data.channels` |

准备就绪后，即可按照《训练指南》启动模型学习，并在《评估指南》中完成测试。
