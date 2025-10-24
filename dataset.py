from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class H5PairedDataset(Dataset):
    """
    HDF5 嵌套结构的配对数据集，读取 LR/HR 样本。
    
    预期的 HDF5 结构：
        file.h5
        ├── TFM/
        │   ├── 000001/data
        │   ├── 000002/data
        │   └── ...
        └── hr/
            ├── 000001/data
            ├── 000002/data
            └── ...
    """

    def __init__(
        self,
        h5_path: str,
        lr_key: str = "TFM",
        hr_key: str = "hr",
        lr_dataset_name: Optional[str] = None,
        hr_dataset_name: Optional[str] = None,
        transpose_lr: bool = False,
        transpose_hr: bool = False,
        use_tfm_channels: bool = False,  # 是否使用 I, X, Y 三通道
        coord_range: Optional[tuple] = None,  # X, Y 坐标的归一化范围 (min, max)
    ):
        self.h5_path = h5_path
        self.lr_key = lr_key
        self.hr_key = hr_key
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name
        self.transpose_lr = transpose_lr
        self.transpose_hr = transpose_hr
        self.use_tfm_channels = use_tfm_channels
        self.coord_range = coord_range if coord_range else (-1.0, 1.0)  # 默认 [-1, 1]

        self._file: Optional[h5py.File] = None
        self._sample_names: Optional[List[str]] = None
        self._lr_paths: Optional[List[str]] = None
        self._hr_paths: Optional[List[str]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None

    def _open_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _init_structure(self) -> None:
        """初始化数据集结构，建立 LR/HR 样本映射"""
        if self._sample_names is not None:
            return

        f = self._open_file()
        
        if self.lr_key not in f or self.hr_key not in f:
            raise KeyError(f"HDF5 文件缺少 '{self.lr_key}' 或 '{self.hr_key}' 分组")

        lr_group = f[self.lr_key]
        hr_group = f[self.hr_key]

        # 找到共同的样本编号
        lr_names = set(lr_group.keys())
        hr_names = set(hr_group.keys())
        shared = sorted(lr_names & hr_names)
        
        if not shared:
            raise ValueError(f"LR 和 HR 分组下没有共同的样本编号")

        self._sample_names = shared
        self._lr_paths = []
        self._hr_paths = []

        # 构建每个样本的 dataset 路径
        for name in shared:
            self._lr_paths.append(self._get_dataset_path(lr_group[name], self.lr_dataset_name, name))
            self._hr_paths.append(self._get_dataset_path(hr_group[name], self.hr_dataset_name, name))

    def _get_dataset_path(self, node: h5py.Group, dataset_name: Optional[str], sample_name: str) -> str:
        """从分组节点中获取 dataset 路径"""
        if isinstance(node, h5py.Dataset):
            return node.name

        # 如果指定了 dataset 名称，直接使用
        if dataset_name:
            if dataset_name not in node:
                raise KeyError(f"样本 '{sample_name}' 下找不到 dataset '{dataset_name}'")
            return node[dataset_name].name

        # 否则自动查找唯一的 dataset
        datasets = [child.name for child in node.values() if isinstance(child, h5py.Dataset)]
        if len(datasets) == 1:
            return datasets[0]
        elif len(datasets) == 0:
            raise ValueError(f"样本 '{sample_name}' 下没有 dataset")
        else:
            raise ValueError(f"样本 '{sample_name}' 下有多个 dataset，请指定 dataset_name 参数")

    @staticmethod
    def _normalize(array: np.ndarray, transpose: bool = False) -> torch.Tensor:
        """将 numpy 数组归一化到 [-1, 1]"""
        # MATLAB 转置：MATLAB 列优先存储，Python 行优先
        # MATLAB (H, W) → HDF5 读为 (W, H)
        # MATLAB (H, W, C) → HDF5 读为 (C, W, H)
        if transpose:
            if array.ndim == 2:
                # 2D 灰度图：(W, H) → (H, W)
                array = array.T
            elif array.ndim == 3:
                # 3D 数组：假设是 (C, W, H)，交换 W 和 H
                # (C, W, H) → (C, H, W)
                array = np.transpose(array, (0, 2, 1))
        
        # 确保是 [C, H, W] 格式（添加通道维度）
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # Min-max 归一化到 [0, 1]
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        
        # 映射到 [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor

    @staticmethod
    def _normalize_intensity(array: np.ndarray) -> torch.Tensor:
        """归一化强度图像 (I)，使用自适应 min-max 到 [-1, 1]"""
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # 自适应 min-max 归一化
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)  # → [0, 1]
        
        tensor = tensor * 2.0 - 1.0  # → [-1, 1]
        return tensor

    @staticmethod
    def _normalize_coords(array: np.ndarray, coord_range: tuple) -> torch.Tensor:
        """归一化坐标数据 (X/Y)，使用固定范围到 [-1, 1]"""
        if array.ndim == 2:
            array = array[None, ...]  # (H, W) → (1, H, W)
        
        tensor = torch.from_numpy(array).float()
        
        # 使用固定范围归一化
        min_val, max_val = coord_range
        tensor = (tensor - min_val) / (max_val - min_val)  # → [0, 1]
        tensor = tensor * 2.0 - 1.0  # → [-1, 1]
        
        return tensor

    def __len__(self) -> int:
        self._init_structure()
        return len(self._sample_names)

    def __getitem__(self, idx: int):
        self._init_structure()
        f = self._open_file()

        sample_name = self._sample_names[idx]
        
        # 处理 LR (TFM) 数据
        if self.use_tfm_channels:
            # 读取 I, X, Y 三个数据集
            lr_group = f[self._lr_paths[idx]].parent  # 获取父分组
            
            # 读取三个通道
            I_array = np.asarray(lr_group['I']) if 'I' in lr_group else np.asarray(f[self._lr_paths[idx]])
            X_array = np.asarray(lr_group['X']) if 'X' in lr_group else None
            Y_array = np.asarray(lr_group['Y']) if 'Y' in lr_group else None
            
            # 转置处理
            if self.transpose_lr:
                I_array = I_array.T if I_array.ndim == 2 else np.transpose(I_array, (0, 2, 1))
                if X_array is not None:
                    X_array = X_array.T if X_array.ndim == 2 else np.transpose(X_array, (0, 2, 1))
                if Y_array is not None:
                    Y_array = Y_array.T if Y_array.ndim == 2 else np.transpose(Y_array, (0, 2, 1))
            
            # 归一化 I（强度）：自适应 min-max
            I_tensor = self._normalize_intensity(I_array)
            
            # 归一化 X, Y（坐标）：固定范围
            if X_array is not None and Y_array is not None:
                X_tensor = self._normalize_coords(X_array, self.coord_range[0])
                Y_tensor = self._normalize_coords(Y_array, self.coord_range[1])
                # 拼接成 [3, H, W]
                lr_tensor = torch.cat([I_tensor, X_tensor, Y_tensor], dim=0)
            else:
                # 如果没有 X, Y，只用 I
                lr_tensor = I_tensor
        else:
            # 原来的单通道模式
            lr_array = np.asarray(f[self._lr_paths[idx]])
            lr_tensor = self._normalize(lr_array, transpose=self.transpose_lr)
        
        # 处理 HR 数据（保持原样）
        hr_array = np.asarray(f[self._hr_paths[idx]])
        hr_tensor = self._normalize(hr_array, transpose=self.transpose_hr)

        return lr_tensor, hr_tensor, sample_name

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


def get_h5_dataloader(
    h5_path: str,
    batch_size: int,
    lr_key: str = "TFM",
    hr_key: str = "hr",
    lr_dataset_name: Optional[str] = None,
    hr_dataset_name: Optional[str] = None,
    transpose_lr: bool = False,
    transpose_hr: bool = False,
    use_tfm_channels: bool = False,
    coord_range: Optional[tuple] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """创建 HDF5 数据集的 DataLoader
    
    Args:
        use_tfm_channels: 是否使用 TFM 的 I, X, Y 三通道模式
        coord_range: X, Y 坐标的归一化范围，默认 (-1, 1)
    """
    dataset = H5PairedDataset(
        h5_path=h5_path,
        lr_key=lr_key,
        hr_key=hr_key,
        lr_dataset_name=lr_dataset_name,
        hr_dataset_name=hr_dataset_name,
        transpose_lr=transpose_lr,
        transpose_hr=transpose_hr,
        use_tfm_channels=use_tfm_channels,
        coord_range=coord_range,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
