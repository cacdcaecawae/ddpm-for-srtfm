import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import Compose, InterpolationMode, Lambda, ToTensor


class ConvertToRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class Identity(nn.Module):
    def forward(self, x):  # type: ignore[override]
        return x


class PairedTransform:
    """
    对配对的 LR/HR 图像应用一致的随机增强，支持 PIL 或张量输入。
    """

    def __init__(self, image_size: int, channels: int = 1, augment: bool = False):
        self.image_size = image_size
        self.channels = channels
        self.augment = augment

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            desired_mode = "L" if self.channels == 1 else "RGB"
            if image.mode != desired_mode:
                image = image.convert(desired_mode)
            tensor = TF.to_tensor(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                if tensor.shape[0] in (1, 3):
                    pass
                elif tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1)
                else:
                    raise ValueError("Unsupported tensor shape for image data.")
            else:
                raise ValueError("Image tensor must have 2 or 3 dimensions.")
            if tensor.max() > 1.0 or tensor.min() < 0.0:
                tensor = tensor.clamp(0.0, 1.0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return self._match_channels(tensor)

    def _match_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError("Tensor input should have shape [C, H, W].")
        if self.channels == 1:
            if tensor.shape[0] == 1:
                return tensor
            if tensor.shape[0] == 3:
                gray = (
                    0.2989 * tensor[0]
                    + 0.5870 * tensor[1]
                    + 0.1140 * tensor[2]
                ).unsqueeze(0)
                return gray
            raise ValueError("Cannot convert tensor with unexpected channels to grayscale.")
        if self.channels == 3:
            if tensor.shape[0] == 3:
                return tensor
            if tensor.shape[0] == 1:
                return tensor.repeat(3, 1, 1)
            raise ValueError("Cannot convert tensor with unexpected channels to RGB.")
        raise ValueError(f"Unsupported target channels: {self.channels}")

    def __call__(self, lr_image: Any, hr_image: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_tensor = self._to_tensor(lr_image)
        hr_tensor = self._to_tensor(hr_image)

        lr_tensor = TF.resize(
            lr_tensor,
            self.image_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        hr_tensor = TF.resize(
            hr_tensor,
            self.image_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        lr_tensor = TF.center_crop(lr_tensor, self.image_size)
        hr_tensor = TF.center_crop(hr_tensor, self.image_size)

        if self.augment:
            if random.random() > 0.5:
                lr_tensor = TF.hflip(lr_tensor)
                hr_tensor = TF.hflip(hr_tensor)

            max_translate = int(0.05 * self.image_size)
            translate_x = random.randint(-max_translate, max_translate)
            translate_y = random.randint(-max_translate, max_translate)
            affine_kwargs = {
                "angle": 0.0,
                "translate": (translate_x, translate_y),
                "scale": 1.0,
                "shear": 0.0,
                "interpolation": InterpolationMode.BILINEAR,
                "fill": 0.0,
            }
            lr_tensor = TF.affine(lr_tensor, **affine_kwargs)
            hr_tensor = TF.affine(hr_tensor, **affine_kwargs)

        mean = [0.5] * self.channels
        std = [0.5] * self.channels
        lr_tensor = TF.normalize(lr_tensor, mean=mean, std=std)
        hr_tensor = TF.normalize(hr_tensor, mean=mean, std=std)

        return lr_tensor, hr_tensor


class PairedImageDataset(Dataset):
    """
    普通文件夹版本的配对数据集。
    """

    def __init__(self, lr_dir: str, hr_dir: str, transform: Optional[PairedTransform] = None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(lr_dir))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)

        with Image.open(lr_path) as lr_img:
            lr_image = lr_img.copy()
        with Image.open(hr_path) as hr_img:
            hr_image = hr_img.copy()

        if self.transform:
            lr_image, hr_image = self.transform(lr_image, hr_image)

        return lr_image, hr_image, img_name


class PairedH5Dataset(Dataset):
    """
    针对根下包含两个分组（如 TFM / hr），每个分组按编号存放样本的 HDF5 数据集。
    - lr_key / hr_key 可以指向分组或直接指向数据集；
    - 若分组下包含多个数据集，可通过 lr_dataset_name / hr_dataset_name 指定目标条目；
    - 默认为交集编号进行配对。
    """

    def __init__(
        self,
        h5_path: str,
        lr_key: str = "TFM",
        hr_key: str = "hr",
        transform: Optional[PairedTransform] = None,
        value_range: Optional[Tuple[float, float]] = None,
        lr_dataset_name: Optional[str] = None,
        hr_dataset_name: Optional[str] = None,
    ):
        self.h5_path = str(h5_path)
        self.lr_key = lr_key
        self.hr_key = hr_key
        self.transform = transform
        self.value_range = value_range
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name

        self._file: Optional[h5py.File] = None
        self._sample_names: Optional[List[str]] = None
        self._lr_dataset_paths: Optional[List[str]] = None
        self._hr_dataset_paths: Optional[List[str]] = None
        self._flat_mode: bool = False
        self._flat_length: int = 0
        self._lr_flat_path: Optional[str] = None
        self._hr_flat_path: Optional[str] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    @staticmethod
    def _resolve_root(file: h5py.File, key: str) -> Union[h5py.Group, h5py.Dataset]:
        if key not in file:
            available = ", ".join(sorted(file.keys()))
            raise KeyError(f"HDF5 文件中不存在键 '{key}'，可用键包括：{available}")
        return file[key]

    @staticmethod
    def _list_members(node: Union[h5py.Group, h5py.Dataset]) -> List[str]:
        if isinstance(node, h5py.Dataset):
            return []
        return sorted(name for name in node.keys())

    def _ensure_structure(self) -> None:
        file = self._ensure_file()
        if self._sample_names is not None:
            return

        lr_root = self._resolve_root(file, self.lr_key)
        hr_root = self._resolve_root(file, self.hr_key)

        if isinstance(lr_root, h5py.Dataset) and isinstance(hr_root, h5py.Dataset):
            length = lr_root.shape[0]
            if hr_root.shape[0] != length:
                raise ValueError("LR 和 HR 数据集长度不一致。")
            self._flat_mode = True
            self._flat_length = int(length)
            self._lr_flat_path = lr_root.name
            self._hr_flat_path = hr_root.name
            self._sample_names = [f"{idx:06d}" for idx in range(length)]
            return

        if not isinstance(lr_root, h5py.Group) or not isinstance(hr_root, h5py.Group):
            raise TypeError("当使用嵌套结构时，lr_key 和 hr_key 必须指向分组。")

        lr_members = self._list_members(lr_root)
        hr_members = self._list_members(hr_root)
        shared = sorted(set(lr_members) & set(hr_members))
        if not shared:
            raise ValueError("在 LR 和 HR 分组下没有找到共同的样本编号。")

        lr_paths: List[str] = []
        hr_paths: List[str] = []
        for name in shared:
            lr_paths.append(
                self._dataset_path_for_sample(
                    lr_root[name], self.lr_dataset_name, "LR", name
                )
            )
            hr_paths.append(
                self._dataset_path_for_sample(
                    hr_root[name], self.hr_dataset_name, "HR", name
                )
            )
        self._sample_names = shared
        self._lr_dataset_paths = lr_paths
        self._hr_dataset_paths = hr_paths

    @staticmethod
    def _dataset_path_for_sample(
        node: Union[h5py.Group, h5py.Dataset],
        dataset_name: Optional[str],
        role: str,
        sample_name: str,
    ) -> str:
        if isinstance(node, h5py.Dataset):
            return node.name
        if dataset_name:
            if dataset_name not in node:
                available = ", ".join(sorted(node.keys()))
                raise KeyError(
                    f"{role} 样本 '{sample_name}' 下找不到数据集 '{dataset_name}'，可用成员：{available}"
                )
            dataset = node[dataset_name]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{role} 样本 '{sample_name}' 中 '{dataset_name}' 不是 dataset。")
            return dataset.name

        dataset_candidates: List[str] = []
        for child in node.values():
            if isinstance(child, h5py.Dataset):
                dataset_candidates.append(child.name)
        if len(dataset_candidates) == 1:
            return dataset_candidates[0]
        if not dataset_candidates:
            raise TypeError(f"{role} 样本 '{sample_name}' 下没有 dataset。")
        available = ", ".join(sorted(dataset_candidates))
        raise TypeError(
            f"{role} 样本 '{sample_name}' 下有多个 dataset ({available})，"
            f"请在配置中指定 {role.lower()}_dataset_name。"
        )

    @staticmethod
    def _ensure_chw(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return array[None, ...]
        if array.ndim == 3:
            if array.shape[0] in (1, 3):
                return array
            if array.shape[-1] in (1, 3):
                return np.moveaxis(array, -1, 0)
        raise ValueError(f"Unsupported array shape {array.shape}, expected HxW or CxHxW.")

    def _array_to_tensor(self, array: np.ndarray, is_lr: bool = False) -> torch.Tensor:
        """
        将 numpy 数组转换为 tensor
        Args:
            array: 输入数组
            is_lr: 是否是 LR 数据，如果是则应用 value_range 转换
        """
        tensor = torch.from_numpy(self._ensure_chw(array)).float()
        if is_lr and self.value_range is not None:
            # 只对 LR 应用 value_range 线性变换
            low, high = self.value_range
            if high <= low:
                raise ValueError("value_range must have high > low.")
            tensor = tensor.clamp(low, high)
            tensor = (tensor - low) / (high - low)
        elif np.issubdtype(array.dtype, np.integer):
            tensor = tensor / 255.0
        else:
            tensor = tensor.clamp(0.0, 1.0)
        return tensor

    def __len__(self) -> int:
        self._ensure_structure()
        if self._flat_mode:
            return self._flat_length
        assert self._sample_names is not None
        return len(self._sample_names)

    def __getitem__(self, idx: int):
        self._ensure_structure()
        file = self._ensure_file()

        if self._flat_mode:
            assert self._lr_flat_path and self._hr_flat_path
            lr_array = np.asarray(file[self._lr_flat_path][idx])
            hr_array = np.asarray(file[self._hr_flat_path][idx])
            sample_name = f"{idx:06d}"
        else:
            assert (
                self._sample_names is not None
                and self._lr_dataset_paths is not None
                and self._hr_dataset_paths is not None
            )
            sample_name = self._sample_names[idx]
            lr_path = self._lr_dataset_paths[idx]
            hr_path = self._hr_dataset_paths[idx]
            lr_array = np.asarray(file[lr_path])
            hr_array = np.asarray(file[hr_path])

        lr_tensor = self._array_to_tensor(lr_array, is_lr=True)  # LR 应用 value_range
        hr_tensor = self._array_to_tensor(hr_array, is_lr=False)  # HR 直接转换

        if self.transform:
            lr_tensor, hr_tensor = self.transform(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor, sample_name


def download_dataset():
    mnist = torchvision.datasets.MNIST(root="./data/mnist", download=True)
    print("length of MNIST", len(mnist))
    idx = 4
    img, label = mnist[idx]
    print(img)
    print(label)
    Path("work_dirs").mkdir(parents=True, exist_ok=True)
    img.save("work_dirs/tmp.jpg")
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root="./data/mnist", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


IMAGE_ROOT = "D:/deep_learning/DDPM/butterfly_images_for_training"
IMAGE_SIZE = 400
CHANNELS = 1


def set_image_shape(image_size: int, channels: int) -> None:
    global IMAGE_SIZE, CHANNELS
    IMAGE_SIZE = image_size
    CHANNELS = channels


def _build_transform(image_size=IMAGE_SIZE, channels=CHANNELS, augment=False):
    color_tf = Identity() if channels == 1 else ConvertToRGB()
    base_transforms = [
        color_tf,
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
    ]

    if augment:
        base_transforms.extend([
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                fill=0,
            ),
        ])

    base_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ])

    return T.Compose(base_transforms)


def get_dataloader1(batch_size: int,
                    root: str = IMAGE_ROOT,
                    image_size: int = IMAGE_SIZE,
                    channels: int = CHANNELS,
                    num_workers: int = 16):
    set_image_shape(image_size, channels)
    transform = _build_transform(image_size, channels)
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2,
    )


def get_paired_dataloader(
    batch_size: int,
    lr_root: Optional[str] = None,
    hr_root: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
    channels: int = CHANNELS,
    num_workers: int = 16,
    augment: bool = False,
    h5_path: Optional[str] = None,
    h5_lr_key: str = "lr",
    h5_hr_key: str = "hr",
    h5_lr_dataset: Optional[str] = None,
    h5_hr_dataset: Optional[str] = None,
    value_range: Optional[Tuple[float, float]] = None,
):
    set_image_shape(image_size, channels)
    transform = PairedTransform(image_size, channels, augment=augment)

    if h5_path:
        dataset = PairedH5Dataset(
            h5_path=h5_path,
            lr_key=h5_lr_key,
            hr_key=h5_hr_key,
            transform=transform,
            value_range=value_range,
            lr_dataset_name=h5_lr_dataset,
            hr_dataset_name=h5_hr_dataset,
        )
    else:
        if lr_root is None or hr_root is None:
            raise ValueError("lr_root and hr_root must be provided when h5_path is not set.")
        dataset = PairedImageDataset(
            lr_dir=lr_root,
            hr_dir=hr_root,
            transform=transform,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2,
    )


def get_img_shape():
    return CHANNELS, IMAGE_SIZE, IMAGE_SIZE


if __name__ == "__main__":
    download_dataset()
