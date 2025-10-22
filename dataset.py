import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Lambda, ToTensor
from PIL import Image
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import os
import random

# 多进程加速需要，lambda会阻塞多进程
class ConvertToRGB(object):
    def __call__(self, image):
        return image.convert('RGB')
    
class Identity(nn.Module):
    def forward(self, x): 
        return x


class PairedTransform:
    """
    配对图像的数据增强类
    确保LR和HR图像使用相同的随机变换参数，保持对齐
    """
    def __init__(self, image_size, channels=1, augment=False):
        """
        Args:
            image_size: 目标图像尺寸
            channels: 通道数（1=灰度，3=RGB）
            augment: 是否使用数据增强（训练时True，评估时False）
        """
        self.image_size = image_size
        self.channels = channels
        self.augment = augment
    
    def _convert_color(self, image):
        """颜色转换（避免使用lambda，确保可以被pickle）"""
        if self.channels == 1:
            return image  # 单通道，保持原样
        else:
            return image.convert('RGB')  # 转RGB
    
    def __call__(self, lr_image, hr_image):
        """
        对LR和HR图像应用相同的变换
        
        Args:
            lr_image: PIL Image, 低分辨率图像
            hr_image: PIL Image, 高分辨率图像
        
        Returns:
            lr_tensor: torch.Tensor, 变换后的LR图像
            hr_tensor: torch.Tensor, 变换后的HR图像
        """
        # 1. 确定性变换：颜色转换
        lr_image = self._convert_color(lr_image)
        hr_image = self._convert_color(hr_image)
        
        # 2. 确定性变换：Resize和CenterCrop
        lr_image = TF.resize(lr_image, self.image_size, interpolation=Image.BICUBIC)
        hr_image = TF.resize(hr_image, self.image_size, interpolation=Image.BICUBIC)
        
        lr_image = TF.center_crop(lr_image, self.image_size)
        hr_image = TF.center_crop(hr_image, self.image_size)
        
        # 3. 随机增强变换（训练时）
        if self.augment:
            # 一次性采样所有随机参数
            # 水平翻转
            if random.random() > 0.5:
                lr_image = TF.hflip(lr_image)
                hr_image = TF.hflip(hr_image)
            
            # 随机平移（最多5%）
            max_translate = int(0.05 * self.image_size)
            translate_x = random.randint(-max_translate, max_translate)
            translate_y = random.randint(-max_translate, max_translate)
            lr_image = TF.affine(lr_image, angle=0, translate=(translate_x, translate_y),
                                scale=1.0, shear=0, fill=0)
            hr_image = TF.affine(hr_image, angle=0, translate=(translate_x, translate_y),
                                scale=1.0, shear=0, fill=0)
        
        # 4. 转换为Tensor并归一化到[-1, 1]
        lr_tensor = TF.to_tensor(lr_image)  # [0, 1]
        hr_tensor = TF.to_tensor(hr_image)  # [0, 1]
        
        lr_tensor = TF.normalize(lr_tensor, mean=[0.5]*self.channels, std=[0.5]*self.channels)  # [-1, 1]
        hr_tensor = TF.normalize(hr_tensor, mean=[0.5]*self.channels, std=[0.5]*self.channels)  # [-1, 1]
        
        return lr_tensor, hr_tensor
    
class PairedImageDataset(Dataset):
    """
    自定义数据集，用于加载成对的原始图像和超分辨率图像。
    """
    def __init__(self, lr_dir: str, hr_dir: str, transform=None):
        """
        初始化数据集。
        Args:
            lr_dir (str): 低分辨率图像（origin）的文件夹路径。
            hr_dir (str): 高分辨率图像（hr）的文件夹路径。
            transform (callable, optional): 应用于图像的转换操作。
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        
        # 获取所有低分辨率图像的文件名列表，并排序以确保顺序一致
        self.image_files = sorted(os.listdir(lr_dir))

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        根据索引idx获取一对图像。
        """
        # 获取文件名
        img_name = self.image_files[idx]
        
        # 构建低分辨率和高分辨率图像的完整路径
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        
        # 使用Pillow库打开图像
        lr_image = Image.open(lr_path)
        hr_image = Image.open(hr_path)
        
        # 🔥 配对变换：确保LR和HR使用相同的随机参数
        if self.transform:
            lr_image, hr_image = self.transform(lr_image, hr_image)

        return lr_image, hr_image, img_name

# 直接运行会下载默认数据集MNIST
# 函数 get_dataloader 为默认加载MNIST数据集
def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp.jpg')
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 这里使用自定义数据集，你可以在文件顶部设置三项超参
IMAGE_ROOT = 'D:/深度学习框架/DDPM/butterfly_images_for_training'  # 改成你的本地目录
IMAGE_SIZE = 400                                # 统一缩放/裁剪到这个尺寸
CHANNELS   = 1                                  # 灰度=1，彩色=3


def set_image_shape(image_size: int, channels: int) -> None:
    global IMAGE_SIZE, CHANNELS
    IMAGE_SIZE = image_size
    CHANNELS = channels


def _build_transform(image_size=IMAGE_SIZE, channels=CHANNELS, augment=False):
    """
    构建图像变换管道
    Args:
        image_size: 目标图像尺寸
        channels: 通道数（1=灰度，3=RGB）
        augment: 是否使用数据增强（训练时True，评估时False）
    """
    # 先把 PIL 图像转到指定通道数，再做尺寸与归一化（[-1,1]）
    color_tf = (
        Identity()
        if channels == 1 else
        ConvertToRGB()
    )
    
    # 基础变换（始终应用）
    base_transforms = [
        color_tf,
        T.Resize(image_size, interpolation=Image.BICUBIC),
        T.CenterCrop(image_size),
    ]
    
    # 数据增强变换（仅训练时）
    if augment:
        augmentation_transforms = [
            T.RandomHorizontalFlip(0.5),    # 50%概率水平翻转
            T.RandomAffine(                 # 随机平移
                degrees=0,                  # 不额外旋转
                translate=(0.05, 0.05),     # 最多平移5%（约5像素）
                fill=0                      # 填充黑色（背景色）
            ),
        ]
        base_transforms.extend(augmentation_transforms)
    
    # 归一化变换（始终应用）
    base_transforms.extend([
        T.ToTensor(),                                  # [0,1]
        T.Normalize(mean=[0.5]*channels, std=[0.5]*channels)  # -> [-1,1]
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
        prefetch_factor=4
    )

def get_paired_dataloader(batch_size: int,
                          lr_root: str,
                          hr_root: str,
                          image_size: int = IMAGE_SIZE,
                          channels: int = CHANNELS,
                          num_workers: int = 16,  # 在Windows上建议设为0或1，Linux上可以更高
                          augment: bool = False): # 是否使用数据增强（训练时True，评估时False）
    """
    基于自定义的PairedImageDataset创建并返回一个DataLoader。
    Args:
        batch_size (int): 批处理大小。
        lr_root (str): 低分辨率图像文件夹的根目录。
        hr_root (str): 高分辨率图像文件夹的根目录。
        augment (bool): 是否使用数据增强（训练时True，评估时False）
        ... 其他参数
    """
    set_image_shape(image_size, channels)
    
    # 🔥 使用配对变换，确保LR和HR对齐
    transform = PairedTransform(image_size, channels, augment=augment)
    
    # 使用我们自定义的Dataset
    dataset = PairedImageDataset(
        lr_dir=lr_root, 
        hr_dir=hr_root, 
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2
    )

def get_img_shape():
    # 因为上面的 transform 强制成固定尺寸与通道，直接返回即可
    return (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

if __name__ == '__main__':
    import os
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
