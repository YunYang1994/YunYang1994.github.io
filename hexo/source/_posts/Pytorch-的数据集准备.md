---
title: Pytorch 的数据集准备
date: 2018-06-11 12:05:19
tags:
categories: 深度学习
---

深度学习的绝大部分工作都是在准备数据集，<strong>Pytorch</strong> 提供了很多工具使数据加载变得更简单。在本节内容中，我们来看看是如何利用 <strong>torch.utils.data.DataLoader</strong> 加载数据集的。

首先需要 import 一些必要的库：

```python
import os
import torch
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
```

然后从[这里](https://download.pytorch.org/tutorial/faces.zip)下载一个名为 <strong>faces</strong> 的文件夹，该文件夹里包含了一些 <strong>68 个特征点（part_0 ~ part_67)</strong> 的人脸图片和 `face_landmarks.csv` 

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/Pytorch-的数据集准备-20210508212935.jpg">
</p>

<!-- more -->

```python
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')
landmarks_frame.head()
```

## 1. Dataset class

`torch.utils.data.Dataset` 是一个抽象的类，我们构造的数据集需要继承它得到，并且重载下面 2 个成员函数：

- `__len__` 函数，通过`len(dataset)`返回数据集大小；
- `__getitem__` 函数，通过索引`dataset[i]`而得到一个样本。

```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = skimage.io.imread(img_name)
        
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample
```

现在我们可以对 `FaceLandmarksDataset` 类构建一个实例 `face_dataset`。其中每个样本都是一个字典，分别是 `'image'` 和 `'landmarks'` 。我们可以索引第 65 个样本将它们的数组形状打印出来。

```python
face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/')
sample = face_dataset[65]             # 取第 65 个样本
print(sample['image'].shape)          # (160, 160, 3)
print(sample['landmarks'].shape)      # (68, 2)
```

## 2. Transforms

上述过程完成了对人脸图片和 65 个特征点的读取，接下来需要对它们进行一些预处理操作。本文将介绍 3 种 Transforms 操作：

- Rescale，对图片进行 `resize` 操作
- RandomCrop，随机地裁剪图片
- ToTensor，将 numpy 的 `array` 类型转变为 torch 的 `tensor` 类型

### 2.1 Rescale

```python
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        img = skimage.transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}
```
然后我们对人脸图片的尺寸 resize 到 (256, 256)

```python
rescale_transform = Rescale((256, 256))
rescale_sample = rescale_transform(sample)
print(rescale_sample['image'].shape)           # (256,  256, 3)
```

<p align="center">
    <img width="27%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/Pytorch-的数据集准备-20210508212935.jpg">
</p>

### 2.2 RandomCrop

```python
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple): Desired output size is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
```
然后我们对人脸图片进行随机裁剪，裁剪的尺寸大小为 (128, 128)

```python
crop_transform = RandomCrop((128, 128))
crop_sample = crop_transform(sample)
print(crop_sample['image'].shape)           # (128,  128, 3)
```

<p align="center">
    <img width="16%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/Pytorch-的数据集准备-20210508212958.jpg">
</p>

### 2.3 ToTensor
现在需要使用 `torch.from_numpy` 函数将数据转化成 `tensor`，在进行这项操作之前，考虑到 torch 的图片输入顺序为 `[C, H, W]`，而 numpy 的图片顺序为 `[H, W, C]`，因此需要通过 transpose 转化。

```python
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```
现在可以尝试改变图片的通道顺序，并转化成 tensor

```python
tensor_transform = ToTensor()
tensor_sample = tensor_transform(sample)
print(tensor_sample['image'].shape)           # (3,  160, 160)
```

## 3. Compose transforms
最后我们可以通过 `transforms.Compose` 函数将这些操作串联起来, 并将它传递 `FaceLandmarksDataset` 类的 `transform` 参数：

```python
composed_transform = transforms.Compose([Rescale((256, 256)),
                               RandomCrop((224, 224)),
                               ToTensor()])
face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/',
                                    transform=composed_transform)
```

## 4. Iterating through the dataset

```python
dataloader = DataLoader(face_dataset, batch_size=32, 
                                         shuffle=True, num_workers=4)
for batch_samples in dataloader:
    print("=> ", batch_samples["image"].shape, batch_samples['landmarks'].shape)
```
打印出来的结果为：

```
=>  torch.Size([32, 3, 224, 224]) torch.Size([32, 68, 2])
=>  torch.Size([32, 3, 224, 224]) torch.Size([32, 68, 2])
=>  torch.Size([5, 3, 224, 224]) torch.Size([5, 68, 2])
```

一共有 69 张人脸图片，分成了 3 个 `batch` （32 + 32 +5）进行吞吐。

