---
title: 基于卷积神经网络的 2D-to-3D 视频转换
date: 2019-12-10 12:04:21
tags:
    - 深度估计
categories: 立体视觉
---

目前制作 3D 电影的方法有两种：一种是直接用昂贵的立体相机设备进行拍摄，这种制作成本非常庞大。另一种则是通过图像处理技术将 2D 电影转化成 3D 格式，这种转换处理通常依赖于“深度艺术家”，他们手工地为每一帧创造深度图，然后利用标准的基于深度图像的渲染算法将与原始图像相结合，得到一个立体的图像对，这需要大量的人力成本。现在来说，每年只有 20 左右部新的 3D 电影发行。

![Deep3D 网络](https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/基于卷积神经网络的-2D-to-3D-视频转-20210508231215.png)

<!-- more -->

在这样强烈需求的工业背景下，这篇文章的目的虽然是为了解决如何利用神经网络将 2D 电影转化成具有立体感的 3D 视频的问题，并且不需要人力来标注图片的深度信息。但是它提出的方法太新颖(很多论文都引用了，可见影响力)，所以也把它拎出来讲。

## 1. 网络结构

作者的网络如上图所示：双目图片的左图作为模型的输入，每经过一次卷积和池化层后都会有两个分支：分支 1 会进行下一个卷积池化层进行特征提取，而分支 2 会进入反卷积层进行上采样得到一个与原图分辨率一致的视差图 (disparity map)，如此反复经过 5 层循环，得到 5 个视差图。最终作者会将这 5 个视差图相加，然后再经过一层卷积层并使用 softmax 激活函数，最后会输出一个与原图分辨率一致的视差概率分布 (probalistic disparity map)。

其实该网络的结构设计和其他语义分割类型的网络大同小异，这里没什么讲的。关键是它的损失函数设计以及 image-to-image 的方法，揭开了深度估计无监督学习的序幕。所以让我们直奔主题，给你一张左图和视差图，你如何去重构出右图？

## 2. 重构右图

我们的直觉做法是将左视角点向右平移视差 D 个单位，然后便得到了右视角点。由于受极线约束，因此计算复杂度为 o(n)。但是这个方法在神经网络里无法进行反向传播，因为它对视差 D 是不可导的，因此我们无法训练。针对这个问题，作者引入了视差的概率分布对网络进行优化。利用左视角点和视差概率分布对右视角点进行重构的过程如下公式所示：

<p align="center">
    <img width="22%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/基于卷积神经网络的-2D-to-3D-视频转-20210508231226.jpg">
</p>

- `O_{i, j}` 表示在图片坐标 (i, j) 上重构的右视角点
- `I_{i, j}^{d}` 表示左视角点 (i, j) 平移 d 个位置后得到的右视角
- `D_{i, j}^{d}` 表示左视角点 (i, j) 的视差概率分布在视差 d 时的概率值

## 3. 代码实践

上述公式有点晦涩难懂，我琢磨了半天，写了个小程序进行实践：假如我们现在有一对分辨率为 200x200 的双目图片，整张图片上的像素视差都是 20。为了感受视差的偏移性质，我们在图片的中间区域设置了一块 10x10 的白点。可以看到从左图到右图，小白点很明显地移动了一小段距离，这就是视差造成的。

<p align="center">
    <img width="37%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/基于卷积神经网络的-2D-to-3D-视频转-20210508231223.png">
</p>

我们假定整张图片的最大视差值为 30， 那么就需要划分 0,1,...,30 共 31 个等级。因此图片的视差概率分布的形状为 [200， 200， 31]，由于真实的视差值为 20， 因此该等级属于 onehot 状态，接着左图上每个像素点在每个等级 i 上都会向右平移 i 个单位，这样一来我们便总共得到了 31 张图片， 程序里用 shift_images 表示，最后再将它与视差的概率分布相乘并求和便得到重构的右图。


```python
import cv2
import numpy as np

l_img = np.zeros([200, 200])
l_img[95:105, 95:105] = 255

r_img = np.zeros([200, 200])
r_img[95:105, 115:125] = 255

cv2.imwrite("l_img.jpg", l_img)
cv2.imwrite("r_img.jpg", r_img)

# 假设整张图片的视差都是20
gt_disp = np.ones([200, 200]) * 20

# construct disparity map
max_disp = 30
prob_disp = np.zeros([200, 200, max_disp])
prob_disp[:, :, 20] = 1.0

shift_images = np.zeros(shape=[200, 200, max_disp])
for disp in range(max_disp):
    shift_images[:, :, disp] = np.roll(l_img, disp, axis=1)

pred_r_img = np.sum(shift_images * prob_disp, axis=2)
cv2.imwrite("pred_r_img.jpg", pred_r_img)
print("reconstruction loss: ", np.sum(pred_r_img - r_img)) # 0.0
```

由于我们给的是真实的视差概率分布，因此重构损失(reconstruction loss)的值为0. 反过来：如果重构损失不为 0， 那么神经网络将会朝着预测正确的视差概率分布去优化。最后我们将预测出来的右图和真实的右图进行了对比，结果一致。

<p align="center">
    <img width="37%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/基于卷积神经网络的-2D-to-3D-视频转-20210508231230.png">
</p>

总结： 这篇文章的新颖之处在于，通过 image-to-image 训练的方式，打开了深度估计网络通往无监督训练的大门。

## 参考文献

- [1] Junyuan Xie, Ross Girshick, Ali Farhadi. [Deep3D: Fully Automatic 2D-to-3D Video Conversion with Deep Convolutional Neural Networks](https://arxiv.org/abs/1604.03650), CVPR 2016
- [2] Deep3D: [Automatic 2D-to-3D Video Conversion with CNNs](https://github.com/piiswrong/deep3d)


