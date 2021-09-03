---
title: FCOS：Fully Convolutional One-Stage Object Detection
date: 2021-09-02 00:00:00
tags:
    - anchor free
categories: 目标检测
---

最近看了不少关于 anchor-free 目标检测的文章，弄得有些审美疲劳了。今天带来一篇我觉得非常有创意的文章，[FCOS：Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf). <strong>这篇文章用图像分割的思想去解决 detection 问题，并提出可以用 FPN 的思路来解决重叠目标的 bad case。</strong>从实验结果来看，FCOS 能够与主流的 anchor-base 检测算法相媲美，达到 SOTA 的效果。

<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902144334.png">
</p>

<!-- more -->

毫无疑问，anchor-free 系列的文章一上来就是<strong>先痛骂一顿 anchor-based 的缺点</strong>。在作者看来，anchor-based 虽然能带来很大的准确率提升，但是也有不可避免的以下几个问题：

- 模型的性能对 anchor 的尺寸、宽高比和数量会比较敏感，例如 RetinaNet 在不同的超参下在 COCO 数据集上的 AP 会有 4% 的波动；
- anchor 的尺寸和宽高都是固定的，如果目标的尺寸范围波动比较大，这会比较比较难预测；
- 为了提高召回率，anchor 的数量通常很大，并且容易出现样本不均衡的情况；
- 在训练阶段还得涉及复杂的 iou 计算，这使得显存的占用也比较大。

## 1. 相关工作
### 1.1 anchor-based detector
大多数 anchor-based 的检测算法都是基于传统的滑动窗口思想，或者用的是 Fast-RCNN 那套候选框机制：<strong>预先定义好滑动窗口或者候选框，然后对它们分类成正样本和负样本，对于正样本还需要额外回归出精确的位置和尺寸。</strong>即使后来的一些有名的单阶段检测算法如 SSD 和 YOLO 系列，也是在 Faster-RCNN 的 RPN 基础上发展的。

但是这种 anchor 会带来大量的超参数，影响到算法的性能，比如 iou 阈值会影响到正负样本标注。因此在训练一个 anchor-based detector 时，我们需要对 anchor 的超参数进行精细的调试。

### 1.2 anchor-free detector
在 anchor-based 机制的痛点刺激下，最近也涌现出了一些不错的 anchor-free 检测算法。比较早的可能要属 <strong>YOLOv1 了，它是直接回归出目标的中心和尺寸。但是因为它只用了中心点来预测框，这使得它的召回率很低。</strong>在那个 RPN 召唤的时代，Joseph Redmon 在 YOLOv2 中立马采用了 RPN 的 anchor 机制，并提出 anchor 的先验尺寸可以通过聚类得到。

<strong>CornerNet</strong> 是最近提出的一个不错的 anchor-free 算法，它是通过检测一对角点来实现目标检测。但是它<strong>在后处理阶段引入了一套额外的 corners grouping 过程</strong>，看起来也不是最好的。

## 2. FCOS 算法介绍
### 2.1 基本思路
近年来全卷积神经网络（ Fully Convolutional Network，FCN）已经在语义分割、深度估计和关键点检测等领域取得了巨大的突破，这不禁令人想到：<strong>我们能不能像 FCN 语义分割那样在 pixel 级别上解决目标检测问题？</strong> 这样一来这些基本的视觉任务几乎就可以在同一套框架内完成了，FCOS 证实了这一猜想。

当像素（x, y）落入任何一个 ground-truth 框时被认为是正样本，并将 ground-truth 框的类别 c 设置为该像素的类别，否则就是 0. 此外还有一个 4D 向量作为位置回归目标，里面每个元素值分别为该像素到 bbox 四条边的距离（如上图所示）。

<p align="center">
    <img width="33%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902195049.png">
</p>

当像素落在多个 ground-truth 框中时，直接选择面积最小的那个作为回归目标。<strong>相对于 anchor-based 的IOU判断，FCOS 能生成更多的正样本来训练 detector 并且也没引入额外的超参。</strong>

### 2.2 FCOS 网络
目标检测里一个比较棘手的问题就是对于重叠目标的识别效果不太好：一个原因是 anchor 对于这类目标的响应具有一定的模糊性，还有另一个重要的原因就是较小的重叠目标的特征在深度下采样时会消失。<strong>FCOS 提出了利用 FPN 模块对不同 size 的目标使用不同分辨率的 feature map 进行预测，并且对 pixel 所属的 feature map level 进行了指定。</strong>比如某个 pixel 满足 max(l∗, t∗, r∗, b∗) > mi 或 max(l∗, t∗, r∗, b∗) < m(i−1) 则认为该 pixel 不属于第 i 层 feature map了，其中 m 是 feature map 的最大回归长度。（这一点其实和 anchor-based 里不同分辨率的 feature map 设置不同大小的 anchor 一样）

<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902211107.png">
</p>

FCOS 结构如下图所示，它最终输出 80D 分类标签向量 p 和 4D box 坐标向量 t = (l, t, r, b)，训练 C 个二分类器而不是多分类器，并在最后特征后面分别接 4 个卷积层用于分类和定位分支。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902202015.png">
</p>

对于每个最终输出 feature map （相对于原图的缩放倍数为 s）上的某个点（x, y），我们可以将它映射回原图得到位置：

<p align="center">
    <img width="25%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902202910.png">
</p>

此外，为了保证预测的长度为正数，论文使用了 exp(x) 函数从而保证任何实数的映射范围在（0，∞).最终的整个训练损失函数如下所示：

<p align="center">
    <img width="55%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210902203920.png">
</p>

L_cls 为 focal loss 分类损失，L_reg 为 UnitBox 里的 iou 损失，N_pos 为正样本数量，λ 为平衡权重. 

### 2.3 Center-ness
在使用多尺度预测后，FCOS 依然和主流的 anchor-based 算法存在一定的差距，这主要是来源于低质量的预测框，这些框大多是由距离目标中心点比较远的像素所产生。因此，论文提出新的独立分支来预测像素的 <strong>center-ness，用来评估像素与目标中心点的距离</strong>：

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210903111701.png">
</p>

center-ness 值的范围为（0， 1）：当像素位于 ground-truth 框边上时，它距离中心位置最远，此时值为 0；当像素位于中心时，此时值为 1.

<p align="center">
    <img width="30%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FCOS-Fully-Convolutional-One-Stage-Object-Detection-20210903112233.png">
</p>

在测试阶段，最终的分数是将分类的 score 分数与 center-ness 进行加权相乘，从而降低那些低质量预测框的分数。














## 参考文献
- [[1] FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)
- [[2] https://github.com/tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)







