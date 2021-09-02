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

### 2.1 基本思想

### 2.2 FCOS 网络

### 2.3 多尺度特征


### 2.4 Center-ness










## 参考文献
- [[1] FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)









