---
title: 老生常谈 Focal Loss —— 解决正负样本不均衡问题
date: 2021-09-03 16:40:57
categories:
	- 深度学习
---

最近看的一些 anchor-free 目标检测算法都普遍用到了 Focal Loss，今天就来老生常谈重新聊聊它吧！

<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/解决正负样本不均衡问题——Focal-Loss-20210903163748.png">
</p>

<!-- more -->

在讲 Focal Loss 之前，咱们先简单回顾一下交叉熵损失（cross entropy loss）函数。

## 1. 交叉熵损失
在物理学中，“熵”被用来表示热力学系统所呈现的无序程度。香农将这一概念引入信息论领域，提出了“信息熵”概念，通过对数函数来测量信息的不确定性。

交叉熵（cross entropy）是信息论中的重要概念，主要用来度量两个概率分布间的差异。假定 𝑝 和 𝑞 是数据 𝑥 的两个概率分布，通过 𝑞 来表示 𝑝 的交叉熵可如下计算：

<p align="center">
    <img width="32%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/解决正负样本不均衡问题——Focal-Loss-20210903170506.png">
</p>

<strong>交叉熵刻画了两个概率分布 𝑝  和 𝑞 之间的距离。</strong>根据公式不难理解，如果交叉熵越小，那么两个概率分布 𝑝  和 𝑞 越接近。

<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/解决正负样本不均衡问题——Focal-Loss-20210903171510.png">
</p>

如上图所示，在神经网络中，<strong>我们通常是利用 softmax 层输出一个多分类的预测概率分布，然后与真实概率分布计算交叉熵损失。</strong>在上面公式中，通常我们是假设 𝑝 为真实概率分布， 𝑞 为预测概率分布。以一个二分类为例， 𝑝 = [1, 0]，𝑞 = [ p, 1-p]，那么计算出交叉熵损失为 `L = - log(p)`

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/解决正负样本不均衡问题——Focal-Loss-20210903172756.png">
</p>

## 2. Focal Loss 损失
Focal Loss 是在交叉熵损失函数上进行改进的，其背景是来源于解决 one-stage detector 里 anchor 正负样本不均衡问题。作者认为 one-stage detector 检测还不够准的原因完全在于：

- 正负样本非常不均衡，而且绝大多数负样本都是 easy example；
- 虽然这些 easy example 的损失可能比较低，但是它们数量众多，依旧对 loss 有很大贡献，从而使得梯度被 easy example 主导。



参考文献:

- [[1] 深度学习基础篇——交叉熵损失函数](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CE_Loss.html)
- [2] [在人脸识别领域业界通常以误报率漏报率作为衡量算法能力的主要指标.](http://www.bio1000.com/news/201811/203913.html)
