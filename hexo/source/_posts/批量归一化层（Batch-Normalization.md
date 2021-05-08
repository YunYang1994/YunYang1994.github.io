---
title: 批量归一化层（Batch Normalization)
date: 2019-07-09 22:28:07
tags:
	- Batch Normalization
categories:
	- 深度学习
mathjax: true
---

通常来说，数据标准化预处理对于浅层模型就足够有效了。<font color=red>但随着模型训练的进行，当每层中参数更新时，靠近输出层的输出容易出现剧烈变化。这令我们难以训练出有效的深度模型，而批量归一化（batch normalization）的提出正是为了应对这种挑战。</font>

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/批量归一化层（Batch Normalization)/Internal_Covariate_Shift.jpg"></p>

## BN 来源

在机器学习领域中，满足一个很重要的假设，即独立同分布的假设：就是假设训练数据和测试数据是满足相同分布的，这样通过训练数据获得的模型就能够在测试集获得一个较好的效果。而在实际的神经网络模型训练中，隐层的每一层数据分布老是变来变去的，这就是所谓的 <font color=red><strong>“Internal Covariate Shift”</strong></font>。

<!-- more -->

在这种背景下，然后就提出了 BatchNorm 的基本思想：<strong>能不能让每个隐层节点的<font color=red>激活输入分布</font>固定下来呢？</strong>

BN不是凭空拍脑袋拍出来的好点子，它是有启发来源的：之前的研究表明如果在图像处理中对输入图像进行白化（Whiten）操作的话 —— <font color=red><strong>所谓白化，就是对输入数据分布变换到 0 均值，单位方差的正态分布</strong></font> —— 因此 BN 作者推断，如果对神经网络的每一层输出做白化操作的话，模型应该也会较快收敛。

## 计算过程
首先对小批量的样本数据求均值和方差：

<p align="center">
    <img width="35%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/批量归一化层（Batch Normalization)/MommyTalk1600748016395.jpg"></p>


接下来，使用按元素开方和按元素除法对样本数据进行标准化:


<p align="center">
    <img width="20%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/批量归一化层（Batch Normalization)/MommyTalk1600748133659.jpg"></p>

<p align="center">
    <img width="20%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/批量归一化层（Batch Normalization)/MommyTalk1600748180578.jpg"></p>

这里 ε > 0是一个很小的常数，保证分母大于 0。在上面标准化的基础上，批量归一化层引入了<strong>两个需要学习的参数：拉伸(scale)参数 γ 和偏移(shift)参数 β。</strong><font color=red>这两个参数会把标准正态分布左移或者右移一点并长胖一点或者变瘦一点，从而使得网络每一层的数据分布保持相似。</font>

```python
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过autograd来判断当前模式是训练模式还是预测模式
    if not autograd.is_training():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使用二维卷积层的情况，计算通道维上(axis=1)的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta # 拉伸和偏移
    return Y, moving_mean, moving_var
```

## BN 位置

在 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) 一文中，作者指出，“we would like to ensure that for any parameter values, the network always produces activations with the desired distribution”（produces activations with the desired distribution，<font color=red><strong>为激活层提供期望的分布</strong></font>）。

<table><tr><td bgcolor= LightSalmon><strong>因此 `Batch Normalization` 层恰恰插入在 conv 层或全连接层之后，而在 relu 等激活层之前。</strong></td></tr></table>

## BN 优点

- <strong><font color=red>解决了 Internal Covariate Shift 的问题</font></strong>：模型训练会更加稳定，学习率也可以设大一点，同时也减少了对权重参数初始化的依赖；
- <strong><font color=red>对防止 gradient vanish 有帮助</font></strong>：一旦有了 Batch Normalization，激活函数的 input 都在零附近，都是斜率比较大的地方，能有效减少梯度消失；
- <strong><font color=red>能有效减少过拟合</font></strong>：据我所知，自从有了 Batch Normaliztion 后，就没有人用 Dropout 了。直观的理解是：对网络的每一层 layer 做了 BN 处理来强制它们的数据分布相似，这相当于对每一层的输入做了约束（regularization）。

## 参考文献

- [1] 李沐，动手深度学习. 2019.9.12
- [2] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
