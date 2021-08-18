---
title: 三维人体动捕模型 SMPL：A Skinned Multi Person Linear Model
date: 2021-08-21 11:20:01
---

在<strong>人体动作捕捉</strong>（motion capture）领域，SMPL 算法最为常见，它是由德国马普所提出的一种参数化的三维人体动捕模型，具有通用性、易于渲染和兼容现有商业软件（比如 UE4 和 Unity）的优点。

<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210816165705.png">
</p>

「这是一篇鸽了很久的文章，今天补上。」
<!-- more -->

## 1. SMPL 简介
### 1.1 线性混合蒙皮
SMPL 涉及到游戏制作和渲染技术里面的一些东西，特别是计算机图形学领域。考虑到 SMPL 是在 LBS（Linear Blending Skinning，线性混合蒙皮）的基础上开发的，因此先对 LBS 做个简单的介绍。

对于一个虚拟形象，其实可以大致分为两大块：<strong>骨架（bones）</strong>和<strong>表皮（skin）</strong>。骨架一般由一套<strong>关节树（joints tree）</strong>构成，表皮则由一系列的<strong>网格顶点（vertices）</strong>组成，每个 vertices 都有坐标位置 xyz，然后这些 vertices 就组成了面（也就是表皮）。 在虚拟形象的美术制作过程中，通常是先制作出一套骨架，然后将这些网格顶点（皮）在 rest-pose 状态下按照一定的权重绑定在每个关节上，这个过程我们称之为<strong>蒙皮（skinning）</strong>。

<p align="center">
    <img width="80%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210818173252.png">
</p>

当虚拟形象发生运动时，骨骼的每个关节也会发生相应的旋转和位移，这个时候所绑定的网格顶点就需要根据每个绑定的关节点的影响加权求和算出运动后的位置。由于整个过程都是可以通过矩阵的线性运算得到，并且考虑到了所有关节点的混合影响，因此称为<strong>线性混合蒙皮</strong>。

LBS 主要是用来计算蒙皮后的网格顶点位置，假设虚拟人物一共有 `{1，2，3，...，m}` 个关节点，`n` 个网格顶点，其数学表达式如下：

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210818175355.png">
</p>

`p'` 为蒙皮后的网格顶点新位置，维度为 `[n, 3]`；`w` 为<strong>权重矩阵</strong>，维度为 `[n, m]`；`T` 则是每个关节点的<strong>仿射变换矩阵</strong>，维度为 `[m, 4, 4]`，该矩阵代表了关节点的旋转和平移；`p` 为蒙皮前的网格顶点位置。

### 1.2 为何提出 SMPL



## 2. SMPL 模型


## 3. SMPL 训练








