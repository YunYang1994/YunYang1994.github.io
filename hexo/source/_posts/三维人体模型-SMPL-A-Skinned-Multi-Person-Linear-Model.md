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

## 1. 人体动捕介绍
### 1.1 动作捕捉技术
目前人体动作捕捉技术在影视制作和游戏领域已经应用得很成熟了，最常见的就是基于可穿戴设备（比如 IMU）的人体动捕技术。
<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819170007.png">
</p>
美术师在制作一个虚拟形象模型时，会让其呈现 T-pose 摆放，并定义一套<strong>人体关节树（skeleton tree）</strong>。该关节树的特点在于每个<strong>关节点（joint）</strong>都有一个<strong>父节点（parent joint）</strong>，整个骨架的旋转和平移则通过<strong>根结点（root）</strong>实现。而在下图中，PELVIS 则是根节点。
<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819173522.png">
</p>

|Index|Joint name|Parent joint|
|:---:|:---:|:---:|
|0|PELVIS|-|
|1|SPINE_NAVAL|PELVIS|
|2|SPINE_CHEST|SPINE_NAVAL|
|...|...|...|

如下图所示：<strong>每个关节点都有一套自己的坐标系，当人体在运动时，每个关节点就会相对其父节点发生旋转，这个旋转过程可以用一个四元数表达。</strong>如果两套虚拟形象的关节点坐标系朝向不一致，那么还需要进行一些适配工作（相当枯燥乏味）。
<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819180003.png">
</p>

### 1.2 线性混合蒙皮
SMPL 涉及到游戏制作和渲染技术里面的一些东西，特别是计算机图形学领域。考虑到 SMPL 是在 LBS（Linear Blending Skinning，线性混合蒙皮）的基础上开发的，因此先对 LBS 做个简单的介绍。

对于一个虚拟形象，其实可以大致分为两大块：<strong>骨架（bones）</strong>和<strong>表皮（skin）</strong>。骨架一般由一套<strong>关节树（skeleton tree）</strong>构成，表皮则由一系列的<strong>网格顶点（vertices）</strong>组成，每个 vertices 都有坐标位置 xyz，然后这些 vertices 就组成了面（也就是表皮）。 在虚拟形象的美术制作过程中，通常是先制作出一套骨架，然后将这些网格顶点（皮）在 rest-pose 状态下按照一定的权重绑定在每个关节上，这个过程我们称之为<strong>蒙皮（skinning）</strong>。

<p align="center">
    <img width="80%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210818173252.png">
</p>

当虚拟形象发生运动时，骨骼的每个关节也会发生相应的旋转和位移，这个时候所绑定的网格顶点就需要根据每个绑定的关节点的影响加权求和算出运动后的位置。由于整个过程都是可以通过矩阵的线性运算得到，并且考虑到了所有关节点的混合影响，因此称为<strong>线性混合蒙皮</strong>。

LBS 主要是用来计算蒙皮后的网格顶点位置，假设虚拟人物一共有 `{1，2，3，...，m}` 个关节点，`n` 个网格顶点，其数学表达式如下：

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210818175355.png">
</p>

`p'` 为蒙皮后的网格顶点新位置，维度为 `[n, 3]`；`w` 为<strong>权重矩阵</strong>，维度为 `[n, m]`；`T` 则是每个关节点的<strong>仿射变换矩阵</strong>，维度为 `[m, 4, 4]`，该矩阵代表了关节点的旋转和平移；`p` 为蒙皮前的网格顶点位置。

## 2. SMPL 模型

### 2.1 SMPL 的背景
LBS 面临的一个难点是：<strong>线性混合蒙皮算法会出现皮肤塌陷和皱褶的问题</strong>，作者称之为<strong> “taffy”（太妃糖） </strong>和<strong> “bowtie”（领结）</strong>。比如下图中当手臂弯曲的时候，LBS 的效果（青绿色）就折叠得比较夸张，而且在关节连接处不能提供平滑自然的过渡。<strong>目前商业上普遍的做法是通过人工绑定（rigging）和手工雕刻 blend shape 来改善这个问题，这个过程会比较耗费人力。</strong>

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819151954.png">
</p>

SMPL 较好地解决了上述的痛点，并且一开始的出发点就是为了提出业界兼容、简单易用和渲染速度快的三维人体重建模型，作者认为像这种以往需要人工绑定（rigging）和手工雕刻 blend shape 的过程其实是可以通过大量数据学习得到。

### 2.2 SMPL 参数定义
SMPL 模型一共定义了 `N=6890` 个 vertices 和 `K=23` 个 joints，并且通过以下两类统计参数对人体进行描述。

- <strong>体型参数 β：</strong>拥有 10 个维度去描述一个人的身材形状，每一个维度的值都可以解释为人体形状的某个指标，比如高矮，胖瘦等。

- <strong>姿态参数 θ：</strong>拥有 24×3 个维度去描述人体的动作姿态，其中 24 指的是 23 个关节点 + 1 个根结点，3 则指的是轴角（axis-angle）里的数值。

在 python 代码中，我们可以这样随机设置 SMPL 参数：

```python
# 设置身材参数 betas 和姿态参数 poses
betas = np.random.rand(10) * 0.03
poses = np.random.rand(72) * 0.20
```

### 2.3 SMPL 的过程
SMPL 过程主要可以分为以下三大阶段：

#### 1. 基于形状的 blend shape （混合变形）
人体的网格顶点（vertices）会随着 shape 参数 β 变化而变化，这个变化过程是在一个<strong>基模版（或者称之为统计上的均值模版，mean template）</strong>上线性叠加的。关于这个<strong>线性叠加偏量</strong>，作者使用了 `Bs(β)` 函数来计算：

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819202619.png">
</p>

它表示每个 shape 参数对 vertices 的影响。其中 S （对应 `smpl['shapedirs']`）是通过数据学习出来的，它的维度为 (6890, 3, 10)。

```python
# 根据 betas 调整 T-pose, 计算 vertices
v_shaped = smpl['shapedirs'].dot(betas) + smpl['v_template']  # 还要与基模版相加
```
#### 2. 基于姿态的 blend shape
前面计算的是人体在静默姿态（T-pose）下的 blend shape，这里将计算人体在不同 pose 参数 θ 下的影响。同样的定义了一个函数 `Bp(θ)` 计算该线性叠加偏量：

<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model-20210819204307.png">
</p>

在上面公式中，<strong>因为我们是计算相对 T-pose 状态下的线性叠加偏量，所以人体的位姿应该也是要相对 T-pose 状态下进行变化，因此括号里减去了 T-pose 位姿的影响。</strong>每个 pose 参数都用旋转矩阵 R 表示，所以是 9K。同样 P （对应 `smpl['posedirs']`）也是通过数据学习出来的，它的维度为 (6890, 3, 207），其中 207 是因为 23x9 得到。




#### 3. 蒙皮过程（blend skinning）





## 3. SMPL 训练








