---
title: 什么是相机的位姿 ？
date: 2019-12-27 10:47:04
tags:
    - 相机位姿
    - 视觉 Slam
categories: 立体视觉
---

在视觉 slam 领域里，相机的位姿是一个特别重要的概念。简单来说，相机的位姿（pose）就是相机的位置和姿态的合称，它描述了世界坐标系与相机坐标系之间的转换关系。

<p align="center">
    <img width="50%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/RT.png">
</p>

<!-- more -->

如上图所示：点 `P` 的世界坐标为 `P_{w}`，可以通过相机的位姿矩阵 `T` 转换到相机坐标系下为 `P_{c}` ：


<p align="center">
    <img width="17%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/MommyTalk1608812694553.jpg">
</p>

当然你可以将点 `P` 从相机坐标系转换到世界坐标系中：


<p align="center">
    <img width="30%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/MommyTalk1608812750071.jpg">
</p>

其中 `T_{cw}` 为该点从世界坐标系变换到相机坐标系的变换矩阵， `T_{wc}` 为该点从相机坐标系变换到世界坐标系的变换矩阵。**它们二者都可以用来表示相机的位姿，前者称为相机的外参**。

> 实践当中使用 `T_{cw}` 来表示相机位姿更加常见。然而在可视化程序中使用 `T_{wc}` 来表示相机位姿更为直观，因为此时它的平移向量即为相机原点在世界坐标系中的坐标。视觉 Slam 十四讲中的第五讲的 joinMap 使用的就是 `T_{wc}` 来表示相机位姿进行点云拼接。

相机位姿矩阵 `T` 其实主要由旋转矩阵 `R` 和平移向量 `t` 组成：

<p align="center">
    <img width="24%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/01.png">
</p>

其中旋转矩阵 `R` 一共有 9 个量，但是一次旋转只有 3 个自由度，因此这种表达方式是冗余的。可以使用欧拉角来描述这种旋转行为，它使用了 3 个分离的转角，把一个旋转分解成了3次绕不同轴的旋转，如下所示：

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/angle.jpg">
</p>

因此旋转矩阵 `R` 可以由三个转角来表示，它们分别是：

- 偏航角 yaw，绕物体的 `Z` 轴旋转的角度, 用 `gamma` 表示；
- 俯仰角 pitch，<font color=Red>**旋转之后**</font>绕 `Y` 轴旋转的角度, 用 `alpha` 表示；
- 滚转角 roll，<font color=Red>**旋转之后**</font>绕 `X` 轴旋转的角度, 用 `beta` 表示；

既然欧拉角可以表示物体的旋转状态，那么旋转矩阵 `R` 应该也能被这个三个角度所表示:首先，旋转矩阵 R 可以被三个矩阵分解得到

<p align="center">
    <img width="30%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/MommyTalk1600755628141.jpg">
</p>

其中：

<p align="center">
    <img width="90%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/MommyTalk1600755686966.jpg">
</p>


因此它们三者相乘便得到旋转矩阵 `R` 的表达形式:

<p align="center">
    <img width="100%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/MommyTalk1600755742741.jpg">
</p>

使用这种方法表示的一个重大缺点就是会碰到著名的<font color=red>万向锁问题</font>：在俯仰角为正负 90 度时，第一次旋转与第三次旋转将会使用同一个轴，使得系统失去了一个自由度（由 3 次旋转变成了 2 次旋转）。理论上可以证明，只要想用 3 个实数来表达三维旋转，都不可避免遇到这种问题。

<p align="center">
    <img width="80%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/CameraPose/Lock.png">
</p>


参考文献：

- [《视觉SLAM十四讲》相机位姿与相机外参的区别与联系](https://blog.csdn.net/CharmingSun/article/details/97445425)