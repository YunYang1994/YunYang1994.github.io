---
title: DeepSort：多目标跟踪算法 Sort 的进化版
date: 2021-08-27 00:00:00
---

在[之前的 Sort 算法](https://yunyang1994.gitee.io/2021/08/14/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking/)中讲到：<strong>尽管 Sort 具有速度快和计算量较小的优点，但它在关联匹配时没有用到物体的表观特征，导致物体被遮挡时容易出现 id-switch 的情况。</strong>针对这个算法的痛点，原 author 团队又发明了 Sort 的进化版 —— [DeepSort: Simple Online and Realtime Tracking with a Deep Associate Metric](https://arxiv.org/abs/1703.07402)
<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827115042.png">
</p>
<!-- more -->

## 1. 卡尔曼滤波追踪器
<strong>DeepSort 的 KalmanFilter 假定跟踪场景是定义在 8 维状态空间（u, v, γ, h, ẋ, ẏ, γ̇, ḣ）中， 边框中心（u, v），宽高比 γ，高度 h 和和它们各自在图像坐标系中的速度。</strong>这里依旧使用的是匀速运动模型，并把（u，v，γ，h）作为对象状态的直接观测量（direct observations of the object state.）。

关于以下公式的变量和符号说明，请参考[卡尔曼滤波算法，永远滴神！](https://yunyang1994.gitee.io/2021/07/10/卡尔曼滤波算法-永远滴神/)

### 1.1 predict 阶段
- step1：首先利用上一时刻 k-1 的后验估计值通过状态转移矩阵 F 变换得到当前时刻 k 的先验估计状态

<p align="center">
    <img width="14%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827150310.png">
</p>
<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827153028.png">
</p>

- step2：然后使用上一时刻 k-1 的后验估计协方差来计算当前时刻 k 的先验估计协方差

<p align="center">
    <img width="23%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827153635.png">
</p>

整个过程如下：

```python
def predict(self, mean, covariance):
    # mean, covariance 相当于上一时刻的后验估计均值和协方差
    
    std_pos = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-2,
        self._std_weight_position * mean[3]]
    std_vel = [
        self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-5,
        self._std_weight_velocity * mean[3]]

    # 初始化噪声矩阵 Q
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

    # x' = Fx
    mean = np.dot(self._motion_mat, mean)

    # P' = FPF^T + Q
    covariance = np.linalg.multi_dot((
        self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
    
    # 返回当前时刻的先验估计均值 x 和协方差 P
    return mean, covariance
```

predict 函数的输入为卡尔曼滤波器在上一时刻的后验估计均值 `x_{k-1}` 和协方差 `P_{k-1}`，输出为当前时刻的先验估计均值`x_{k}` 和协方差 `P_{k}`。

### 1.2 update 阶段
- step1：首先利用先验估计协方差矩阵 P 和观测矩阵 H 以及测量状态协方差矩阵 R （这里 R 为零矩阵，认为测量是绝对正确的）计算出卡尔曼增益矩阵 K

<p align="center">
    <img width="30%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827170204.png">
</p>

- step2：然后将卡尔曼滤波器的先验估计值 x 通过观测矩阵 H 投影到测量空间，并计算出与测量值 z 的残差 y

<p align="center">
    <img width="16%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827162140.png">
</p>
<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827173356.png">
</p>

- step3：将卡尔曼滤波器的预测值和测量值按照卡尔曼增益的比例相融合，得到后验估计值 x

<p align="center">
    <img width="16%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827171038.png">
</p>

- step4：计算出卡尔曼滤波器的后验估计协方差

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827171350.png">
</p>

整个过程如下：

```python
def update(self, mean, covariance, measurement):

    # 将先验估计的均值和协方差映射到检测空间，得到 Hx' 和 HP'
    projected_mean, projected_cov = self.project(mean, covariance)

    chol_factor, lower = scipy.linalg.cho_factor(
        projected_cov, lower=True, check_finite=False)

    # 计算卡尔曼增益 K
    kalman_gain = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        check_finite=False).T

    # y = z - Hx'
    innovation = measurement - projected_mean

    # x = x' + Ky
    new_mean = mean + np.dot(innovation, kalman_gain.T)

    # P = (I - KH)P'
    new_covariance = covariance - np.linalg.multi_dot((
        kalman_gain, projected_cov, kalman_gain.T))

    # 返回当前时刻的后验估计均值 x 和协方差 P
    return new_mean, new_covariance
```

<table><center><td bgcolor=Plum><font color=black>
最后总结一下：predict 阶段和 update 阶段都是为了计算出卡尔曼滤波的<strong><font color=blue>估计均值 x </font></strong>和<strong><font color= blue>协方差 P</font></strong>，不同的是前者是基于上一历史状态做出的<strong><font color=green>先验估计</font></strong>，而后者则是融合了测量值信息并作出校正的<font color=green><strong>后验估计</strong></font>。
</font></strong></td></center></table>

## 2. 关联问题

