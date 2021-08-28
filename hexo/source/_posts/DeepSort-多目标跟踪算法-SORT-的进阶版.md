---
title: DeepSort：多目标跟踪算法 Sort 的进化版
date: 2021-08-27 00:00:00
tags:
    - DeepSort
categories: 目标跟踪
---

在[之前的 Sort 算法](https://yunyang1994.gitee.io/2021/08/14/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking/)中讲到：<strong>尽管 Sort 具有速度快和计算量较小的优点，但它在关联匹配时没有用到物体的表观特征，导致物体被遮挡时容易出现 id-switch 的情况。</strong>针对这个算法的痛点，原 author 团队又发明了 Sort 的进化版 —— [DeepSort: Simple Online and Realtime Tracking with a Deep Associate Metric](https://arxiv.org/abs/1703.07402)
<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210827115042.png">
</p>
<!-- more -->

## 1. 卡尔曼滤波追踪器
<strong>DeepSort 的 KalmanFilter 假定跟踪场景是定义在 8 维状态空间（u, v, γ, h, ẋ, ẏ, γ̇, ḣ）中， 边框中心（u, v），宽高比 γ，高度 h 和和它们各自在图像坐标系中的速度。</strong>这里依旧使用的是匀速运动模型，并把（u，v，γ，h）作为对象状态的直接观测量（direct observations of the object state.）。在目标跟踪中，需要估计目标的以下两个状态：

- <strong>均值(Mean)</strong>：包含目标的中心位置和速度信息，由 8 维向量（u, v, γ, h, ẋ, ẏ, γ̇, ḣ）表示，其中每个速度值初始化为 0。均值 Mean 可以通过观测矩阵 H 投影到测量空间输出（u，v，γ，h）。
- <strong>协方差(Covariance)</strong>：表示估计状态的不确定性，由 8x8 的对角矩阵表示，矩阵中数字越大则表明不确定性越大。

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
- step1：首先利用先验估计协方差矩阵 P 和观测矩阵 H 以及测量状态协方差矩阵 R 计算出卡尔曼增益矩阵 K

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

### 1.3 跟踪器的状态
DeepSort 的跟踪器一共有 3 种状态：当 tracker 初始化时，分配为待定状态（tentative）；如果连续 n_init 帧匹配上，则转化为确定状态（confirmed），否则为删除状态（deleted）；如果 tracker 在确定状态下连续 max_age 帧没匹配上，那么就会变成删除状态被回收。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210828221215.png">
</p>

## 2. 关联度量
解决卡尔曼滤波器的预测状态和测量状态之间的关联可以通过构建匈牙利匹配来实现，在这个过程中需要结合两个合适的指标来整合物体的运动信息和外观特征。

### 2.1 马氏距离
为了整合物体的运动信息，使用了预测状态和测量状态之间的（平方）[马氏距离](https://zh.wikipedia.org/wiki/马哈拉诺比斯距离)：

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210828135516.png">
</p>

在上式中，d 和 y 分别代表测量分布和预测分布，S 为两个分布之间的协方差矩阵。由于测量分布的维度（4 维）和预测分布的维度（8 维）不一致，因此需要先将预测分布通过观测矩阵 H 投影到测量空间中（这一步其实就是从 8 个估计状态变量中取出前 4 个测量状态变量，详见 [project 函数](https://github.com/nwojke/deep_sort/blob/280b8bdb255f223813ff4a8679f3e1321b08cdfc/deep_sort/kalman_filter.py#L125)）。

```python
# Project state distribution to measurement space.
def project(self, mean, covariance):
    std = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-1,
        self._std_weight_position * mean[3]]

    # 初始化测量状态的协方差矩阵 R
    innovation_cov = np.diag(np.square(std)) # 使用的是对角矩阵，不同维度之间没有关联

    # 将均值向量映射到检测空间 得到 Hx
    mean = np.dot(self._update_mat, mean)

    # 将协方差矩阵映射到检测空间，得到 HP'H^T
    covariance = np.linalg.multi_dot((
        self._update_mat, covariance, self._update_mat.T))

    return mean, covariance + innovation_cov # 加上测量噪声
```

协方差矩阵 S 是一个实对称正定矩阵，可以使用 Cholesky 分解来求解马氏距离，这部分内容就不展开讨论。在 DeepSort 代码中，计算马氏距离的整个过程如下：

```python
# Compute gating distance between state distribution and measurements.
def gating_distance(self, mean, covariance, measurements,
                    only_position=False):

    # 首先需要先将预测状态分布的均值和协方差投影到测量空间
    mean, covariance = self.project(mean, covariance)

    # 假如仅考虑中心位置
    if only_position:
        mean, covariance = mean[:2], covariance[:2, :2]
        measurements = measurements[:, :2]

    # 对协方差矩阵进行 cholesky 分解
    cholesky_factor = np.linalg.cholesky(covariance)

    # 计算两个分布之间对差值
    d = measurements - mean

    # 通过三角求解计算出马氏距离
    z = scipy.linalg.solve_triangular(
        cholesky_factor, d.T, lower=True, check_finite=False,
        overwrite_b=True)

    # 返回平方马氏距离
    squared_maha = np.sum(z * z, axis=0)
    return squared_maha
```

马氏距离通过计算检测框距离预测框有多远的偏差来估计跟踪器状态的不确定性，此外还可以通过在 95% 的置信区间上从逆 χ2 分布中计算出的马氏距离来排除不可能的关联。在 DeepSort 的 4 维测量空间中，相应的马氏距离阈值为 9.4877，如果两个匹配框之间的马氏距离大于这个值，那么就认为两个框是不可能关联了。

### 2.2 外观特征
当物体运动状态的不确定性比较低时，使用马氏距离确实是一个不错的选择。由于卡尔曼滤波器使用的是匀速运动模型，它只能对物体的运动位置提供一个相对粗略的线性估计。<strong>当物体突然加速或减速时，跟踪器的预测框和检测框之间的距离就会变得比较远，这时仅使用马氏距离就会变得非常不准确</strong>。

因此 DeepSort 还对每个目标<strong>设计了一个深度外观特征描述符，它其实是一个在行人重识别数据集上离线训练的 ReID 网络提取到的 128 维单位特征向量（模长为 1 ）</strong>。对于每个追踪器 tracker，保留它最后 100 个与检测框关联成功的外观特征描述符集合 R 并计算出它们和检测框的最小余弦距离：

<p align="center">
    <img width="45%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210828151324.png">
</p>

同样，可以设置一个合适的阈值来排除那些外观特征相差特别大的匹配。


### 2.3 互相补充

上述两个指标可以互相补充从而解决关联匹配的不同问题：<strong>一方面，马氏距离基于运动可以提供有关可能的物体位置的信息，这对于短期预测特别有用；另一方面，当运动的判别力较弱时，余弦距离会考虑外观信息，这对于长时间遮挡后恢复身份特别有用</strong>。

为了建立关联问题，我们使用加权总和将两个指标结合起来：

<p align="center">
    <img width="45%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/DeepSort-多目标跟踪算法-SORT-的进阶版-20210828155133.png">
</p>

如果它同时在两个指标的门控范围之内，我们称其为可接受的关联。可以通过超参数 λ 来控制每个指标对合并成本的影响。在论文的实验过程中，发现当摄像机运动较大时，将 λ=0 是合理的选择（此时仅用到了外观信息）。

## 3.1 匹配问题
### 3.1 级联匹配
为了解决全局分配问题中检测框与跟踪器的关联，我们引入了一个级联来解决一系列的子问题。不妨先考虑这种场景：<strong>当物体被长时间遮挡时，后续的卡尔曼滤波预测结果会增大与物体位置关联的不确定性。因此概率分布会在状态空间中弥散，观察似然性就会变得比较平坦。</strong>关联度量应该增加检测框到预测框之间的距离来考虑这种概率分布的弥散，故而引入了一个级联匹配来优先考虑年龄较小的跟踪器：

```python
def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    # 遍历不同年龄
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        # 挑选出对应年龄的跟踪器
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        # 将跟踪器和尚未匹配的检测框进行匹配
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l

    # 挑选出未匹配的跟踪器
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections
```

在级联匹配的花费矩阵里，元素值为马氏距离和余弦距离的加权和。该匹配的精髓在于：<strong>挑选出所有 confirmed tracks，优先让那些年龄较小的 tracks 和未匹配的检测框相匹配，然后才轮得上那些年龄较大的 tracks 。</strong>这就使得在相同的外观特征和马氏距离的情况下，年龄较小的跟踪器更容易匹配上。<strong>至于年龄 age 的定义，跟踪器每次 predict 时则 age + 1。</strong> 

### 3.2 IOU 匹配

这个阶段是发生在级联匹配之后，匹配的跟踪器对象为那些 unconfirmed tracks 以及上一轮级联匹配失败中 age 为 1 的 tracks. 这有助于解决因上一帧部分遮挡而引起的突然出现的外观变化，从而减少被遗漏的概率。

```python
# 从所有的跟踪器里挑选出 unconfirmed tracks
unconfirmed_tracks = [
    i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

# 从上一轮级联匹配失败的跟踪器中挑选出连续 1 帧没有匹配上（相当于age=1）
# 的跟踪器，并和 unconfirmed_tracks 相加
iou_track_candidates = unconfirmed_tracks + [
    k for k in unmatched_tracks_a if
    self.tracks[k].time_since_update == 1]

# 将它们与剩下没匹配上的 detections 进行 IOU 匹配
matches_b, unmatched_tracks_b, unmatched_detections = \
    linear_assignment.min_cost_matching(
        iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        detections, iou_track_candidates, unmatched_detections)
```

## 参考文献
- [[1] Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
- [[2] https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- [[3] Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

