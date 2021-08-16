---
title: 多目标追踪 SORT 算法：Simple Online and Realtime Tracking
date: 2021-08-14 16:52:10
tags:
    - sort
categories: 目标跟踪
---

在<strong>多目标跟踪</strong>（multiple object tracking）领域，[SORT（Simple Online and Realtime Tracking）](https://arxiv.org/abs/1602.00763)算是最经典的入门算法了。[这份代码](https://github.com/YunYang1994/openwork/tree/main/sort)对该算法进行了 python 和 C++ 实现，感兴趣的可以点开看看。

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking-20210812202019.png">
</p>

<!-- more -->

整个流程如下图所示：在第 1 帧时，人体检测器 detector 输出 3 个 bbox（黑色），模型会分别为这 3 个 bbox 创建卡尔曼滤波追踪器 kf1，kf2 和 kf3，对应人的编号为 1，2，3 。在第 2 帧的过程 a 中，这 3 个跟踪器会利用上一帧的状态分别输出棕红色 bbox、黄色 bbox 和 青绿色 bbox。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking-20210812201919.png">
</p>

<table><center><td bgcolor=Plum><font color=black>
由于 detector 输出的黑色 bbox 是没有 id 的，因此需要将它们和上一帧的跟踪器相对应起来。换句话说就是把 detector 输出的目标检测框和 KalmanFilter 输出的预测框相<font color=black><strong>关联</strong></font>，这样使得每个目标检测框的 id 就是它所关联的跟踪器 id。这里关联的核心是：<font color=green><strong>用 iou 计算 bbox 之间的距离 ➕ 匈牙利算法匹配。</strong></font>
</font></strong></td></center></table>

首先计算出 <strong><font color=red>detector 输出的目标检测框</font></strong>（黑色框）和 <strong><font color=red>KalmanFilter 输出的预测框</font></strong> （棕红色 bbox、黄色 bbox 和 青绿色 bbox）之间的 <strong><font color=red>iou 表格</font></strong>如下：

|iou|黑色 bbox1|黑色 bbox2|黑色 bbox3|
|:---:|:---:|:---:|:---:|
|棕红色 bbox|0.91|0.00|0.00|
|黄色 bbox|0.00|0.98|0.00|
|青绿色 bbox|0.00|0.00|0.99|

上表可以抽象成一个矩阵，如果是如上表所示的求和最小问题，那么这个矩阵就叫做<strong>花费矩阵（Cost Matrix）</strong>；如果要求的问题是使之和最大化，那么这个矩阵就叫做<strong>利益矩阵（Profit Matrix）</strong>。由于这里 iou 表格的总和越大，则代表 bbox 之间关联匹配得越好，因此是一个利益最大化的问题。可以考虑使用 [匈牙利算法](https://www.pythonf.cn/read/34325) 进行求解，它的最坏时间复杂度为 `O(n^3)`，python 求解如下：

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

profit_matrix = np.array([[0.91, 0.00, 0.00],
                          [0.00, 0.98, 0.00],
                          [0.00, 0.00, 0.99]])

rows, cols = linear_sum_assignment(profit_matrix, maximize=True)
matches = list(zip(rows, cols))        # [(0, 0), (1, 1), (2, 2)]
```

在完成目标检测框和跟踪器预测框的关联匹配后，我们还需要更新校正卡尔曼滤波跟踪器，并输出优化后的 bbox（如 frame2-b 所示）。需要补充以下两点：

- 这里会对目标检测框进行修正和优化，是因为 detector 的检测框都有一定的抖动，而卡尔曼滤波刚好可以用于防抖。

- 跟踪器在 frame2-a 过程中输出的是基于上一历史状态的预测框，是一种先验估计；而在 frame2-b 过程中输出的则是将测量框（目标检测 bbox）与预测框（先验估计）加权后的结果，是一种后验估计。


## KalmanBoxTracker

```python
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
```

### 状态变量 x

状态变量 x 的设定是一个 7维向量：`x=[u, v, s, r, u^, v^, s^]T`。u、v 分别表示目标框的中心点位置的 x、y 坐标，s 表示目标框的面积，r 表示目标框的宽高比。u^、v^、s^ 分别表示横向 u(x方向)、纵向 v(y方向)、面积 s 的运动变化速率。

- u、v、s、r 初始化：根据第一帧的观测结果进行初始化。
- u^、v^、s^ 初始化：当第一帧开始的时候初始化为0，到后面帧时会根据预测的结果来进行变化。

### 状态转移矩阵 F

定义的是一个 `7x7` 的单位方阵，运动形式和转换矩阵的确定都是基于匀速运动模型，状态转移矩阵F根据运动学公式确定，跟踪的目标假设为一个匀速运动的目标。通过 `7x7` 的状态转移矩阵F 乘以 `7*1` 的状态变量 x 即可得到一个更新后的 `7x1` 的状态更新向量x。

```python
        self.kf.F = np.array([[1,0,0,0,1,0,0],      # 7x7 维度
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
```

### 观测矩阵 H

定义的是一个 `4x7` 的矩阵，乘以 `7x1` 的状态更新向量 x 即可得到一个 `4x1` 的 `[u,v,s,r]` 的估计值。

```python
        self.kf.H = np.array([[1,0,0,0,0,0,0],      # 4x7 维度
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
```

### 协方差矩阵 RPQ

```python
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
```

- 测量噪声的协方差矩阵 R：`diag([1,1,10,10]T)`
- 先验估计的协方差矩阵 P：`diag([10,10,10,10,1e4,1e4,1e4]T)`。1e4：1x10 的 4 次方。
- 过程激励噪声的协方差矩阵 Q：`diag([1,1,1,1,0.01,0.01,1e-4]T)`。

### predict 预测阶段
在预测阶段，追踪器不仅需要预测 bbox，还要记录它自己的活跃度。如果这个追踪器连续多次预测而没有进行一次更新操作，那么表明该跟踪器可能已经“失活”了。因为它的预测框和检测框没有匹配上，说明它之前记录的目标很有可能已经消失了。但是也不一定会发生这种情况，还一种结果是目标在连续几帧消失后又出现在画面里。

```python
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if(self.time_since_update>0):   # 一旦出现不匹配的情况，连续匹配次数被归 0
            self.hit_streak = 0
        self.time_since_update += 1     # 连续不匹配的次数 + 1
        
        self.history.append(convert_x_to_bbox(self.kf.x))  # [u,v,s,r] --> [x1,y1,x2,y2]
        return self.history[-1]
```

考虑到这种情况，使用 <strong>time_since_update 记录了追踪器连续没有匹配上的次数</strong>，该变量在每次 predict 时都会加 1，每次 update 时都会归 0. 假如我们设定跟踪器出现超过连续 2 帧没有匹配关联上，即当 tracker.time_since_update > 2 时，该跟踪器则会被判定失活而被移除列表。

### update 更新阶段
<table><center><td bgcolor=LightPink><font color=black>
大家都知道，卡尔曼滤波器的更新阶段是使用了观测值 z 来校正误差矩阵和更新卡尔曼增益，并计算出先验估计值和测量值之间的加权结果，该加权结果即为后验估计值。
</font></strong></td></center></table>

```python
    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.history = []
        
        self.time_since_update = 0   # 连续不匹配的次数归 0
        self.hits += 1               # 总的匹配次数 + 1
        self.hit_streak += 1         # 连续匹配次数 + 1
        
        # 卡尔曼滤波器更新校正
        self.kf.update(convert_bbox_to_z(bbox))  # bbox 是观测值 [x1,y1,x2,y2] --> [u,v,s,r]
```

每次更新时，总的匹配次数 hit 会加 1，连续匹配次数 hit_streak 也加 1. <strong>而如果一旦出现不匹配的情况时，hit_streak 变量会在 predict 阶段被归 0 而重新计时。</strong>

## bbox 关联匹配

bbox 的关联匹配过程在前面已经讲得很详细了，它是将 tracker 输出的预测框（注意是先验估计值）和 detector 输出的检测框相关联匹配起来。输入是 dets： [[x1,y1,x2,y2,score],...] 和 trks： [[x1,y1,x2,y2,tracking_id],...] 以及一个设定的 iou 阈值，该门槛是为了过滤掉那些低重合度的目标。

```python
def associate_detections_to_trackers(dets, trks, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
        dets:
            [[x1,y1,x2,y2,score],...]
        trks:
            [[x1,y1,x2,y2,tracking_id],...]
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
```

该过程返回：matches（已经匹配成功的追踪器）, unmatched_detections（没有匹配成功的检测目标） and unmatched_trackers（没有匹配成功的跟踪器）。

<table><center><td bgcolor=DarkTurquoise><font color=black>
对于已经匹配成功的追踪器，则需要用观测值（目标检测框）去更新校正 tracker 并输出修正后的 bbox；对于没有匹配成功的检测目标，则需要新增 tracker 与之对应；对于没有匹配成功的跟踪器，如果长时间处于失活状态，则可以考虑删除了。
</font></strong></td></center></table>

## 整体流程
有一位[大佬](https://www.codenong.com/cs106088758/)画出了整个 sort 的大概流程，我觉得还可以，这里分享下：

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking-20210813161621.png">
</p>

讲了这么多，那么 sort 有哪些优缺点呢？

- 作者在卡尔曼滤波器追踪中使用了匀速运动模型，这一点可能在某些场景下是不合理的。
- 关联匹配过程中没有使用 feature，这就造成两个物体在重合度较高的时候会发生 id-switch。
- 优点也很明显，那就是非常快，而且计算量小。

参考文献:

- [[1] Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- [[2] SORT跟踪算法的详细解释，不容错过](https://www.geek-share.com/detail/2765731307.html)
- [[2] 车流量检测实现：多目标追踪、卡尔曼滤波器、匈牙利算法、SORT/DeepSORT、yoloV3、虚拟线圈法、交并比IOU计算](https://www.codenong.com/cs106088758/)
