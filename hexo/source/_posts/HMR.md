---
title: 讲一讲目前深度学习下基于单目的三维人体重建技术
date: 2021-09-14 11:20:01
categories: 姿态估计
---

近年来，基于深度学习的单目三维人体重建技术已经取得了巨大的进展。特别是基于马普所的 SMPL 参数化人体模型这块，今天就来简单聊聊这一系列的相关工作。

<p align="center">
    <img width="65%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/讲一讲目前深度学习下基于单目的三维人体重建技术-20210914160852.png">
</p>

<!-- more -->

如上图所示，给定一张 RGB 图片，我们希望恢复出图中人体的姿态和形状等信息。从前面讲的 [SMPL 参数化人体模型](https://yunyang1994.gitee.io/2021/08/21/三维人体模型-SMPL-A-Skinned-Multi-Person-Linear-Model/)可知，我们只需要估计出数字人体模型的相关参数即可。例如在 HMR 算法中，则需要预测出 23x3 + 10 + 2 + 1 + 3 = 85 个参数：

- 23 个关节点的 pose 参数，每个参数由轴角表示（axis-angle representation）
- 10 个人体形状 shape 参数
- 相机外参数 t (图像平面上的 2D 平移）和缩放尺度 s 以及根结点的旋转轴角 R

对这些参数直接回归是比较困难的，所以网络预测的是相对于初始 SMPL 参数的偏移量。一般是通过两层全连接网络作为回归器 Regressor 输出偏移量，然后再与预测的 SMPL 参数拼接相加然后继续迭代 3 次得到。

<p align="center">
    <img width="22%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/讲一讲目前深度学习下基于单目的三维人体重建技术-20210914172101.png">
</p>

```python
pred_pose = init_pose
pred_shape = init_shape
pred_cam = init_cam
for i in range(n_iter):
    xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
    xc = self.fc1(xc)
    xc = self.drop1(xc)
    xc = self.fc2(xc)
    xc = self.drop2(xc)
    pred_pose = self.decpose(xc) + pred_pose
    pred_shape = self.decshape(xc) + pred_shape
    pred_cam = self.deccam(xc) + pred_cam
```

考虑到目前业界里 3D 动捕数据的稀缺性，而目前市面上能获得很多人工标注的 2D 关键点数据集（比如 [COCO](https://cocodataset.org/#keypoints-2020) 和 [PoseTrack](https://posetrack.net) 等）。因此会引入 3D -> 2D 的[<strong>重投影损失（reprojection loss）</strong>](https://github.com/mkocabas/VIBE/blob/master/lib/core/loss.py#L149)，即将 SMPL 人体的 3D 关键点投影到 2D 图像上与人工标注的 2D 关节点计算损失。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/讲一讲目前深度学习下基于单目的三维人体重建技术-20210914173824.png">
</p>

```python
# 定义 2D 关键点的损失函数
self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)

# 将 SMPL 人体 3D 关键点重投影到 2D 关键点
pred_keypoints_2d = projection(pred_joints, pred_cam)

def keypoint2d_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
    """
    Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight              # 如果没有标注，gt_weight = 0，否则为 1
    loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss
```

尽管 3D 数据集比较难获得，但是 3D 关键点损失也要考虑在内。它的损失函数与 2D 关键点损失一样，都是 nn.MSELoss 函数，输入则为 24 （23+1）个关键点的 3D 标注坐标和预测坐标。详见 [keypoint_3d_loss 损失函数](https://github.com/mkocabas/VIBE/blob/master/lib/core/loss.py#L161)。

<p align="center">
    <img width="30%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/讲一讲目前深度学习下基于单目的三维人体重建技术-20210914175801.png">
</p>

一些数据集如 [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 不仅能获得人体的 3D 关键点坐标和 pose 信息，还用到了 3D scan 扫描获得人体的 mesh 信息。我们可以通过使用 Mosh 工具将这些标注数据转化成 SMPL 的（β，θ）参数，然后对它们直接进行监督，详见 [smpl_losses 损失函数](https://github.com/mkocabas/VIBE/blob/master/lib/core/loss.py#L185)。

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/讲一讲目前深度学习下基于单目的三维人体重建技术-20210914180527.png">
</p>











