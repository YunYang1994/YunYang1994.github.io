---
title: 滚蛋吧，Anchor 君！旷视新科技，YOLOX
date: 2021-09-08 17:26:24
tags:
	- anchor free
categories:
	- 目标检测
---

天下苦 anchor 久矣，这两年是 anchor-free 系列目标检测算法的爆发时间段。但是 YOLO 系列最新推出的 v4 和 v5 依然抱着 anchor 不放，在这种背景下旷视科技推出了基于 anchor-free 的 YOLOx 算法。今天就来盘点一下这里面一些有意思的东西。

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/滚蛋吧-Anchor-君-旷视新科技-YOLOX-20210907145112.png">
</p>

<!-- more -->

作者选用的是 YOLOv3-SPP 作为 baseline，在这个基础上使用了很多 trick 不断升级打怪将 AP 提升了<strong> 8.8个百分点</strong>。

<p align="center">
    <img width="70%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/滚蛋吧-Anchor-君-旷视新科技-YOLOX-20210907152659.png">
</p>

### decoupled head
<strong>在目标检测里，回归任务和分类任务是有冲突的，</strong>所以最近几年出现的一些 anchor free 算法如 CornerNet、CenterNet 和 FCOS 等，都是将 regression 和 classification 分开进行预测。如下图所示：之前 regression 和 classification 任务都是长在一个头（head）上的，现在用了两个分支叉开了，因此称为 decoupled head。
<p align="center">
    <img width="70%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/滚蛋吧-Anchor-君-旷视新科技-YOLOX-20210907154324.png">
</p>
<p align="center">
    <img width="65%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/滚蛋吧-Anchor-君-旷视新科技-YOLOX-20210907154908.png">
</p>
实验结果发现，解藕操作不仅提升了 YOLOX 的性能和收敛速度，还为检测下游任务的一体化带来可能：

- 和 yolact  相似，实现端侧的实例分割
- YOLOX + 34 层输出，实现端侧人体的 17 个关键点检测。

### Data Augmentation
数据增强在目标检测里一直有人在做，但是将 AP 提升了 2.4% 还是比较难得的。Mosaic 经过 yolov4 和 v5 的验证，表明对结果有显著的提升。作者在 YOLOX-L 模型上尝试了 [copy-paste](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf)，也带来了 0.8% 的提升。此外还将<strong>关闭 Aug 的时间节点设定为终止前的 10~15 个 epoch，目的是为了为了让检测器避开不准确标注框的影响，在自然图片的数据分布下完成最终的收敛。</strong>
<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/滚蛋吧-Anchor-君-旷视新科技-YOLOX-20210907161944.png">
</p>
作者还提出一个观点：<strong>只有当模型容量足够大时，相对于先验知识（各种 tricks，hand-crafted rules ），更多的后验（data augmentation）知识才会产生本质影响。</strong>

### anchor free
将 YOLO 算法从 anchor-base 切换到 anchor-free 的做法非常简单，直接套用 FCOS 那套逻辑即可。但是想要提高预测精度达到 SOTA 却不简单，作者在知乎上对正负样本的匹配经验做了一些总结，大家可以移步[知乎](https://www.zhihu.com/question/473350307/answer/2021031747)。

### Multi positives
为了和 YOLOv3 正负样本匹配机制一致，作者刚开始是只选择了一个 positive sample（目标中心）。<strong>但是作者在实验过程中发现，中心附近的 positive samples 有助于缓解正负样本不均衡问题，不应该完全被忽略掉。</strong>因此在训练过程中，作者把中心 3x3 区域内的像素都当作正样本，这个其实也是 FCOS 里的 center sampling 操作。最终这个骚操作使精度提升了 2.1 个百分点，效果显著啊。

### SimOTA
OAT（Optimal Transport Assignment）是旷视发的一篇 CVPR 2021 论文，它把样本匹配建模成最优传输问题，找出前面 k 个最优的正样本。SimOTA 是在 OTA 的基础做了简化， 有兴趣可以去看看[原文](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf)，这里就不做介绍了。

最后一些碎碎念：要说把这个算法取名 YOLOX，我是不服的。因为它其实和 YOLO 相关性并不大，反而和 FCOS 非常相似，所以感觉这样取名也是蹭了 YOLO 的热度吧 😂。


## 参考文献
- [[1] YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
- [[2] 知乎：如何评价旷视开源的YOLOX，效果超过YOLOv5?](https://www.zhihu.com/question/473350307/answer/2021031747)












