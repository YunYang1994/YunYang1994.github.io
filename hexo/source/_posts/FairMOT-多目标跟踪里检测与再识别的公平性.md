---
title: FairMOT：讨论多目标跟踪里检测与再识别的公平性
date: 2021-09-09 00:00:00
categories: 目标跟踪
---

基于 anchor-free 的目标跟踪 family 又迎来了一位新成员，FairMOT. 它是在 CenterNet 基础上进行创新的，并真正意义上实现了端到端地将 Detection 和 ReId 任务进行联合训练。

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FairMOT-多目标跟踪里检测与再识别的公平性-20210909145011.gif">
</p>
<!-- more -->

以往大多数的目标跟踪都是采用 Detection + ReId 的方式，没有实现 jointly 端到端地联合训练，使得算法的跟踪精度也有限。FairMOT 分析了这种结果不佳的原因，总结下来主要有三点：

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FairMOT-多目标跟踪里检测与再识别的公平性-20210909153247.png">
</p>

- 作者认为 anchor-base 不适合 ReId 任务，应该使用 anchor-free 的方法。原因是可能会出现<strong>一个 anchor 响应多个目标或者多个 anchor 响应一个目标</strong>的情况（如上图所示），导致歧义性。如果我们只用一个中心点去看待，那么就不会出现这种情况。

- 现有的目标跟踪算法过度地依赖 Detection 精度，导致 ReId 任务受到不公平的忽视。ReId 任务需要高低层不同分辨率的特征融合，这在目前大多数的 Detection + ReId 框架里不太好做到。

- 在 MOT 中 ReID 特征的维数不宜过高，因为 MOT 的数据集一般来说都比较小。维度过高容易造成过拟合，而且显存和计算量都会增大。

FairMOT 网络有两个分支：Detection 分支和 ReID 分支。Detection 分支与 CenterNet 里的基本一样，这里不做介绍，让我们重点来看看 ReID 分支。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/FairMOT-多目标跟踪里检测与再识别的公平性-20210909154840.png">
</p>

ReID 分支的作用是在于输出每个目标的 embedding 向量。在 backbone 顶部设有一个 128 核的卷积层来提取<strong>每个位置</strong>的 embedding，如上图所示输出的是一个维度为 [128, H, W] 的 Re-ID Embeddings，然后喂入分类器：[nn.Linear(self.emb_dim, self.nID)](https://github.com/ifzhang/FairMOT/blob/ca63d27f19e8d2170b84edb80cc2dc348c3dcd5a/src/lib/trains/mot.py#L32) 计算 loss（需要注意的是，作者通过 reg_mask 对正样本进行了<strong>挑选</strong>，也就是说<strong>只有正样本才会计算 regression loss 和 ReID loss</strong>)。

```python
# https://github.com/ifzhang/FairMOT/blob/ca63d27f19e8d2170b84edb80cc2dc348c3dcd5a/src/detection_demo.py#L33

#--------------- 每个分支和它所对应的输出维度 ---------------#
reid_dim = 128
heads = {'hm': num_classes, 'wh': 2 if not ltrb else 4, 'id': reid_dim, 'reg': 2}

#------------ 根据 heads 字典和它的维度创建每个分支-----------#
# https://github.com/ifzhang/FairMOT/blob/ca63d27f19e8d2170b84edb80cc2dc348c3dcd5a/src/lib/models/networks/resnet_dcn.py#L155

for head in self.heads:          # 创建每个分支
    classes = self.heads[head]   # classes 为每个分支的输出维度，如 hm(高斯热图): num_classes
    if head_conv > 0:            #                                  wh(宽和高): 2
        fc = nn.Sequential(
            nn.Conv2d(64, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, classes,           # 分支的最后一层卷积核为 1x1，输出对应维度
            kernel_size=1, stride=1,
            padding=0, bias=True))
        if 'hm' in head:
            fc[-1].bias.data.fill_(-2.19)
        else:
            fill_fc_weights(fc)

#---------------------- 计算 re-ID 的 loss ------------------#
# https://github.com/ifzhang/FairMOT/blob/ca63d27f19e8d2170b84edb80cc2dc348c3dcd5a/src/lib/trains/mot.py#L56

self.classifier = nn.Linear(self.emb_dim, self.nID)
self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

if opt.id_weight > 0:
    id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
    id_head = id_head[batch['reg_mask'] > 0].contiguous()   # 选出正样本
    id_head = self.emb_scale * F.normalize(id_head)
    id_target = batch['ids'][batch['reg_mask'] > 0]

    id_output = self.classifier(id_head).contiguous()
    id_loss += self.IDLoss(id_output, id_target)
```

<table><center><td bgcolor=Plum><font color=black>
作者很巧妙地将整个过程中每个目标 ID 设置成一个类别，这样 ReID  Loss 就变成了一个多分类交叉熵损失函数。在训练阶段，ReID task 成了多分类任务；在测试阶段，砍掉 Linear 层直接取出 embedding 向量计算余弦距离，这其实和人脸识别过程一模一样。
</font></strong></td></center></table>

至于何时创建和销毁 tracker 以及它和 detection 之间怎么关联，这和 DeepSort 里的流程基本一致，具体请移步于[DeepSort：多目标跟踪算法 Sort 的进化版](https://yunyang1994.gitee.io/2021/08/27/DeepSort-多目标跟踪算法-SORT-的进阶版/)。


## 参考文献
- [[1] FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking,](http://arxiv.org/abs/2004.01888)
- [[2] https://github.com/ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)

