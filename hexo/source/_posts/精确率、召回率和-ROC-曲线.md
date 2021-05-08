---
title: 精确率、召回率和 ROC 曲线
date: 2017-03-11 13:40:57
tags:
	- 精确率、召回率和ROC曲线
categories:
	- 深度学习
mathjax: true
---

机器学习领域里评估指标这么多，今天就简单回顾下常用的准确率(accuracy)，精确率(precision) 和召回率(recall)等概念吧！哎，有时候时间太久了就会记不太清楚了。

<p align="center">
    <img width="60%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210547.png">
</p>

<!-- more -->

首先来看看混淆矩阵：

<p align="center">
    <img width="45%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210636.jpg">
</p>


<font color=blue>精确率(precision)</font>是针对<font color=red>预测结果</font>而言的，它表示的是<font color=red>预测为正的样本中有多少是对的</font>。那么预测为正就有两种可能：一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)。

<p align="center">
    <img width="18%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210652.jpg">
</p>

<font color=blue>召回率(recall)</font>是针对<font color=red>原来样本</font>而言的，它表示的是<font color=red>样本中的正例有多少被预测正确了</font>。那也有两种可能：一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。

<p align="center">
    <img width="18%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210658.jpg">
</p>


一般来说，精确率和召回率是一对矛盾的度量。精确率高的时候，往往召回率就低；而召回率高的时候，精确率就会下降。当我们不断地调整评判阈值时，就能获得一条 <font color=red><strong>P-R 曲线</strong></font>。

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210718.jpg">
</p>

但是在人脸识别领域里，关注更多的还是 <strong><font color=red>ROC 曲线</font></strong>。例如，某某公司经常说自己的人脸识别算法可以做到在千万分之一误报率下其准确率超过 99%，这说的其实就是 ROC 曲线。要知道这个概念，就必须了解 <font color=blue>TPR（True Positive Rate）</font>和 <font color=blue>FPR（False Positive Rate）</font>的概念。

<font color=red>TPR: 原来是对的，预测为对的比例。</font>（当然越大越好，1 为理想状态）


<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210709.jpg">
</p>


<font color=red>FPR: 原来是错的，预测为对的比例，误报率。</font>（当然越小越好，0 为理想状态）

<p align="center">
    <img width="20%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210713.jpg">
</p>

同样不断地调整评判阈值，就能获得一条 ROC 曲线。其中，ROC曲线下与坐标轴围成的面积称为 <font color=red><strong>AUC</strong></font>（AUC 的值越接近于 1，说明该模型就越好）。

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210722.jpg">
</p>

最后，必须要注意的是<font color=blue>准确率(accuracy)</font>和<font color=blue>精确率(precision)</font>是不一样的: 

<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/精确率、召回率和-ROC-曲线-20210508210704.jpg">
</p>


参考文献:

- [1] 张志华，《机器学习》. 清华大学出版社，2016.
- [2] [在人脸识别领域业界通常以误报率漏报率作为衡量算法能力的主要指标.](http://www.bio1000.com/news/201811/203913.html)
