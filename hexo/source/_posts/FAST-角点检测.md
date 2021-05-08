---
title: FAST 角点检测
date: 2018-07-17 16:25:10
tags:
    - 角点检测
categories: 图像处理
---

FAST 是一种角点，主要检测局部像素灰度变化明显的地方，以速度快著称。<font color=Red>它的思想是: 如果一个像素与它邻域的像素差别较大(过亮或过暗), 那它更可能是角点。相比于其他角点检测算法，FAST 只需比较像素亮度的大小， 十分快捷。</font>它的检测过程如下:

<p align="center">
    <img width="60%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/FAST-角点检测/01.png">
</p>

<!-- more -->

1. 在图像中选取像素 p，假设它的亮度为 C，并设置一个阈值 T。
2. 以像素 p 为中心, 选取半径为 3 的圆上的<font color=Red> 16 个像素点</font>（如上图所示)。

```python
def circle(row,col):
    '''
    对于图片上一像素点位置 (row,col)，获取其邻域圆上 16 个像素点坐标，圆由 16 个像素点组成
    
    args:
        row：行坐标 注意 row 要大于等于3
        col：列坐标 注意 col 要大于等于3       
    '''
    point1  = (row-3, col)
    point2  = (row-3, col+1)
    point3  = (row-2, col+2)
    point4  = (row-1, col+3)
    point5  = (row, col+3)
    point6  = (row+1, col+3)
    point7  = (row+2, col+2)
    point8  = (row+3, col+1)
    point9  = (row+3, col)
    point10 = (row+3, col-1)
    point11 = (row+2, col-2)
    point12 = (row+1, col-3)
    point13 = (row, col-3)
    point14 = (row-1, col-3)
    point15 = (row-2, col-2)
    point16 = (row-3, col-1)
    
    return [point1, point2,point3,point4,point5,point6,point7,point8,point9,point10,point11,point12, point13,point14,point15,point16]
```

3. <font color=Red>假如这 16 个点中，有连续的 N 个点的亮度大于 C + T 或小于 C − T，那么像素 p 可以被认为是角点。</font>
4. 为了排除大量的非角点提出了一种高速测试方法：<font color=Red>直接检测邻域圆上的第  1，5，9，13 个像素的亮度。只有当这四个像素中有三个同时大于 C + T 或小于  C − T 时，当前像素才有可能是一个角点，否则应该直接排除。</font>这样的预测试操作大大加速了角点检测。

```python
def is_corner(image,row,col,threshold):
    '''
    检测图像位置(row,col)处像素点是不是角点
    如果圆上有12个连续的点满足阈值条件，那么它就是一个角点
    
    方法：
        如果位置1和9它的像素值比阈值暗或比阈值亮，则检测位置5和位置15
        如果这些像素符合标准，请检查像素5和13是否相符
        如果满足有3个位置满足阈值条件，则它是一个角点
        重复循环函数返回的每个点如果没有满足阈值，则不是一个角落
        
        注意：这里我们简化了论文章中的角点检测过程，会造成一些误差
    
    args:
        image：输入图片数据,要求为灰度图片
        row：行坐标 注意row要大于等于3
        col：列坐标 注意col要大于等于3 
        threshold：阈值        
    return : 
        返回True或者False
    '''
    # 校验
    rows,cols = image.shape[:2]
    if row < 3 or col < 3 : return False    
    if row >= rows-3 or col >= cols-3: return False
       
    intensity = int(image[row][col])
    ROI = circle(row,col)
    # 获取位置1,9,5,13的像素值
    row1, col1   = ROI[0]
    row9, col9   = ROI[8]
    row5, col5   = ROI[4]
    row13, col13 = ROI[12]
    intensity1  = int(image[row1][col1])
    intensity9  = int(image[row9][col9])
    intensity5  = int(image[row5][col5])
    intensity13 = int(image[row13][col13])
    # 统计上面4个位置中满足  像素值  >  intensity + threshold点的个数
    countMore = 0
    # 统计上面4个位置中满足 像素值  < intensity - threshold点的个数
    countLess = 0
    if intensity1 - intensity > threshold:
        countMore += 1 
    elif intensity1 + threshold < intensity:
        countLess += 1
    if intensity9 - intensity > threshold:
        countMore += 1
    elif intensity9 + threshold < intensity:
        countLess += 1
    if intensity5 - intensity > threshold:
        countMore += 1
    elif intensity5 + threshold < intensity:
        countLess += 1
    if intensity13 - intensity > threshold:
        countMore += 1
    elif intensity13 + threshold < intensity:
        countLess += 1
        
    return countMore >= 3 or countLess>=3
```

5. 在第一遍检测后，原始的 FAST 角点经常出现“扎堆”的现象。因此需要使用非极大值抑制，在一定区域内仅保留响应极大值的角点，避免角点集中的问题。

<p align="center">
    <img width="60%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/FAST-角点检测/02.png">
</p>

> 第一张图片使用了非最大值抑制，而第二张没有使用。可以明显看到，第二张图的关键点的位置重复比较严重。

为了方便，我们可以在 OpenCV 里直接创建 FAST 特征点检测器并使用它:

```python
import cv2 

img = cv2.imread('simple.jpg',0)
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# enable nonmaxSuppression
fast.setNonmaxSuppression(1)

# find and draw the keypoints
kp = fast.detect(img, None)

img = cv.drawKeypoints(img, kp, None, color=(255,0,0))
```

参考文献:

- [[1] 第十四节、FAST角点检测(附源码)](https://www.cnblogs.com/zyly/p/9542164.html#_label3)
- [[2] OpenCV 中文文档 4.0.0 - 角点检测的FAST算法](https://www.bookstack.cn/read/opencv-doc-zh-4.0/docs-4.0.0-5.6-tutorial_py_fast.md)