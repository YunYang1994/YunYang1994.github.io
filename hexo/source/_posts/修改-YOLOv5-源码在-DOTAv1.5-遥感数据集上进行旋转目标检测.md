---
title: 修改 YOLOv5 源码在 DOTAv1.5 遥感数据集上进行旋转目标检测
date: 2021-04-20 17:26:24
tags:
	- rotated object detection
categories:
	- 目标检测
---

[YOLOv5](https://github.com/ultralytics/yolov5) 发布已经有一段时间了，但是我一直还没有怎么去用过它。机会终于来了，最近需要做一个「旋转目标检测」的项目。于是我想到用它来进行魔改，使其能输出目标的 `rotated bounding boxes`。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/修改-YOLOv5-源码在-DOTAv1.5-遥感数据集上进行旋转目标检测-20210509005417.jpg">
</p>

<!-- more -->

## DOTA-v1.5 遥感数据集

[DOTA](https://captain-whu.github.io/DOTA/index.html) 是武汉大学制作的一个关于航拍遥感数据集，里面的每个目标都由一个任意的四边形边界框标注。这个数据集目前一共有 3 个版本，这里我们只使用和介绍其中的 DOTA-v1.5 版本。

### 图片类别
一共有 2806 张图片，40 万个实例，分为 16 个类别：***飞机，轮船，储罐，棒球场，网球场，篮球场，地面跑道，港口，桥梁，大型车辆，小型车辆，直升机，环形交叉路口，足球场，游泳池，起重机。***

### 标注方式

每个目标都被一个四边框 **oriented bounding box (OBB)** 标注，其 4 个顶点的坐标表示为（x1, y1, x2, y2, x3, y3, x4, y4）。标注框的起始顶点为黄色，其余 3 个顶点则按照顺时针顺序排列。


<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/修改-YOLOv5-源码在-DOTAv1.5-遥感数据集上进行旋转目标检测-20210509005422.jpg">
</p>

每张图片的标注内容为：在第一行 `imagesource` 表示图片的来源，`GoogleEarth` 或者 `GF-2`；第二行 `gsd` 表示的是[地面采样距离（Ground Sample Distance，简称 GSD）](https://zh.wikipedia.org/wiki/地面采样距离)，如果缺失，则为 `null`；第三行以后则标注的是每个实例的四边框、类别和识别难易程度。

```
'imagesource':imagesource
'gsd':gsd
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
...
```

### 数据下载

- 百度云盘：[Training set](https://pan.baidu.com/s/1kWyRGaz)，[Validation set](https://pan.baidu.com/s/1qZCoF72)，[Testing images](https://pan.baidu.com/s/1i6ly9Id)
- 谷歌云盘：[Training set](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK?usp=sharing)，[Validation set](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing)，[Testing images](https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing)

建议通过百度云盘下载，得到的数据容量为：`train`，9.51G；`val`，3.11G。

## 数据集预处理

### 加载显示图片

我们需要用到官方发布的 [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 来对数据集进行预处理，不妨根据 `README` 先下载安装该工具。然后我们可以使用 [`DOTA.py`](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/DOTA.py) 来加载指定的图片并显示目标的 obb 框。如果你还想将数据格式转换成 COCO 格式，那么就可以使用 [DOTA2COCO.py](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/DOTA2COCO.py)。

### 分割数据集

数据集里的图片分辨率都很高（最高达到了 `20000x20000`），显然我们的 GPU 不能满足这样的运算要求。如果我们 resize 图片，则会损失图片的信息，尤其是那种大量的小目标（低于 `20x20`）可能会直接消失。因此我们可以考虑使用 [`ImgSplit.py`](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/ImgSplit.py#L241) 将单张遥感图像裁剪切割成多张图片：

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/修改-YOLOv5-源码在-DOTAv1.5-遥感数据集上进行旋转目标检测-20210509005429.jpg">
</p>

由于切割图片时会有重叠区域（gap），一般 gap 设为切割图像尺寸的 20% 为宜。分割后会将裁剪的位置信息保留在裁剪后的图片名称中，例如图片 `P0770__1__586___334.png` 就是从原图 `P0770.png` 中 `x=586, y=334` 的位置处开始裁剪。

### 最小外接矩形

DOTA 图片里的标注框为任意四边形，而我们需要的是带旋转角度的标准矩形，这就要用到 OpenCV 的 [cv2.minAreaRect()](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle) 来求任意四边形的最小外接矩形，示例代码如下：

```python
import cv2
import numpy as np

image = cv2.imread("demo.png")
poly  = [[462.0, 785.0],        # 任意四边形的顶点坐标
         [630.0, 787.0],
         [623.0, 952.0],
         [467.0, 953.0]]
poly  = np.array(poly, dtype=np.float32)

rect  = cv2.minAreaRect(poly)
# 返回中心坐标 x, y 和长宽 w, h 以及与 x 轴的夹角 Q

box   = cv2.boxPoints(rect).astype(np.int)
# 返回四个顶点的坐标 (x1, y1), (x2, y2), (x3, y3) 和 (x4, y4)

image = cv2.drawContours(image, [box], 0, (255, 0, 0), 5)
```

<p align="center">
    <img width="30%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/修改-YOLOv5-源码在-DOTAv1.5-遥感数据集上进行旋转目标检测-20210509005434.png">
</p>

> 绿色框为任意四边形，红色框是它的最小外接矩形。

## 修改算法的思路
可能会有人要问，为什么针对遥感图片的目标就需要进行旋转检测呢，水平检测难道就不行吗？<font color=red>如果采用水平检测的方法，那么得到的检测框就容易高度重叠，而目前绝大多数的目标检测算法对于这种高度重叠的目标都容易发生漏检。</font>

<p align="center">
    <img width="35%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/修改-YOLOv5-源码在-DOTAv1.5-遥感数据集上进行旋转目标检测20210524144517.png">
</p>

我们想到：假如在原有的水平目标检测上，给预测框（boudning boxes) 加个角度 theta，那不就实现了旋转目标检测嘛。最简单的思想是，将这个角度预测当作是分类任务去处理，即一共有 180 个类别，便实现了 0~179 个角度。 在模型结构中，我们只需要修改 yolov5 的 Detect 层：

```python
class Detect(nn.Module):  # 定义检测网络
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=16, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        # number of classes
        self.nc = nc

        # number of outputs per anchor   （xywh + score + num_classes + num_angle）
        self.angle = 180
        self.no = nc + 5 + self.angle
        
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl  # init grid   [tensor([0.]), tensor([0.]), tensor([0.])] 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # shape(3, ?(3), 2)
        
        self.register_buffer('anchors', a)  # shape(nl,na,2) = (3, 3, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch) # 最后一层卷基层
        '''
        m(
            (0) :  nn.Conv2d(in_ch[0]（17）, (nc + 5 + self.angle) * na, kernel_size=1)
            (1) :  nn.Conv2d(in_ch[1]（20）, (nc + 5 + self.angle) * na, kernel_size=1)
            (2) :  nn.Conv2d(in_ch[2]（23）, (nc + 5 + self.angle) * na, kernel_size=1)
        )
        '''
```






