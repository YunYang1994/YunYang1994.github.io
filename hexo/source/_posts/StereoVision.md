---
title: 双目测距和三维重建
date: 2019-12-29 12:04:21
tags:
    - 三维重建
categories: 立体视觉
mathjax: true
---

双目相机通过同步采集左右相机的图像，计算图像间视差，来估计每一个像素的深度。一旦我们获取了物体在图像上的每个像素深度，我们便能重构出一些它的三维信息。

<p align="center">
    <img width="50%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/result.gif">
</p>

双目相机一般由左眼和右眼两个水平放置的相机组成，其距离称为双目相机的基线(Baseline, 记作 b)，是双目的重要参数。由于左右两个相机之间有一定距离，因此同一个物体在左右图上的横坐标会有一些差异，称为<font color=red><strong>视差(Disparity)</strong></font>。

<!-- more -->

<p align="center">
    <img width="30%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/disparity.png">
</p>


根据视差，我们可以估计一个像素离相机的距离。

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/triangle_measurement.png">
</p>

<p align="center">
    <img width="28%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/MommyTalk1600749965692.jpg">
</p>

根据相机坐标系点 `P` 坐标为 `(X, Y, Z)` 到图像坐标系 `(u, v)` 之间的投影关系：

<p align="center">
    <img width="20%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/MommyTalk1600750017380.jpg">
</p>

从而反推得到点 `P` 的坐标信息

<p align="center">
    <img width="17%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/MommyTalk1600750052615.jpg">
</p>

例如，以下图片为例：

|||
|:---:|:---:|
|![原图](https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/000004_10.jpg)|![视差图](https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/StereoVision/000004_10_disp.png)|

> 视差图采用十六位 (uint16) 整数来存取，并将视差值扩大了 256 倍，所以在读取时需要除以256。

鼠标右键将原图和视差图下载下来，然后安装好第三方库 PyOpenGL==3.1.0 和 [pangolin](https://github.com/uoip/pangolin) 即可执行以下程序便得到了动图的结果。

```python
import cv2
import pangolin
import numpy as np
import OpenGL.GL as gl

fx = 721.5377
fy = 721.5377
cx = 607.1928
cy = 185.2157
B  = 0.54

img_disp = cv2.imread('000004_10_disp.png', -1) / 256.
imgL = cv2.imread('000004_10.jpg')
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

h, w = imgL.shape[:2]
f = 0.5 * w
points, colors = [], []

for v in range(h):
    for u in range(w):
        disp = img_disp[v, u]
        if disp > 0.:
            depth = B * fx / disp
            z_w = depth
            x_w = (u - cx) * z_w / fx
            y_w = (v - cy) * z_w / fy
            points.append([x_w, y_w, z_w])
            colors.append(imgL[v, u])
points = np.array(points)
colors = np.array(colors) / 255.

pangolin.CreateWindowAndBind('Main', 640, 480)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define Projection and initial ModelView matrix
scam = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(640, 480, 2000, 2000, 320, 240, 0.1, 1000),
    pangolin.ModelViewLookAt(0, 0, -20, 0, 0, 0, 0, -1, 0))
handler = pangolin.Handler3D(scam)

# Create Interactive View in window
dcam = pangolin.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
dcam.SetHandler(handler)

while not pangolin.ShouldQuit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    dcam.Activate(scam)
    gl.glPointSize(3)
    pangolin.DrawPoints(points, colors)
    pangolin.FinishFrame()
```
