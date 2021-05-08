---
title: 手写双目立体匹配 SGM 算法（下)
date: 2020-01-20 23:52:04
tags:
    - 视差估计
mathjax: true
categories: 立体视觉
---

上节的内容主要对 SGM 算法的匹配代价体 (Cost Volume) 进行了详细介绍，发现如果只寻找逐个像素匹配代价的最优解会使得视差图对噪声特别敏感。因此在能量方程中考虑使用该点像素的邻域视差数据来构造惩罚函数以增加平滑性约束, 这个求解过程也称为<strong>代价聚合 (Cost Aggregation) </strong>的过程。 

## 能量方程

SGM 算法建立了能量方程，并引入了视差约束条件，以对此进行优化：

<p align="center">
    <img width="80%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/MommyTalk1600751174763.jpg">
</p>

<!-- more -->

等式右边第一项表示像素点 `p` 在视差范围内所以匹配代价之和; 第二项和第三项是指当前像素 `p` 和其邻域内所有像素 `q` 之间的平滑性约束，增加了惩罚因子 `P1` 和 `P2`: 若 `p` 和 `q` 的视差的差值等于 1，则惩罚因子 `P1`，若差值大于 1，则惩罚因子为 `P2`。 

为了高效地求解它，SGM 提出一种路径代价聚合的思路，即将像素所有视差下的匹配代价进行像素周围所有路径上的一维聚合得到路径下的路径代价值，然后将所有路径代价值相加得到该像素聚合后的匹配代价值。


## 路径代价

设 `L_{r}` 表示穿过 `r` 方向的扫描路径代价，其计算方式如下所示：

<p align="center">
    <img width="80%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/MommyTalk1600751250322.jpg">
</p>

等号右边第一项表示像素点 `p` 的初始匹配代价；第二项表示 `p` 的前一个像素点 `p − r` 的最小匹配代价：若和 `p` 的视差差值为 0，无需加任何惩罚因子，差值为 1，加惩罚因子 `P1` ，若差值大于 1，则惩罚因子为 `P2`；第三项表示前一个像素点 `p − r` 沿 `r` 路径上的最小匹配代价，加入该项的目的是抑制 `L_{r}( p, d )` 的数值过大，并不会影响视差空间。

```python
def get_path_cost(slice, offset):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    P1 = 5
    P2 = 70

    slice_dim, disparity_dim = slice.shape
    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)
    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=np.uint32)

    penalties[np.abs(disparities - disparities.T) == 1] = P1
    penalties[np.abs(disparities - disparities.T)  > 1] = P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path
```

## 代价聚合

### 单路聚合
如果我们只考虑单路扫描会是什么样的结果呢？假设代价聚合路线为从上至下，即 south 方向（2 号路线）。如下图所示，那么

<p align="center">
    <img width="25%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/south.png">
</p>


```python
main_aggregation   = np.zeros(shape=(image_h, image_w, max_disparity), dtype=np.uint32)
aggregation_volume = np.zeros(shape=(image_h, image_w, max_disparity, 1), dtype=np.uint32)

for x in range(0, image_w):
    south = cost_volume[0:image_h, x, :]
    main_aggregation[:, x, :] = get_path_cost(south, 1)

aggregation_volume[:, :, :, 0] = main_aggregation
```

便得到了<strong>代价聚合体（aggregation_volume）</strong>进行视差计算，找到该路径下的聚合最小匹配代价值：

```python
def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map
```
最后通过最小化该路径在一维聚合下的匹配代价，得到了视差图如下图所示。

```python
disparity_map = select_disparity(aggregation_volume)
disparity_map = normalize(disparity_map)
cv2.imwrite("scan_south_disp.png", disparity_map)
```

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/scan_south_disp.png">
</p>

### 多路聚合

从单路扫描的结果可以看出，一维扫描线优化仅仅只受一个方向约束，容易造成“条纹效应”，所以在 SGM 算法中，聚合了多个扫描路径上的匹配代价。

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/scanline.png">
</p>

一共有八个方向：south 和 north，east 和 west，south_east 和 north_west， south_west 和 north_east。刚刚实现了 south 方向单路扫描聚合代价的最优求解，而其他路扫描聚合代价的过程和它是类似的。限于篇幅，在这里就不补充了，最后得到的视差图如下所示：

<p align="center">
    <img width="40%" src="https://gitee.com/yunyang1994/BlogSource/raw/master/hexo/source/images/SGM/disparity_map.png">
</p>


