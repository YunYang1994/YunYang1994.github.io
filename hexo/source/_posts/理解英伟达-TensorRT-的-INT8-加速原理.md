---
title: 理解英伟达 TensorRT 的 INT8 加速原理
date: 2021-06-9 14:19:54
tags:
    - TensorRT 部署
    - INT8 加速原理
categories: 深度学习
---

目前的神经网络推理大部分是利用 32bit float 类型的数据进行计算的，bit 位数的多少直接限制了数据类型能够表达的数据范围，比如 float 32 的数据是由 1bit 表示符号，8bit 表示整数部，23 位表示分数部组成。但是这种运算比较耗时和消耗计算资源，因此诞生了 int8 量化算法。

<p align="center">
    <img width="70%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152045.png">
</p>

int8 量化是将数据保存为 int8 格式，这样一样计算时间和占用内存大大减小。目前量化有两种方式：一种是通过训练量化finetune原来的模型，另一种是直接对模型和计算进行量化。后者的代表便是英伟达的方案了，目前 PPT 已经公开，但是代码并没有开源。

<!-- more -->

## 1. 量化卷积核权重
量化的目的是为了把原来的 float32 位的卷积操作，转换为 int8 的卷积操作，这样**计算就变为原来的 `1/4`，但是访存并没有变少哈，因为我们是在 kernel 里面才把 float32 变为 int8 进行计算的**。

比如说 `float32` 位的变量 `a=6.238561919405008`，可以通过 `scale=23.242536 ` 映射到 `int8` 空间上到整数 `a*scale=145`。

<p align="center">
    <img width="50%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152310.png" style="max-width:70%;"></p>
 
 如上所示，这个 scale 是根据最大的权重绝对值 `thresh` 决定的，然后计算 `127` 与它的比值，便得到了 `scale` 值。

```python
max_val = np.max(group_weight)
min_val = np.min(group_weight)
thresh  = max(abs(max_val), abs(min_val))
scale = 127 / thresh                       # int8 范围: -127 ~ 127
```

**由于卷积运算是卷积核(`weights`)和数据流(`blob`)之间乘加操作，因此光对卷积核量化是不够的，还需要对数据流进行量化！**

## 2 量化输入数据流
我们发现，量化的过程与数据的分布有关。如下图所示：当数据的直方图分布比较均匀时，高精度向低精度进行映射就会将刻度利用比较充分；如果分布不均匀，就会浪费很大空间。通俗地来讲，就会出现很多数字挤在一个刻度里：比如 `scale=23.456` 时，`6.1817` 和 `6.1823` 都表示成了 `145`。

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152346.png" style="max-width:100%;"></p>

关于上面这种直接将量化阈值设置为 `|max|` 的方法，它的显著特点是低精度的 `int8` 空间没有充分利用，因此称为<strong><font color=red>不饱和量化(no saturation quantization)</font></strong>。针对这种情况，我们可以选择一个合适的**量化阈值(threshold)**，舍弃那些超出范围的数进行量化，这种量化方式充分利用了低精度空间，因此称为<strong><font color=red>饱和量化(saturation quantization)</font></strong>。

<p align="center">
    <img width="75%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152419.png" style="max-width:80%;"></p>

> 图中黄色部分被舍弃掉了，它们将直接被量化为 -127

通过对比两种量化方式我们可以发现，它们各有优缺点：不饱和量化方式的量化范围大，但是可能会浪费一些低精度空间从而导致量化精度低；饱和量化方式虽然充分利用了低精度空间，但是会舍弃一些量化范围。**因此这两种方式其实是一个量化精度和量化范围之间的平衡**。那么如何选择合适的量化方式呢，英伟达说了：卷积核权重量化应该使用不饱和量化，数据流量化应该使用饱和量化方式。那么问题来了，对于数据流的饱和量化，**怎么在数据流中找到这个最佳阈值(threshold) ？**

我们首先应该将经过网络每一层的数据流(`Blob`)给提取出来，得到它们的直方图分布。为了提取每一层的输入流，我们可以使用 [`torch.nn.Module.register_forward_pre_hook`](https://discuss.pytorch.org/t/understanding-register-forward-pre-hook-and-register-backward-hook/61457) 函数来操作。它就像一个钩子，可以把我们想要的东西给钩出来，并且不会对数据进行修改。不妨先在 `QuantizeLayer` 设置一个 hook 函数

```python
def hook(self, modules, input):
    self.blob = input[0].cpu().detach().numpy().flatten()
```
然后模型每次执行 `forward` 函数时，都会去执行 `hook` 函数里的内容：把该层的输入流复制给 `blob` 变量。

**跟权重量化的原理类似，数据流量化也需要找到最大绝对值 blob_max**。理论上这个值应该是全局的，但是模型的测试图片数量和分布都具有不可穷举性，因此我们往往会选择一些图片进行校**准 (calibration)**，使得`blob_max`尽量接近理论值。因此我们在上面这个钩子函数里设置了一个动态规划，数据流每次经过该层时都会对该值进行更新：

```python
def hook(self, modules, input):
    """
    VGG16 模型每次 forward 时，都会调用 QuantizeLayer.hook 函数，更新数据流的直方图分布
    """
    self.blob = input[0].cpu().detach().numpy().flatten()
    max_val = np.max(self.blob)
    min_val = np.min(self.blob)
    self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))
```

在获得这个 `blob_max` 值之后，我们会在 (0, blob_max) 划分 `2048` 个刻度（你也可以划分成4096个刻度，只要你喜欢），然后统计每个刻度范围内的数字出现的个数。例如，VGG16 最后一层输入流的直方图分布为：

<p align="center">
    <img width="55%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152447.png" style="max-width:60%;"></p>

> 可以看到第 `0` 个刻度的概率特别大，这表明大部分的数字都集中在(`-blob_max/2048, blob_max/2048`）区间内。

假如我们设定阈值 `Threshold=512`，因此需要将 `512` 个刻度合并成 `128` 个刻度。假设合并前 `(0,512)` 的分布为 P， 合并成 `(0, 128)` 后的分布为 Q。那么我们肯定要计算这两个分布的差异性，并希望它们之间的差异越小越好。

<p align="center">
    <img width="55%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152516.png" style="max-width:80%;"></p>
    
怎么计算两个分布的差异性呢？使用[**`KL 散度`**](https://baike.baidu.com/item/相对熵/4233536?fromtitle=KL散度&fromid=23238109&fr=aladdin)就可以！它又称为交叉熵，等于概率分布的信息熵(**Shannon entropy**)的差值。我们可以使用[scipy.stats.entropy(p, q)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)进行计算，但是它要求两个输入的长度必须相等：`len(p) == len(q)`。而`P`和`Q`的长度分别为512和128，因此我们需要将 `Q` 拓展回去，其长度也变成512。例如：

<p align="center">
    <img width="100%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/理解英伟达-TensorRT-的-INT8-加速原理-20210609152539.png" style="max-width:100%;"></p>
