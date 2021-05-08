---
title: TensorFlow 模型转化 tflite
date: 2019-05-16 00:44:31
tags:
	- 移动端部署
categories:
	- 深度学习
---

自从有了TensorFlow Lite，应用开发者可以在移动设备上很轻松地部署神经网络。

<p align="center">
    <img width="90%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/Tensorflow-模型转化-tflite-20210508220505.jpg">
</p>

<!-- more -->

Tensorflow Lite 转化器可以将我们的训练模型转化成 `.tflite` 文件，它分别支持 [<font color=Red>SavedModel directories</font>](https://tensorflow.google.cn/guide/saved_model), [<font color=Red>concrete functions</font>](https://tensorflow.org/guide/concrete_function) 和 [<font color=Red>tf.keras models</font>](https://tensorflow.google.cn/guide/keras/overview)三种结构。由于我经常使用的是 `tf.keras.model` 结构，因此只对它进行详细介绍。

以 mtcnn 网络的 rnet 模型为例：

```python
import tensorflow as tf

class RNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(28, 3, 1, name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu1")

        self.conv2 = tf.keras.layers.Conv2D(48, 3, 1, name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu2")

        self.conv3 = tf.keras.layers.Conv2D(64, 2, 1, name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu3")

        self.dense4 = tf.keras.layers.Dense(128, name='conv4')
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=None, name="prelu4")

        self.dense5_1 = tf.keras.layers.Dense(2, name="conv5-1")
        self.dense5_2 = tf.keras.layers.Dense(4, name="conv5-2")

        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=False):
        out = self.prelu1(self.conv1(x))
        out = tf.nn.max_pool2d(out, 3, 2, padding="SAME")
        out = self.prelu2(self.conv2(out))
        out = tf.nn.max_pool2d(out, 3, 2, padding="VALID")
        out = self.prelu3(self.conv3(out))
        out = self.flatten(out)
        out = self.prelu4(self.dense4(out))
        score = tf.nn.softmax(self.dense5_1(out), -1)
        boxes = self.dense5_2(out)
        return boxes, score
```

接下来就是对模型进行转化和量化了，转换器可以配置为应用各种优化措施（optimizations），这些优化措施可以提高性能，减少文件大小。

```python
rnet.predict(tf.ones(shape=[1, 24, 24, 3]))
rnet_converter = tf.lite.TFLiteConverter.from_keras_model(rnet)

# 量化（quantization）可以减小模型的大小和推理所需的时间
rnet_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

# 保存 tflite 模型
with open("rnet.tflite", "wb") as f:
    rnet_tflite_model = rnet_converter.convert()
    f.write(rnet_tflite_model)
```

参考文献:

- [1] [TensorFlow 中文开发指南](https://tensorflow.google.cn/lite/guide/get_started#4_optimize_your_model_optional)
- [2] [TensorFlow2.0-Example](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/MTCNN/mtcnn.py)
