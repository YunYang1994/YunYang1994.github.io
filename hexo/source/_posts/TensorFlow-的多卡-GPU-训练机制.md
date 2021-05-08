---
title: TensorFlow 的多卡 GPU 训练机制
date: 2020-02-07 12:29:23
tags:
	- 多卡GPU训练
categories:
	- 深度学习
---

武汉疫情还没过去，这几天窝在家里琢磨了下 TensorFlow 的多卡 GPU 分布式训练的机制。本文将使用流行的 MNIST 数据集上训练一个 MobileNetV2 模型，并利用 `tf.distribute.Strategy` 函数实现多卡 GPU 对训练方式。 详细代码见 [<font color=Red>TensorFlow2.0-Example</font>](https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/7-Utils/multi_gpu_train.py)

<p align="center">
    <img width="40%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/TensorFlow-的多卡-GPU-训练机制-20210509000745.jpg">
</p>

<!-- more -->

## 下载 MNIST 数据集

点击[<font color=Red>这里</font>](https://github.com/YunYang1994/yymnist/releases/download/v1.0/mnist.zip)可以下载到 mnist.zip，将它们解压得到以下目录结构：

```
├── test
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   ├── 6
│   ├── 7
│   ├── 8
│   └── 9
└── train
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    ├── 4
    ├── 5
    ├── 6
    ├── 7
    ├── 8
    └── 9

22 directories, 0 files
```

## 创建一个分发变量和图的策略

接下来将会使用到 `tf.distribute.MirroredStrategy` ，它是如何运作的？

- 所有变量和模型图都复制在副本上；
- 输入都均匀分布在副本中；
- 每个副本在收到输入后计算输入的损失和梯度；
- 通过求和，每一个副本上的梯度都能同步；
- 同步后，每个副本上的复制的变量都可以同样更新。

你可以这样创建一个策略：

```python
strategy = tf.distribute.MirroredStrategy()
```
或者指定使用特定的 GPU

```python
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:2", "/gpu:3"])
```

## 构建 MobileNetV2

使用 `tf.keras.applications.mobilenet_v2.MobileNetV2` 创建一个模型。你也可以使用模型子类化 API 来完成这个。

```python
# Defining Model
with strategy.scope():
    model = applications.mobilenet_v2.MobileNetV2(include_top=False, weights=None,
                                                  input_shape=(IMG_SIZE,IMG_SIZE,3))
    x = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    y = model(x)
    y = tf.keras.layers.AveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(512,  activation=None)(y)
    y = tf.keras.layers.Dense(10,   activation='softmax')(y)
    model = tf.keras.models.Model(inputs=x, outputs=y)
    optimizer = tf.keras.optimizers.Adam(0.001)
```

## 定义损失函数
在多卡 GPU 的训练方式中，`tf.distribute.Strategy` 是如何计算损失的呢？

- 举一个例子，假设您有 4 个 GPU，批量大小为 64. 输入的一个批次分布在各个副本（ 4个 GPU）上，每个副本获得的输入大小为 16。
- 每个副本上的模型使用其各自的输入执行正向传递并计算损失, 使用 `tf.nn.compute_average_loss` 来获取每张 GPU 卡的训练损失，并通过 `global_batch_size` 返回缩放损失。（相当于`scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)`）

```python
# Defining Loss and Metrics
with strategy.scope():
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy'
    )
```

## 训练循环

- 我们使用 `for x in ...` 迭代构造 train_dataset ；
- 缩放损失是 `distributed_train_step` 的返回值。 这个值会在各个副本使用`tf.distribute.Strategy.reduce` 的时候合并，然后通过 `tf.distribute.Strategy.reduce` 叠加各个返回值来跨批次。

```python
# Defining Training Loops
with strategy.scope():
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
    for epoch in range(EPOCHS):
        batchs_per_epoch = len(train_generator)
        train_dataset    = iter(train_generator)

        with tqdm(total=batchs_per_epoch,
                  desc="Epoch %2d/%2d" %(epoch+1, EPOCHS)) as pbar:
            for _ in range(batchs_per_epoch):
                batch_loss = distributed_train_step(next(train_dataset))
                batch_acc  = train_accuracy.result()
                pbar.set_postfix({'loss' : '%.4f' %batch_loss,
                                  'accuracy' : '%.6f' %batch_acc})
                train_accuracy.reset_states()
                pbar.update(1)
```
