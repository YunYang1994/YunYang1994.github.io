---
title: 使用 Pytorch 对 mnist 数字进行分类
date: 2018-09-09 13:54:24
tags:
	- mnist 分类
categories:
	- 深度学习
---

一直以来就非常喜欢 Pytorch，今天就小试牛刀一下，用它实现对 mnist 数字进行分类。

<p align="center">
    <img width="80%" src="https://cdn.jsdelivr.net/gh/YunYang1994/blogimgs/使用-Pytorch-对-mnist-数字进行分类-20210508214147.jpg">
</p>

<!-- more -->

首先可以从[<font color=Red>这里</font>](https://github.com/YunYang1994/yymnist/releases/download/v1.0/mnist.zip)下载 mnist 数据集并解压，然后代码的开始部分如下：

```python
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

之后，我们会想办法构造一个手写数字数据集:

```python
class MnistDataset(Dataset):
    """custom mnist dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []
        for i in range(10):
            img_paths = glob.glob(root_dir + "%d/*.jpg" %i)
            for img_path in img_paths:
                img_label = {'img_path':img_path, 'label':i }
                self.data_list.append(img_label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.data_list[idx]['img_path']
        image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        label = self.data_list[idx]['label']
        return image, label
```

对数据集的装载使用的是 `torch.utils.data.DataLoader` 类，类中的 `dataset` 参数用于指定我们载入的数据集名称，`batch_size` 设置了每个 `batch` 的样本数量，`shuffle=True` 会在装载过程将数据的顺序打乱然后打包。

```python
train_dataset = MnistDataset("./mnist/train/")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    						batch_size=32,
    						shuffle=True,
    						num_workers=4)
```

在顺利完成数据的加载后，我们就可以搭建一个简单的 CNN 模型，如下所示：

```python
class ConvNet(nn.Module):
    """ Convolutional neural network (two convolutional layers) """
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

model = ConvNet(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
在编写完搭建卷积神经网络的模型代码后，我们就可以开始对模型进行训练和对参数进行优化了。

```python
# Define training loops
for epoch in range(20):
    model.train()
    loss_value = 0.
    acc_value  = 0.
    num_batch  = 0
    with tqdm(total=len(train_loader),
                desc="Epoch %2d/20" %(epoch+1)) as pbar:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            num_batch  += 1

            loss_value += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc_value += correct / len(target)

            pbar.set_postfix({'loss' :'%.4f' %(loss_value / num_batch),
                              'acc'  :'%.4f' %(acc_value  / num_batch)})
            pbar.update(1)
```

在测试阶段，代码也非常简洁:

```python
# Define testing step
with torch.no_grad():
    model.eval()
    model(data)
```

接着我们可以保存模型：

```python
torch.save(model.state_dict(), 'model.pth')
```

然后下次便可以重新加载模型:

```python
model = ConvNet(num_classes=10).to(device)
model.load_state_dict(torch.load("model.pth"))
```
