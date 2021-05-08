### 1. 下载仓库

```
$ git clone https://github.com/YunYang1994/YunYang1994.github.io.git
$ cd YunYang1994.github.io
$ git checkout source
```

### 2. 开始构建

```
$ bash init.sh
```

### 3. 开始编译

```
$ bash build.sh
```

### 4. 上传至 github 

```
$ cd build
$ hexo g
$ hexo d
```


### 5. One More Thing

- 首先下载安装 [Picgo](https://github.com/Molunerfinn/PicGo) 软件，具体使用见[这里](https://cloud.tencent.com/developer/article/1651601)
- 在 Picgo 的 「指定存储路径」这一栏填写博客的标题，并在 「Picgo设置」里打开时间戳命名
- 直接将图片拖拽上传至 [blogimgs](https://github.com/YunYang1994/blogimgs)，然后复制图片链接至博客文档中
- 本博客同时支持在 Github 和 Gitee 上部署，只需要在 `./hexo/hexo_config.yml` 修改下 [deploy 源](https://github.com/YunYang1994/YunYang1994.github.io/blob/source/hexo/hexo_config.yml#L106)即可
- 注意如果在 Gitee 上部署，需要[「服务」](https://gitee.com/yunyang1994/YunYang1994/pages)里手动更新
