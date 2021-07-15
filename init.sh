#! /bin/bash
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#   
#   Editor      : VIM 
#   File name   : build.sh
#   Author      : YunYang1994
#   Created date: 2020-10-12 13:13:17
#   Description : 
#
#================================================================

rm -rf build
mkdir build
cd build && hexo init
git clone https://github.com/tufu9441/maupassant-hexo.git themes/maupassant

npm install --save hexo-deployer-git
npm install --save hexo-renderer-pug
npm install --save hexo-renderer-sass
npm install --save hexo-generator-searchdb 
npm install --save hexo-generator-search
npm install --save hexo-wordcount
npm install --save hexo-helper-qrcode
npm uninstall hexo-generator-index --save
npm install hexo-generator-index-pin-top --save

rm -rf source
cp -rf ../hexo/* . 
mv hexo_config.yml _config.yml
mv theme_config.yml themes/maupassant/_config.yml

hexo g
hexo s
