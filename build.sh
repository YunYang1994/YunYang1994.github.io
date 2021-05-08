#! /bin/bash

#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#   
#   Editor      : VIM 
#   File name   : build.sh
#   Author      : YunYang1994
#   Created date: 2021-03-30 14:13:23
#   Description : 
#
#================================================================


cd build

rm -rf source
cp -rf ../hexo/* . 
mv hexo_config.yml _config.yml
mv theme_config.yml themes/maupassant/_config.yml

hexo g
hexo s
