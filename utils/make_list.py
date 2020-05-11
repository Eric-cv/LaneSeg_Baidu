#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

"""
   ColorImage/
     road02/
       Record002/
         Camera 5/
           ...
         Camera 6
       Record003
       ....
     road03
     road04
   Label/
     Label_road02/
      Label
       Record002/
         Camera 5/
          ...
         Camera 6
       Record003
       ....
     Label_road03
     Label_road04     
     
"""

label_list = []
image_list = []
'''
for s1 in os.listdir(image_dir):

    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, 'Labels_' + str.lower(s1), 'Label')

    for s2 in os.listdir(image_sub_dir1):

        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)

        for s3 in os.listdir(image_sub_dir2):

            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)

            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg', '_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                if not os.path.exists(image_sub_dir4):
                    print('image not exists', image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print('label not exists', label_sub_dir4)
                    continue
                # image和label的list中添加数据集
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)
'''
image_dir = '/content/drive/My Drive/LaneSeg_p03colab/image'
label_dir = '/content/drive/My Drive/LaneSeg_p03colab/label'

for s1 in os.listdir(image_dir):

    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, s1)
    label_sub_dir1 = label_sub_dir1.replace('.jpg', '.png').replace('image', 'label')

    if not os.path.exists(image_sub_dir1):
        print('image not exists', image_sub_dir1)
        continue
    if not os.path.exists(label_sub_dir1):
        print('label not exists', label_sub_dir1)
        continue
    # image和label的list中添加数据集
    image_list.append(image_sub_dir1)
    label_list.append(label_sub_dir1)

# 判断两个列表长度是否一致
assert len(image_list) == len(label_list)
print(len(image_list), len(label_list))

# 将储存image和label图片地址的列表转成dataframe类型，打乱顺序
save = pd.DataFrame({'image': image_list, 'label': label_list})
save_shuffle = shuffle(save)

# 划分train、validation、test 7：1:2
# total:21914 train:15339 test:4383 validation:2192
# total:14264 train:998 test:285 validation:143
length = len(save_shuffle)
train = save_shuffle[0:int(length * 0.7)]
print(len(train))

test = save_shuffle[int(length * 0.7):int(length * 0.9)]
print(len(test))

validation = save_shuffle[int(length * 0.9):]
print(len(validation))
# 分别存成csv文件
train.to_csv('./data_list/train.csv', index=False)
test.to_csv('./data_list/test.csv', index=False)
validation.to_csv('./data_list/val.csv', index=False)

# file_path ='./data_list/train.csv'
# df = pd.read_csv(file_path)
# print(df.head(1))
# print(df.info())
