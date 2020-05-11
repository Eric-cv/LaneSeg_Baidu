#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from utils.process_labels import encode_labels, decode_labels, decode_color_labels

# 对百分之50的图片应用aug的数据增强方法,另外50不用
sometimes = lambda aug: iaa.Sometimes(0.5, aug)


# crop the image to discard useless parts
def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    # 裁剪和resize函数：
    # 输入：image,label(默认为0)，ideal size，y方向裁剪起始点offset(默认690)
    # 输出：image and label after croping and resizing
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    # original size:3384*1710
    # roi_image (after crop）:3384*1020
    roi_image = image[offset:,:]
    # label存在时，进一步resize为理想size:1024*384
    # 训练时传label，对image和label都做resize(插值放缩)，测试时不传label，只对image做resize
    if label is not None:
        roi_label = label[offset:, :]
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image


class LaneDataset(Dataset):
    # 准备数据：声明数据集类（继承Dataset类）：定义__getitem__, __len__函数
    # 输入：csv 默认transform=None  输出：sample
    def __init__(self, csv_file, transform=None):
        super(LaneDataset, self).__init__()
        # 定义self.data，以DataFame的形式，读取csv文件
        self.data = pd.read_csv(os.path.join(os.getcwd(), "utils\data_list", csv_file), header=None,
                                  names=["image", "label"])
        #self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,
        #                          names=["image", "label"])
        # 定义self.images和self.labels以series形式存放image和label地址
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]
        # 定义增强参数
        self.transform = transform

    # 计算epoch时候需要dataset的长度
    def __len__(self):
        # 求labels数量
        # 输出：label的总数量
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # __getitem__函数，单张读取图片
        # 输入：图片的idx， 输出：经过裁剪、放缩、数据增强以及对label转换id的train_image,train_mask
        # 按照地址读入ori_image ori_label
        ori_image = cv2.imread(self.images[idx])
        if(ori_image is None):
            print("idx:{},path:{}".format(idx, self.images[idx]))
        ori_mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        if(ori_mask is None):
            print("idx:{},path:{}".format(idx,self.labels[idx]))
        # 对ori_image和ori_mask都进行裁剪和放缩得到理想size的图片,train_img,train_mask
        train_img, train_mask = crop_resize_data(ori_image, ori_mask)
        # 对mask进行id 转换--->train_mask
        train_mask = encode_labels(train_mask)
        # 复制train_img,train_mask 放入sample列表
        sample = [train_img.copy(), train_mask.copy()]
        # 有数据增强就做增强
        if self.transform:
            sample = self.transform(sample)
        return sample



class ImageAug(object):
    # pixel augmentation
    # 50%概率 做高斯噪声，锐化，高斯模糊
    # 输入：sample 输出：image mask
    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0,1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
            # 将方法应用在原图像上
            image = seq.augment_image(image)   # seq.augment_image?
        return image, mask



class DeformAug(object):
    # deformation augmentation
    # crop and pad  裁剪和填充
    # 输入：sample 输出：image mask
    def __call__(self, sample):
        image, mask = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        # 确定一个数据增强的序列
        seg_to = seq.to_deterministic()
        # 将方法应用在原图像上
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        return image, mask


class ScaleAug(object):
    # 尺度放缩
    # 输入：sample 输出：aug_image aug_mask
    def __call__(self, sample):
        image, mask = sample
        # 定义放缩系数scale
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()
        # 对image和mask进行放缩
        aug_image = cv2.resize(aug_image, (int (scale * w), int (scale * h)))
        aug_mask = cv2.resize(aug_mask, (int (scale * w), int (scale * h)))
        # 放缩系数小于1，padding
        if (scale < 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        # 放缩系数大于1，裁剪
        if (scale > 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int ((new_h - h) / 2)
            pre_w_crop = int ((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return aug_image, aug_mask


class CutOut(object):
    # 随机擦除/遮挡
    # 输入：sample  输出：image mask
    def __init__(self, mask_size, p):
        # 定义遮挡size，随机遮挡概率p
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        # 定义mask_size的一半
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]
        # 确定mask的中心位置范围
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
        # 在确定的范围内随机产生mask中心
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        # 产生mask坐标
        xmin, ymin = cx - mask_size_half, cy - mask_size_half
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
        # 随机产生mask（image上全0值）
        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return image, mask


class ToTensor(object):
    # np.transpose对image进行空间转置：np.array形式，再转成tensor形式
    # 输入：sample  输出：image和mask的tensor字典
    def __call__(self, sample):

        image, mask = sample
        # 对图像进行转置（坐标轴交换），列，维度，行，并制定类型为浮点型
        image = np.transpose(image,(2,0,1))
        image = image.astype(np.float32)
        # 提高浮点数精度
        mask = mask.astype(np.long)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}


def expand_resize_data(prediction=None, submission_size=(3384, 1710), offset=690):
    pred_mask = decode_labels(prediction)
    expand_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask


def expand_resize_color_data(prediction=None, submission_size=(3384, 1710), offset=690):
    color_pred_mask = decode_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
    color_expand_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask
