#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.atrous_resnet import resnet50_atrous
from model.module import ASPP
from model.config import Config as cfg


class DeeplabV3Plus(nn.Module):
    # backbone:resnet50_atrous + ASPP
    def __init__(self, cfg):
        super(DeeplabV3Plus, self).__init__()
        # 定义DeeplavV3Plus的backbone使用res50_atrous 返回值是outstride=2, 4, 8, 1 时不同layers下的x list
        self.backbone = resnet50_atrous(pretrained=True, os=cfg.OUTPUT_STRIDE)
        input_channel = 2048
        # 定义aspp, 默认参数为：ASPP(2048, 256, 1)
        self.aspp = ASPP(in_chans=input_channel, out_chans=cfg.ASPP_OUTDIM, rate=16//cfg.OUTPUT_STRIDE)
        # 定义droput
        self.dropout1 = nn.Dropout(0.5)
        # 定义双线性插值上采样UpsamplingBilinear2d，默认参数为4倍
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.OUTPUT_STRIDE//4)

        indim = 256
        # 定义shortcut_conv(Sequential操作),nn.Conv2d(256, 48, 1, 1, 0, bias=False)+bn+relu
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.SHORTCUT_DIM, cfg.SHORTCUT_KERNEL, 1, padding=cfg.SHORTCUT_KERNEL//2, bias=False),
                nn.BatchNorm2d(cfg.SHORTCUT_DIM),
                nn.ReLU(inplace=True),
        )
        # 定义cat_conv(Sequential操作),nn.Conv2d(256+48, 256, 3, 1, 1, bias=False)+bn+relu+dropout+...
        # nn.Conv2d(256, 256, 3, 1, 1, bias=False)+bn+relu+dropout
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.ASPP_OUTDIM+cfg.SHORTCUT_DIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.ASPP_OUTDIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(cfg.ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        # 定义cls.conv = nn.Conv2d(256, 8, 1, 1, 0)
        self.cls_conv = nn.Conv2d(cfg.ASPP_OUTDIM, cfg.NUM_CLASSES, 1, 1, padding=0)

        # 每层进行kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): # 没做ASPP ?
        # 使用res50_atrous backbone返回值：不同layers下的x list时不同layers下的x list
        # outstride=4, 8, 16（atrous=2), 16(atrous=2*2*2*2) outchannel 分别对应为：256 512 1024 2048
        layers = self.backbone(x)
        # 取outstride=16,inchannel=2048 即经过atrous空洞卷积增大感受野16倍的out,outchannel=256
        feature_aspp = self.aspp(layers[-1])
        # 对其dropuot
        feature_aspp = self.dropout1(feature_aspp)
        # 四倍双线性插值上采样,feature_aspp的size为：outside=4
        feature_aspp = self.upsample_sub(feature_aspp)
        # 取outstride=4的out，inchannel=256,进行1*1conv转变channel为48+bn+relu,outchannel=48
        feature_shallow = self.shortcut_conv(layers[0])
        # 对其进行channel方向拼接得到feature_cat,channel=256+48
        feature_cat = torch.cat([feature_aspp, feature_shallow],1)
        # 3*3conv将维度下降到256，+bn+relu+dropout
        result = self.cat_conv(feature_cat)
        # 1*1conv下降维度到8
        result = self.cls_conv(result)
        # 再上采样4倍回到outstride=1，channel=8
        result = self.upsample4(result)
        # 返回x,outstride=1，channel=8
        return result
