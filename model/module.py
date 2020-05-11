#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # 常规 block: input-->conv-->bn-->relu -->out （默认3*3的conv标准模式）
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class ResBlock(nn.Module):
    # resblock: input-->bn-->relu-->conv-->out （相对于常规block更改了顺序，默认3*3的conv标准模式）
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    # over 50 layers
    # 常规 Bottleneck：1*1 下降维度，3*3 保持维度不变，1*1 升高维度  + shortcut
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out



class DownBottleneck(nn.Module):
    # over 50 layers
    # 下采样 Bottleneck：1*1 下降维度/4，3*3 保持维度不变，1*1 升高维度*4 + shortcut + stride=2
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=2):
        super(DownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)  # shortcut支路下采样
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = self.conv1(x)  # shortcut支路下采样
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


def make_layers(in_channels, layer_list, name="vgg"):
    # 构造不同网络结构下的各层的函数，layers list + sequential
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        # resnet有5组卷积，每组卷积有x个Bottleneck (Bottleneck*x), 每组卷积的第一个Bottleneck都是DownBottleneck
        layers += [DownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        # 剩下的x-1个Bottleneck都是常规做Bottleneck
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]
            # 输出替换下一次输入
            in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    # 给make_layers函数传具体的参数（in_channels, layer_list, name=net_name）前向传播，return feature map
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out


class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling(空洞空间金字塔池化)
    # input: x

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        # 定义平行操作branch1: 1*1conv 不改变size
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        # 定义平行操作branch2：3*3conv dilation=6 不改变size
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        # 定义平行操作branch3：3*3conv dilation=12 不改变size
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        # 定义平行操作branch4：3*3conv dilation=18 不改变size
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        # 定义平行操作branch5:定义全局平均池化操作+1*1卷积
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 定义conv_cat: 1*1conv下降输出维度至输入维度的1/5+bn+relu
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        # 使用branch1-4: 1*1 和 3*3 dilated卷积+bn+relu，得到4个尺寸相同的output（b c h w）
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 使用branch5：全局平均池化+1*1卷积+bn+relu，再进行双线性插值上采样回到（b c h w）
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        # 5个branch按维度cat，得到feature map (torch.cat 是哪个维度的叠加？)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 对feature map调整维度，降到1维，+ bn + relu
        result = self.conv_cat(feature_cat)

        # 返回经过ASPP空洞卷积多尺度融合后的feature map
        return result


