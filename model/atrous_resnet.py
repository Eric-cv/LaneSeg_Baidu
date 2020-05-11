#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# model_zoo是和导入预训练模型相关的包
bn_mom = 0.0003

# model_urls这个字典是预训练模型权重的下载地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    # BasicBlock(under 50 layers)
    # 3*3 维度不变，3*3 维度不变（in_channels, out_channels）+ shortcut + atrous
    # input:in_chans,out_chans,默认stride=1, atrous=1, downsample=None
    # output:out_chans
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride, atrous)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck (over 50 layers)
    # 1*1 维度不变，3*3 维度不变，1*1 升高维度 （in_channels, out_channels* expansion）+ shortcut + atrous
    # input:in_chans,out_chans,默认stride=1, atrous=1, downsample=None
    # output:out_chans*expansion

    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        # 初始化stride_list 决定每层block第一个residual是否进行2倍下采样的stride参数列表
        stride_list = None
        # 如果outstride = 8，即下采样到8倍,则采用stride_list = [2, 1, 1]
        if os == 8:
            stride_list = [2, 1, 1]
        # 如果outstride = 16，即下采样到16倍，则采用stride_list = [2, 2, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            # 只支持下采样8倍和16倍
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        # conv1: out_chans:64 outstride = 2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # maxpool: out_chans:64 outstride = 4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2 out_chans: 64*4=256  注意conv2不做下采样
        # 调用时传入的参数 layers = [3,4,6,3]  (under 50 layers), layers = [3,4,23,3]  (over 50 layers)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])

        # conv3 out_chans: 128*4=512, stride=2, outstride=8
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])

        # conv4 out_chans: 256*4=1024, stride=2, outstride=16
        # 在此不再进行下采样，使用空洞卷积增大感受野，atrous =1,
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        #self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=[item * 16 // os for item in atrous])

        # conv5 out_chans:512*4=2048, stride=1,调用时传入的参数 atrous = [1, 2, 1], 相当于outstride=64
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2], atrous=[item * 16 // os for item in atrous])

        # conv6 out_chans:512*4=2048, stride=1, atrous = [1, 2, 1], 相当于outstride=128
        self.layer5 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        # conv7 out_chans:512*4=2048, stride=1, atrous = [1, 2, 1], 相当于outstride=256
        self.layer6 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layers = []

        # kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, in_chans, out_chans, blocks, stride=1, atrous=None):
        # 作用：构建每层的block。
        # 传入参数：block类型(BasicBlock Bottleneck)，输入输出channel，该层block中含有的residual个数(如3 or 23...)，默认stride=1，atrous=None
        # 返回值：该层block的sequential()序列
        downsample = None
        # 初始化atrous_list
        # 若传入的atrous为空，则atrous = [1,1,...]，即residual不做空洞卷积
        if atrous == None:
            atrous = [1] * blocks
        # 若传入的atrous不为空，则atrous = [atrous,atrous,...] [[1,2,1], [1,2,1]...]， 即residual按系数做空洞卷积
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list

        # 初始化下采样：restnet中如果传入的步长不等于1 或者in_channel 不等于out_channel的4倍就定义下采样操作
        if stride != 1 or in_chans != out_chans * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        # 定义储存该层系列操作的列表
        layers = []
        # 每个block有x个residual，每个blocks的第一个residual结构都要进行下采样，将此操作储存在列表中
        layers.append(block(in_chans, out_chans, stride=stride, atrous=atrous[0], downsample=downsample))

        in_chans = out_chans*4
        for i in range(1, blocks):
            # 每个blocks的剩下x-1 个residual结构保存在layers列表中，这样就完成了一个blocks的构造
            layers.append(block(in_chans, out_chans, stride=1, atrous=atrous[i]))


        return nn.Sequential(*layers)

    def forward(self, x):
        layers_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layers_list.append(x)

        x = self.layer2(x)
        layers_list.append(x)

        x = self.layer3(x)
        layers_list.append(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        layers_list.append(x)

        # 返回 outstride=4, 8, 16（atrous=2）, 16(atrous=2*2*2*2) 时的 不同layers下的x list
        # outchannel 分别对应为：256 512 1024 2048
        return layers_list

# resnet50这个函数，参数pretrained默认是False
# 如果参数pretrained=True，那么就会通过model_zoo.py中的load_url函数根据model_urls字典下载或导入相应的预训练模型。
# 通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。
# if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))，
# 主要就是通过model_zoo.py中的load_url函数根据model_urls字典导入相应的预训练模型
def resnet50_atrous(pretrained=True, os=16, **kwargs):
    # Constructs a atrous ResNet-50 model.
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        # 按照地址取出res50的预训练模型
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        # 取出model中的权重参数
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        # 使用预训练模型参数更新model中的参数
        model_dict.update(old_dict)
        # 模型加载更新的权重
        model.load_state_dict(model_dict)
        # 返回resnet50_atrous model
    return model


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
