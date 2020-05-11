#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.network import ResNet101v2
from model.module import Block


class UNetConvBlock(nn.Module):
    # 定义UnetConvBlock:连续的两个3*3conv+bn+relu, padding 可以自定义，默认为0，常规3*3conv out_size=in_size-2
    # Conv2d的默认参数是：torch.nn.Conv2d(in_channels, out_channels, kernel_size,
    # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    # 定义上采样block（上采样函数，中心裁剪函数，上采样+裁剪+拼接+卷积函数）
    # input: x , bridge
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(UNetUpBlock, self).__init__()
        # 转置卷积模式上采样：up函数，stride= input_size/output_size, out = (in-1)*s -2p + k
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        # 双线性插值模式上采样：up函数
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, True)

    # 定义中心裁剪函数，传入现有的feature map size和目标size
    def center_crop(self, layer, target_size):
        # n c w h 返回现有feature map的 w h
        _, _, layer_height, layer_width = layer.size()
        # 计算高度差，宽度差
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2

        # 返回feature map按target中心裁剪后的结果
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        # 上采样+裁剪+拼接+卷积
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        # 返回上采样+裁剪+拼接+卷积的结果
        return out

class ResNetUNet(nn.Module):
    # 定义模型config和训练config
    def __init__(
        self,
        config
    ):
        super(ResNetUNet, self).__init__()
        #网络参数要跟deeplabv3p一样的参数，是同一个config
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        #encode改成resnet101v2
        self.encode = ResNet101v2()
        #上一层的给出的就是2048
        prev_channels = 2048

        # 定义self.up_path为nn.Modulelist()，存放三个UnetUpBlock上采样模块，输出channel为输入channel的一半，循环三次
        # [UNetUpBlock(2048, 1024),UNetUpBlock(1024, 512),UNetUpBlock(512, 256)]
        self.up_path = nn.ModuleList()

        for i in range(3):
            self.up_path.append(
                UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding)
            )
            prev_channels //= 2

        # block: input-->conv-->bn-->relu -->out （默认3*3的conv标准模式）
        # 定义self.conv_block1,conv_block2，使用3*3常规conv-->bn-->relu -->out模式,下降维度到32维-->16维
        self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        # 定义self.last 1*1conv下降维度到self.n_classes
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)

        # 每层都进行kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encode(resnet101) return 就是f2到f5 4个不同layer下的output, outstride = 4,8,16,32
        input_size = x.size()[2:]
        blocks = self.encode(x)
        # 最后一个作为上采样的输入,outstride = 32
        x = blocks[-1]

        # 对up_path进行for循环
        for i, up in enumerate(self.up_path):

            # 将最后输出的outstride=64的x与前面blocks[-2,-3,-4]上outstirde=16,8,4的三个x都依次进行上采样（上采样+裁剪+拼接+卷积）
            # 得到stride=4的上采样融合特征map
            x = up(x, blocks[-i - 2])

        # 再用双线性插值上采样成输入的尺寸，align_corners的意思是rensize的时候，边缘是不是跟原图对齐
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        # 3*3连续降维到32到16
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        # 1*1降维到n_classes
        x = self.last(x)
        return x
