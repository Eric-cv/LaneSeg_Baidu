#!/usr/bin/python
# -*- coding: utf-8 -*-
class Config(object):
    # 参数管理：模型参数，训练参数，用于初始化实例
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8
    # train config
    #EPOCHS = 50
    EPOCHS = 200
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.0006


