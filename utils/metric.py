#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def compute_iou(pred, gt, result):
    # 计算iou
    # 输入：pred, gt, result字典储存列表
    # 输出：{'TP': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    #      'TA': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}}
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    # 对pred gt 转化成numpy形式
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt = gt==i
        single_pred = pred==i
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        # 记录不同类别的TP TA,存入result字典中
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result
