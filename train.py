#!/usr/bin/python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from model.config import Config
from tensorboardX import SummaryWriter

# GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设备列表
device_list = [0]

# 训练网络采用DeeplabV3Plus
train_net = 'deeplabv3p'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}



def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    # 定义：train_eopch函数，按batch（iteration），完成一次epoch计算, 更新模型权重
    # 输入：net, epoch, train_data_batch, optimizer, trainF, lane_config

    # 训练模式
    net.train()

    # dataprocess=train_data_batch，按batch_size分组得到的tensor形式的字典
    # batch_item:{'image':tensor[2,3,384,1024], 'mask':tensor[2 384 1024]} n c h w
    # 操作：batch_size的image和mask输入网络前向传播，计算loss，后向传播，更新权重
    # train_data_batch = 7669, len(train_data_batch) * batch_size = len(train_dataset)=15338
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        mask = mask.long()

        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])

        # 将优化器中更新过的节点对之前网络的保存记忆清除
        optimizer.zero_grad()
        # 得到预测
        out = net(image)
        # 交叉熵损失得到loss
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
        total_mask_loss += mask_loss.item()
        # 后向传播
        mask_loss.backward()
        # 更新权重
        optimizer.step()
        # 进度条描述
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))

    # 在trainF.csv中写入记录：Epoch:{}, mask loss is {:.4f}
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    # 把文件从内存buffer（缓冲区）中强制刷新到硬盘中，同时清空缓冲区
    trainF.flush()
    return  total_mask_loss


def test(net, epoch, dataLoader, testF, config):
    # 输入：net, epoch, train_data_batch, testF, lane_config
    # 测试状态
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    # {'TP': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    # 'TA': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}}
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    with torch.no_grad():
        # 从train_data_batch字典中按batch取出数据，每个bacth循环一次
        for batch_item in dataprocess:
            # 一个batch的image n c h w和mask图像 n h w
            image, mask = batch_item['image'], batch_item['mask']
            mask = mask.long()
            if torch.cuda.is_available():
                image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
            # 正向传播得到推断out  n c h w
            out = net(image)
            # 计算CrossEntropyLoss
            mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, mask)
            # 加入到总损失中
            total_mask_loss += mask_loss.detach().item()
            # softmax评分, 对channel进行合并,得到预测pred: n h w
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            # 调用compute_iou函数，结合pred与mask n h w 计算各类别的TP和TA
            result = compute_iou(pred, mask, result)
            # 进度条描述
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
        # 记录test log
        testF.write("Epoch:{} \n".format(epoch))
        iou = 0
        for i in range(8):
        # 计算每一类的iou, 打印到终端并写入testF文件
            result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
            print(result_string)
            iou += result["TP"][i]/result["TA"][i]
            testF.write(result_string)
        miou = iou/8
        # 将整个epoch的total_mask_loss写入testF文件
        testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
        testF.write("Epoch:{}, miou is {:.4f} \n".format(epoch, miou))
        # 把文件从内存buffer（缓冲区）中强制刷新到硬盘中，同时清空缓冲区
        testF.flush()


def adjust_lr(optimizer, epoch):
    # 学习率函数
    # 输入：优化器, epoch
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        lr = 1e-5
        return lr

    # 设置优化器参数组中的学习率参数
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    # 配置超参数 lane_config
    lane_config = Config()
    # 配置cuda：4线程并行计算（4张卡），batchsize的设置一定要可以被4整除（2张卡则要被2整除）
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    # 初始化网络：DeeplabV3Plus
    net = nets[train_net](lane_config)

    # 创建logs,如果文件夹logs不存在
    if not os.path.exists(lane_config.SAVE_PATH):
        # shutil.rmtree(lane_config.SAVE_PATH)
        # 创建名为logs的目录，如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常
        os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    # 在logs目录下建立train.csv，test.csv文件, 记为trainF, testF，记录log
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'a')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'a')


    # 准备数据: LaneDataset类的初始化(读入csv数据，做裁剪、放缩、数据增强以及label的id转换)
    # 输入：train.csv(path=cwd\data_list\filename),transform参数  输出：tensor形式的sample
    # 注意:使用transforms.Compose进行组织增强方法时，上一个方法的返回值要和下一个方法的输入对应上，否则会报错。
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                    ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    # 生成数据：调用Dataloader,加载训练数据, 按batch_size分为(total/n)组, 每组数据shape: n c h w
    # 输入：training_dataset，batch_size, shuffle=True, 不同batch之间进行shuffle
    # 参数：drop_last=True: 一个iteration中不足一个batch_size的样本扔掉，**kwargs: 配置cuda
    train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)
    # 生成验证集数据
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))
    val_data_batch = DataLoader(val_dataset, batch_size=4*len(device_list), shuffle=False, drop_last=False, **kwargs)
    

    # 将net传入cuda上
    if torch.cuda.is_available():
        print("cuda is available")
        net = net.cuda(device=device_list[0])
        # 在这里加了一个数据并行，相当于加了一个moduel
        #net = torch.nn.DataParallel(net, device_ids=device_list)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)

    # 定义adam优化器(传入模型参数，学习率，权重的l2损失)
    optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    

    # 定义Resume 恢复训练
    Resume = True
    # 定义epoch断点为65(大概总体epoch的一半)
    #epoch_to_continue = 65
    epoch_to_continue = 975

    # 若是要恢复训练，则在Resume块中重新加载参数
    if Resume is True:
        # 定义checkpoint路径=save路径(当前路径\logs\epoch{}Net.pth.tar)
        checkpoint_path = os.path.join(os.getcwd(), lane_config.SAVE_PATH, "epoch{}Net.pth.tar".format(epoch_to_continue))
        if not os.path.exists(checkpoint_path):
            print("checkpoint_path not exists!")
        else:
            # 从checkpoint_path路径加载模型到内存,设置map_location参数只定加载到gpu进程上
            checkpoint = torch.load(checkpoint_path, map_location = 'cuda:{}'.format(device_list[0]))
            # model_param = torch.load(checkpoint_path)['state_dict']
            # model_param = {k.replace('module.', ''):v for k, v in model_param.items()}

            # 从内存中加载模型参数字典
            net.load_state_dict(checkpoint['state_dict'])
            # 加载optimizer参数字典
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 更新断点位置
            epoch_to_continue = checkpoint['epoch']
    
    # 加入数据并行
    if torch.cuda.is_available():
        # 在这里加了一个数据并行，相当于加了一个moduel
        net = torch.nn.DataParallel(net, device_ids=device_list)


    # 开始训练(按config中设定的epoch进行循环50个epoch，有断点从断点处执行)
    for epoch in range(epoch_to_continue+1, epoch_to_continue+lane_config.EPOCHS):
        # 调用学习率函数
        lr = adjust_lr(optimizer, epoch)
        # 调用train_epoch，每个epoch内，按batch进行迭代更新模型权重，完成一次所有数据的训练
        total_mask_loss = train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        # 每5个epoch储存一下参数
        if epoch % 5 == 0:
            # 存储的参数是net的模型参数，没有网络结构
            #torch.save({'state_dict': net.module.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
            #torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
            torch.save({
                        'epoch':epoch,
                        'state_dict': net.module.state_dict(), # 加了module
                        'optimizer_state_dict':optimizer.state_dict(),
                       }, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "epoch{}Net.pth.tar".format(epoch)))
        # 每1个epoch都测试一下
        test(net, epoch, val_data_batch, testF, lane_config)
        writer = SummaryWriter()
        writer.add_scalar('learning_rate', lr, epoch)
        writer.add_scalar('loss', total_mask_loss, epoch)
    writer.export_scalars_to_json("./train.json")
    writer.close()
    # 关闭log记录文件
    trainF.close()
    testF.close()
    #torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))

   
# 执行训练函数
if __name__ == "__main__":
    main()
