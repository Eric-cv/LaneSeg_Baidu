from tqdm import tqdm
import torch
import os
import pandas as pd
from PIL import Image
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
from torch.autograd import Variable

# print(torch.__version__)
# print(torch.version.cuda)
path_train = r'H:\ZQ_file\project03_colab\utils\data_list\train.csv'
data = pd.read_csv(path_train, header=None, names=["image", "label"])
images = data["image"].values[1:]
labels = data["label"].values[1:]

for idx in tqdm(range(len(images))):
    img = Image.open(images[idx])
    lab = Image.open(labels[idx])
    img.save(r'H:\ZQ_file\LaneSeg_p03colab\image\image_%03s.jpg' % idx)
    lab.save(r'H:\ZQ_file\LaneSeg_p03colab\label\label_%03s.png' % idx)

print('finish')
# path_val = r'H:\ZQ_file\project03_colab\utils\data_list\val.csv'
# data = pd.read_csv(path_val, header=None, names=["image", "label"])
# images = data["image"].values[1:]
# labels = data["label"].values[1:]
#
# for idx in range(len(images)):
#     img = Image.open(images[idx])
#     lab = Image.open(labels[idx])
#     img.save(r'H:\ZQ_file\LaneSeg_p03colab\val_data\image\image_%03s.jpg' % idx)
#     lab.save(r'H:\ZQ_file\LaneSeg_p03colab\val_data\label\label_%03s.png' % idx)
