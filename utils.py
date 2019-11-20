import os
import time
import datetime

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from resnet_skip import SkipResnet50
from resnet import ResNet, Bottleneck
from first_pixel import FirstPixel
from sklearn.preprocessing import MinMaxScaler


from skimage import measure

from torch.autograd import Variable
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def PathList(path, train_ratio):

        jpg_list = []
        png_list = []
        jpg_list_train = []
        jpg_list_test = []
        png_list_train = []
        png_list_test = []
        path_list = os.listdir(path)
        path_list.sort()
        for files in path_list: 
            if os.path.splitext(files)[1]=='.jpg': 
                jpg_list.append(files)
            else:
                png_list.append(files)
        seed =50
        random.seed(seed)
        random.shuffle(jpg_list)
        random.seed(seed)
        random.shuffle(png_list)
        train_size = int(len(jpg_list)*train_ratio)
        jpg_list_train = jpg_list[0:train_size]
        jpg_list_test = jpg_list[train_size:len(jpg_list)]
        png_list_train = png_list[0:train_size]
        png_list_test = png_list[train_size:len(png_list)]
            
        return jpg_list_train, jpg_list_test, png_list_train, png_list_test

class MakeDataset(Dataset):
    def __init__(self, baseroot, jpglist, pnglist):                   
        self.baseroot = baseroot
        self.jpglist = jpglist
        self.pnglist = pnglist
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform = transforms.Normalize((0.5,), (0.5,))
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        
    def __len__(self):
        return len(self.jpglist)

    def __getitem__(self, index):
        jpgpath = os.path.join (self.baseroot, self.jpglist[index])                
        pngpath = os.path.join (self.baseroot, self.pnglist[index])              
        jpgimg_ori = cv2.imread(jpgpath,cv2.IMREAD_COLOR)
        jpgimg = cv2.cvtColor(jpgimg_ori, cv2.COLOR_BGR2RGB)

        pngimg = cv2.imread(pngpath,cv2.IMREAD_COLOR)
        pngimg = cv2.cvtColor(pngimg, cv2.COLOR_BGR2RGB)
                                    

        jpgimg = jpgimg[58:426, 143:511, :]
        pngimg = pngimg[58:426, 143:511, :]
        inputs = self.transform2(jpgimg)
        
        im = cv2.cvtColor(jpgimg, cv2.COLOR_RGB2GRAY)
        im = im.reshape([368, 368, 1])
        inputs_gray = self.transform1(im)

        x = cv2.Sobel(im, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(im, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst = dst.reshape([368, 368, 1])        
        inputs_sobel = self.transform1(dst)
        
        # laplacian
        gray_lap = cv2.Laplacian(im, cv2.CV_16S, ksize=3)
        dst = cv2.convertScaleAbs(gray_lap)
        dst = dst.reshape([368, 368, 1])        
        inputs_laplacian = self.transform1(dst)


        pngimg = cv2.cvtColor(pngimg, cv2.COLOR_RGB2GRAY)
        targets = torch.from_numpy(pngimg)


        contour = measure.find_contours(pngimg, 0.5)
        contour = contour[0]
        con_array = np.array(contour)
        con_array = np.squeeze(contour).astype(np.int)
        contour_GT = con_array
        targets_pixel = torch.from_numpy(contour_GT)

        return inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_pixel








