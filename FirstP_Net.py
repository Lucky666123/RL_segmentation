from __future__ import division

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
from grid_features import GridFeatures
from skimage import morphology


from skimage import measure

from torch.autograd import Variable
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
from utils import PathList
from encoder import EncoderNet

from sklearn.preprocessing import MinMaxScaler



import shapely.geometry as geom
import copy
import matplotlib.patches as patches
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MakeDataset(Dataset):
    def __init__(self, baseroot, jpglist, pnglist):                   
        self.baseroot = baseroot
        self.jpglist = jpglist
        self.pnglist = pnglist
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform = transforms.ToTensor()
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

        xs = con_array[:, 1]/float(opt.img_size[1])
        ys = con_array[:, 0]/float(opt.img_size[0])
        contour_point = np.array([ys, xs]).T
        sz=[]
        sz[:] = [x-1 for x in opt.grid_size]
        contour_point = np.floor(contour_point * sz).astype(np.int32)
        contour_GT = cv2.approxPolyDP(contour_point, 0, False)[:, 0, :]

        contour_mask = np.zeros((opt.grid_size[1], opt.grid_size[0]), np.float32)
        contour_mask[contour_GT[:, 0], contour_GT[:, 1]] = 1

        targets_mask = torch.from_numpy(contour_mask)

        return inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_mask, targets_pixel


class FirstPNet(nn.Module):
    def __init__(self):
        super(FirstPNet, self).__init__()

        print 'Building Encoder'
        self.resnet_skip = SkipResnet50()
        self.first_pixel = FirstPixel()


    def forward(self, x):

        """
        x: [bs, 3, 100, 75]
        """

        def ConvertPixel(pred_pixel, feat_grid_size= opt.grid_size, ori_grid_size = opt.img_size):

            curr_p = pred_pixel
            x = (curr_p / feat_grid_size[0]) + 1
            y = (curr_p % feat_grid_size[0]) + 1
            
            ori_x = x * 8
            ori_y = y * 8
            ori_p = torch.stack([ori_x,ori_y], dim = 1)

            return ori_p

        concat_features, final_features = self.resnet_skip(x)

        conv_pixel, pixel_logits, pred_first = self.first_pixel(final_features)

        ori_p = ConvertPixel(pred_first)
        ori_p = ori_p.squeeze()

        return final_features, conv_pixel, pixel_logits, pred_first, ori_p

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 60, help = 'number of epochs of training')
parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
parser.add_argument('--lr', type = float, default = 1e-5, help='Adam: learning rate')
parser.add_argument('--weight_decay', type = float, default = 1e-5, help = 'Adam: weight-decay')
parser.add_argument('--loss_decay', type = int, default = 52, help = 'weight decline for optimizer')
parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type = int, default = 10, help = 'interval between model checkpoints')
parser.add_argument('--grid_size', type = list, default = [46, 46], help = 'grid size')
parser.add_argument('--img_size', type = list, default = [368, 368], help = 'image size')
parser.add_argument('--resnet_path', type = str, default = "/home/xjj/Desktop/5.26/MSRA10K_Imgs_GT/resnet50.pth", help = 'resnet50 path')


opt = parser.parse_args()

def Trainer(opt):

    cudnn.benchmack = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def adjust_learning_rate(optimizer, epoch, opt):
        lr = opt.lr * (0.1 ** (epoch // opt.loss_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # ----------------------------------------------
    #                  Network Dataset
    # ----------------------------------------------
    path = os.path.join(os.path.dirname(__file__),"LV_small_RoI")
    jpg_list_train, jpg_list_test, png_list_train, png_list_test = PathList(path, train_ratio = 0.7)

    TrainDataset = MakeDataset(path, jpg_list_train, png_list_train) 

    print 'The number of training data:', len(jpg_list_train)


    TrainLoader = DataLoader(TrainDataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)

    FirstP = FirstPNet().cuda()

    FirstP.train() 
    optimizer = optim.Adam(FirstP.parameters(), lr=opt.lr) 

    
    all_epoch_loss = []
    prev_time = time.time()

    for epoch in range(opt.epochs):
        train_loss = []
        for batch_idx, (inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_mask, targets_pixel) in enumerate(TrainLoader):
            inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_mask, targets_pixel = inputs.cuda(), \
                                 inputs_gray.cuda(), inputs_sobel.cuda(), inputs_laplacian.cuda(), targets.cuda(), targets_mask.cuda(), targets_pixel.cuda()
            optimizer.zero_grad
            final_features, conv_pixel, pixel_logits, pred_first, ori_p = FirstP(inputs)
            
            targets_mask = targets_mask.view(opt.batch_size, -1)

            loss = F.binary_cross_entropy_with_logits(pixel_logits, targets_mask)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.array(train_loss)

        # Determine approximate time left
        batches_done = epoch * len(TrainLoader) + batch_idx
        batches_left = opt.epochs * len(TrainLoader) - batches_done
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time) / batches_done)

        print("\r[Epoch %d/%d] [loss: %0.5f] [time left: %s]" % ((epoch+1), opt.epochs, train_loss.mean(), time_left))

        if (epoch+1) % opt.checkpoint_interval == 0:
            torch.save(FirstP, 'FirstPNet_LV_small_RoI_epoch%d.pth' % (epoch+1))
            print('model is saved!')

        adjust_learning_rate(optimizer, epoch, opt)
        all_epoch_loss.append(float(train_loss.mean()))

    plt.plot(all_epoch_loss)
    plt.show()

# Trainer(opt)

def Testing():

    cudnn.benchmack = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # ----------------------------------------------
    #                  Network Dataset
    # ----------------------------------------------
    path = os.path.join(os.path.dirname(__file__),"LV_small_RoI")
    jpg_list_train, jpg_list_test, png_list_train, png_list_test = PathList(path, train_ratio = 0.7)


    TestDataset = MakeDataset(path, jpg_list_test, png_list_test) 

    print 'The number of testng data:', len(jpg_list_test)


    TestLoader = DataLoader(TestDataset, batch_size = 1, shuffle = False, num_workers = opt.num_workers)

    FirstP = torch.load('FirstPNet_LV_small_RoI_epoch60.pth').cuda()

    for batch_idx, (inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_mask, targets_pixel) in enumerate(TestLoader):
                
        inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_mask, targets_pixel = inputs.cuda(), \
                                inputs_gray.cuda(), inputs_sobel.cuda(), inputs_laplacian.cuda(), targets.cuda(), targets_mask.cuda(), targets_pixel.cuda()

        final_features, conv_pixel, pixel_logits, pred_first, ori_p = FirstP(inputs)

        img = inputs.cpu().squeeze()
        targets_pixel = targets_pixel.cpu().numpy().squeeze()
        
        img = np.transpose(img,(1,2,0))
        img = 255*(img*0.5+0.5)
        img_numpy = np.array(img, dtype = np.uint8)
        ax = plt.gca()
        ax.imshow(img_numpy, cmap = 'gray')
        plt.axis('off')
        ax.plot(targets_pixel[:,1], targets_pixel[:,0], marker = '.', c ='b')
        ax.plot(ori_p[1], ori_p[0], markersize = 10, marker = '*', c = 'r')
        ax.set_title(str(jpg_list_test[batch_idx]))
        plt.show()


        targets_mask = targets_mask.view(1, -1)
        loss = F.binary_cross_entropy_with_logits(pixel_logits, targets_mask)
        loss = torch.mean(loss)
        
        print("\r[loss: %0.5f]" % (loss.item()))


# Testing()
