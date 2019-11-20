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
from resnet_skip import SkipResnet50

from skimage import measure

from torch.autograd import Variable
from roialign.roi_align.crop_and_resize import CropAndResizeFunction


def GridFeatures(ori_p, radius, pool_size, img_size, feature_map):

    # pool_size = 7
    bbox1 = ori_p - radius
    bbox2 = ori_p + radius

    bbox = torch.stack([bbox1,bbox2], dim=0)
    bbox = bbox.reshape(-1, 4)
    height = img_size[0]
    width = img_size[1]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = torch.stack( \
    [bbox[:, 0].clamp(float(window[0]), float(window[2])),
        bbox[:, 1].clamp(float(window[1]), float(window[3])),
        bbox[:, 2].clamp(float(window[0]), float(window[2])),
        bbox[:, 3].clamp(float(window[1]), float(window[3]))], 1).float()  # long tensor

    # Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False).cuda()  # long tensor
    normalized_boxes = boxes / norm

    # Add back batch dimension
    # normalized_boxes = normalized_boxes.unsqueeze(0)

    ind = Variable(torch.zeros(boxes.size()[0]),requires_grad=False).int().cuda()
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_map, normalized_boxes, ind) # final_features ---> image_inputs
        # avg_pool = nn.AdaptiveAvgPool2d(1)
        # avg_features = avg_pool(pooled_features)
        # avg_features = torch.squeeze(avg_features, 2)
        # avg_features = torch.squeeze(avg_features, 2)
    # import pdb; pdb.set_trace()

    return pooled_features