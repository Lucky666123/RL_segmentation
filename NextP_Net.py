from __future__ import division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from utils import MakeDataset
from encoder import EncoderNet
from FirstP_Net import FirstPNet
from sklearn.preprocessing import MinMaxScaler


import shapely.geometry as geom
import copy
import matplotlib.patches as patches
import math
from scipy.fftpack import fft,ifft,fftshift


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.EncoderNet = EncoderNet()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 8)


    def forward(self, x):

        features = self.EncoderNet(x)   
        avg_features = self.avg_pool(features)
        avg_features = avg_features.view(-1,512)
        action_prob = F.relu(self.fc1(avg_features))
        return action_prob


class DQN(nn.Module):

    def __init__(self, opt):
        super(DQN, self).__init__()

        self.EPISILON = 0.8      
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        for name, p in self.eval_net.named_parameters():
            print(name)
        for name, p in self.target_net.named_parameters():
            p.requires_grad = False

        self.opt = opt
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((opt.MEMORY_CAPACITY, opt.NUM_STATES * 2 + 2), dtype = np.float32)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=opt.LR)
        self.loss_func = nn.MSELoss()
        self.polyak = 0.995


    def choose_action(self, state):

        if np.random.rand() <= self.EPISILON:# greedy policy
            action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].item()
        else: # random policy
            action = np.random.randint(0,self.opt.NUM_ACTIONS)
        return action

    def choose_action_test(self, state):
        
        action_value = self.eval_net(state)
        action = torch.max(action_value, 1)[1].item()
        
       
        return action


    def store_transition(self, state, action, reward, next_state):
        state = state.cpu().detach().numpy().squeeze()
        next_state = next_state.cpu().detach().numpy().squeeze()
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.opt.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        if self.learn_step_counter != 0 and self.learn_step_counter % self.opt.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\033[1;31mUpdate target net parameters!\033[0m')

        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(self.opt.MEMORY_CAPACITY, self.opt.BATCH_SIZE, replace = False)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.cuda.FloatTensor(batch_memory[:, :self.opt.NUM_STATES])
        batch_action = torch.cuda.LongTensor(batch_memory[:, self.opt.NUM_STATES:self.opt.NUM_STATES+1].astype(int))
        batch_reward = torch.cuda.FloatTensor(batch_memory[:, self.opt.NUM_STATES+1:self.opt.NUM_STATES+2])
        batch_next_state = torch.cuda.FloatTensor(batch_memory[:,-self.opt.NUM_STATES:])

        #q_eval
        batch_state = batch_state.view(-1, self.opt.layers, self.opt.grid_size[0], self.opt.grid_size[1])
        batch_next_state = batch_next_state.view(-1, self.opt.layers, self.opt.grid_size[0], self.opt.grid_size[1])
        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # gather index LongTensor
        q_next = self.target_net(batch_next_state).detach()
        ## Vanilla Q learning
        # q_target = batch_reward + self.opt.GAMMA * q_next.max(1)[0].view(self.opt.BATCH_SIZE, 1)
        ## Double Q learning
        q_double = self.eval_net(batch_next_state).detach()
        eval_act = q_double.max(1)[1]
        q_target = batch_reward + self.opt.GAMMA * q_next[:,eval_act].diag().view(self.opt.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        

        #increase epsilon
        # if self.learn_step_counter % 2100 == 0:
        #     self.EPISILON = self.EPISILON + self.EPISILON_increment if self.EPISILON < self.EPISILON_MAX else self.EPISILON_MAX
        #     print('EPSILON = ' + str(self.EPISILON))
        if self.learn_step_counter % 8000 == 0:
            q = q_eval[:10,:]
            print('q_eval = ' + str(q))
        self.optimizer.zero_grad()
        loss.backward()
        # import pdb; pdb.set_trace()
        self.optimizer.step()

    def adjust_learning_rate(self, lr, epoch, loss_decay):
        lr_new = lr
        if epoch % loss_decay == 0:
            lr_new = lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new
        if lr_new != lr:
            print('\033[1;31mlearning rate has updated!\033[0m' + '     learning rate = ' + str(lr_new))
        
        return lr_new




