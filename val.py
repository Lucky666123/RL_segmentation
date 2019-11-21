# -*- coding: utf-8 -*-
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
import torch.nn.functional as Fr


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
from NextP_Net import DQN
from scipy.fftpack import fft,ifft,fftshift

# import pdb; pdb.set_trace()

# edge distance reward
def reward_contour(next_p, target, reward_ratio):
    next_p = next_p.cpu().numpy()
    point = geom.Point(next_p)  
    target = target.cpu().numpy().squeeze()
    contours_line = geom.LineString(target)
    distance = point.distance(contours_line)

    if distance <= 10:
        reward = reward_ratio * (10 - distance) 
    else:
        reward = 0
       
    distance = round(distance, 4)

    return distance, reward

# difference IoU reward
def reward_iou(target, points):
    target = target.cpu().numpy().squeeze()
    points = np.array(points)
    points[:,[0,1]] = points[:,[1,0]]

    prediction = np.zeros([opt.img_size[0], opt.img_size[1]], dtype = np.uint8)
    cv2.fillConvexPoly(prediction, points, 1)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    intersection = np.sum(intersection)
    union = np.sum(union)
    iou_score = intersection / union

    points_pre = points[:-5, :]
    prediction_pre = np.zeros([opt.img_size[0], opt.img_size[1]], dtype = np.uint8)
    cv2.fillConvexPoly(prediction_pre, points_pre, 1)
    intersection = np.logical_and(target, prediction_pre)
    union = np.logical_or(target, prediction)
    intersection = np.sum(intersection)
    union = np.sum(union)
    iou_score_pre = intersection / union

    if iou_score - iou_score_pre > 0:
        reward = 1
    elif iou_score - iou_score_pre == 0:
        reward = 0
    else:
        reward = -1

    return iou_score, reward


# points clustering reward
def reward_cluster(points_cluster):
    points = np.array(points_cluster)
    points_x = points[:,0]
    points_y = points[:,1]
    x_std = np.std(points_x)
    y_std = np.std(points_y)
    std = np.max((x_std, y_std), 0)
    if std <= 10:
        reward = -0.5
    else:
        reward = 0
    
    return std, reward


parser = argparse.ArgumentParser()
# hyper-parameters in DQN
parser.add_argument('--BATCH_SIZE', type = int, default = 256, help = 'batch size when updating dqn')
parser.add_argument('--LR', type = float, default = 1e-4, help='Adam: learning rate')
parser.add_argument('--GAMMA', type = float, default = 0.9, help='GAMMA')
parser.add_argument('--MEMORY_CAPACITY', type = int, default = 21000, help = 'capacity of replay buffer')
parser.add_argument('--Q_NETWORK_ITERATION', type = int, default = 2000, help = 'interval of assigning the target network parameters to evaluate network')
parser.add_argument('--NUM_ACTIONS', type = int, default = 8, help = 'number of actions')
parser.add_argument('--NUM_STATES', type = int, default = 51*51*5, help = 'number of a state')

# parameters in main function
parser.add_argument('--train_ratio', type = float, default = 0.7, help='ratio of trainging set and testing set')
parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs of training')
parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size')
parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--grid_size', type = list, default = [51, 51], help = 'grid size')
parser.add_argument('--img_size', type = list, default = [368, 368], help = 'image size')
parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model saving checkpoints')
parser.add_argument('--ini_total_steps', type = int, default = 300, help = 'total steps in one episode')
parser.add_argument('--step', type = int, default = 5, help = 'n skip neighborhoods')
parser.add_argument('--radius', type = int, default = 25, help = 'boxes radius')
parser.add_argument('--pool_size', type = int, default = 51, help = 'ROI pooling size')
parser.add_argument('--cluster_points', type = int, default = 20, help = 'cluster points for computing reward3')
parser.add_argument('--iou_points', type = int, default = 10, help = 'iou points for computing reward2')
parser.add_argument('--terminal_distance', type = int, default = 200, help = 'terminal distance')
parser.add_argument('--loss_decay', type = int, default = 100, help = 'loss decay')
parser.add_argument('--edge_reward_ratio', type = float, default = 0.05, help='ratio of edge distance reward')
parser.add_argument('--layers', type = int, default = 5, help = 'layers used')


opt = parser.parse_args()


def main():

    cudnn.benchmack = True
    

    path = os.path.join(os.path.dirname(__file__),"LV_small_RoI")
    jpg_list_train, jpg_list_test, png_list_train, png_list_test = PathList(path, train_ratio = opt.train_ratio)

    TrainDataset = MakeDataset(path, jpg_list_train, png_list_train) 
    TestDataset = MakeDataset(path, jpg_list_test, png_list_test) 

    print('The number of training data: ' + str(len(jpg_list_train)))
    print('The number of testing data: ' + str(len(jpg_list_test)))

    TrainLoader = DataLoader(TrainDataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)
    TestLoader = DataLoader(TestDataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)

    FirstP = torch.load('FirstPNet_LV_small_RoI.pth').cuda()
    dqn = DQN(opt).cuda()
    path = "NextPNet_LV_small_RoI.pth"
    print("Reloading resnet from: ", path)
    dqn.load_state_dict(torch.load(path))
    
    print("Collecting Experience....")


    test_reward_list = []
    test_reward_post_list = []
    test_distance_list = []
    test_iou_list = []
    test_cluster_list = []
    test_total_list = []
    test_precision_list = []
    test_recall_list = []
    test_precision_post_list = []
    test_recall_post_list = []
    test_APD_list = []
    test_APD_post_list = []
    test_iouvalue_list = []


    all_average_F_testing = []
    all_average_F_testing_post = []
    all_average_precision_testing = []
    all_average_recall_testing = []
    all_average_precision_testing_post = []
    all_average_recall_testing_post = []
    all_average_APD_testing = []
    all_average_APD_testing_post = []
    all_average_iou2_testing = []

    marker_size = 20
    line_width = 2

    
    lr = opt.LR
    prev_time = time.time()
    batch_random = 0
    
    

    total_steps = 300
    terminal_distance = 200

    # start testing
    print('\033[1;31mStart Testing!\033[0m')
    for batch_idx, (inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_pixel) in enumerate(TestLoader):
        step = opt.step     
        inputs, inputs_gray, inputs_sobel, inputs_laplacian, targets, targets_pixel = inputs.cuda(), inputs_gray.cuda(), inputs_sobel.cuda(), inputs_laplacian.cuda(), targets.cuda(), targets_pixel.cuda()
        

        _, conv_pixel, _, _, ori_p_pre = FirstP(inputs)
        
        conv_pixel = conv_pixel.cpu().detach().numpy().squeeze()

        input_5 = cv2.resize(conv_pixel, (opt.pool_size, opt.pool_size))
        min_max_scaler = MinMaxScaler()
        input_5 = min_max_scaler.fit_transform(input_5)
        input_5 = input_5.reshape([opt.pool_size, opt.pool_size, 1])
        input_5 = transforms.ToTensor()(input_5).cuda()
        input_5 = transforms.Normalize((0.5,), (0.5,))(input_5)
        input_5 = input_5.float()

        layer5 = input_5.unsqueeze(0)
        
        pixel = cv2.pyrUp(conv_pixel)
        pixel = cv2.pyrUp(pixel)
        pixel = cv2.pyrUp(pixel)
        min_max_scaler = MinMaxScaler()
        pixel = min_max_scaler.fit_transform(pixel)
        pixel = pixel.reshape([opt.img_size[0], opt.img_size[1], 1])
        input_pixel = transforms.ToTensor()(pixel).cuda()
        input_pixel = transforms.Normalize((0.5,), (0.5,))(input_pixel)
        input_pixel = input_pixel.float()
        layer3 = input_pixel.unsqueeze(0)

        layer4 = np.zeros((368, 368), dtype = np.float32)
        layer4 = layer4 - 1
        layer4 = torch.from_numpy(layer4)
        layer4 = layer4.cuda()
        layer4 = layer4.unsqueeze(0)
        layer4 = layer4.unsqueeze(0)

        input_cat = torch.cat((inputs_gray, inputs_sobel, layer3, layer4), dim = 1)
        # input_cat = torch.cat((inputs_gray, inputs_sobel, layer3), dim = 1)

        contour = targets_pixel.cpu().numpy().squeeze()
        contour = contour.tolist()
        length = len(contour)

        ep_APD = 0
        reward_contour_ratio = 0.05

        _, _, _, _, ori_p = FirstP(inputs)
        distance, _ = reward_contour(next_p = ori_p, target = targets_pixel, reward_ratio = reward_contour_ratio)              
        ep_APD += distance
        first_p = ori_p.cpu().numpy()
        points = []  # compute the reward for iou
        points.append(first_p)

        
        
        ep_reward = 0
        ep_con_reward = 0
        ep_iou_reward = 0
        ep_cluster_reward = 0
        ep_distance = 0
        
        for i in range(total_steps):
            state = GridFeatures(ori_p, opt.radius, opt.pool_size, opt.img_size, input_cat)
            state = torch.cat((state, layer5), dim = 1)

            action = dqn.choose_action(state)


            if action == 0:
                    y = ori_p[0] - step
                    x = ori_p[1]
                    next_p = torch.stack([y,x], dim = 0)
            if action == 1:
                    y = ori_p[0] - step
                    x = ori_p[1] + step
                    next_p = torch.stack([y,x], dim = 0)
            if action == 2:
                    y = ori_p[0] 
                    x = ori_p[1] + step
                    next_p = torch.stack([y,x], dim = 0)
            if action == 3:
                    y = ori_p[0] + step
                    x = ori_p[1] + step
                    next_p = torch.stack([y,x], dim = 0)
            if action == 4:
                    y = ori_p[0] + step
                    x = ori_p[1] 
                    next_p = torch.stack([y,x], dim = 0)
            if action == 5:
                    y = ori_p[0] + step
                    x = ori_p[1] - step 
                    next_p = torch.stack([y,x], dim = 0)
            if action == 6:
                    y = ori_p[0] 
                    x = ori_p[1] - step
                    next_p = torch.stack([y,x], dim = 0)
            if action == 7:
                    y = ori_p[0] - step 
                    x = ori_p[1] - step
                    next_p = torch.stack([y,x], dim = 0)

            height = opt.img_size[0] - 1
            width = opt.img_size[1] - 1
            window = np.array([0, 0, height, width]).astype(np.int)
            next_p_new = torch.stack( \
                    [next_p[0].clamp(int(window[0]), int(window[2])),
                    next_p[1].clamp(int(window[1]), int(window[3]))], 0).int()
            point_cordinate = next_p_new.cpu().numpy()
            points.append(point_cordinate)

            next_layer4 = np.zeros((368, 368), dtype = np.float32)
            next_layer4 = next_layer4 - 1              
            points_array = np.array(points)
            rows = points_array[-50:,0]
            cols = points_array[-50:,1]
            next_layer4[rows, cols] = 1
            next_layer4 = torch.from_numpy(next_layer4)
            next_layer4 = next_layer4.cuda()
            next_layer4 = next_layer4.unsqueeze(0)
            next_layer4 = next_layer4.unsqueeze(0)

            input_cat = torch.cat((inputs_gray, inputs_sobel, layer3, next_layer4), dim = 1)
            # input_cat = torch.cat((inputs_gray, inputs_sobel, next_layer4), dim = 1)

            distance, reward1 = reward_contour(next_p = next_p_new, target = targets_pixel, reward_ratio = reward_contour_ratio)              
            ep_APD += distance
            

            iou_score = 0
            reward2 = 0
            if i >= opt.iou_points - 1:
                iou_score, reward2 = reward_iou(targets, points)

            reward3 = 0
            if i >= opt.cluster_points - 1:
                points_cluster = points[-opt.cluster_points:]
                std, reward3 = reward_cluster(points_cluster)

            reward = reward1 + reward2 + reward3

            ep_reward += reward
            ep_con_reward += reward1
            ep_iou_reward += reward2
            ep_cluster_reward += reward3

            last_p = ori_p.cpu().numpy()
            f_l_length = last_p - first_p
            f_l_length = math.hypot(f_l_length[0],f_l_length[1])
            if i >= 100 and f_l_length <= 40:
                step = 3
            if i >= 100 and f_l_length <= 20:
                break
            
            ori_p = next_p_new
            
        test_distance_list.append(ep_con_reward)
        test_iou_list.append(ep_iou_reward)
        test_cluster_list.append(ep_cluster_reward)
        test_total_list.append(ep_reward)
        
        if (batch_idx+1) % 1 == 0:
            # import pdb; pdb.set_trace()
            points.append(first_p)

            points = np.array(points)
            img = inputs.cpu().squeeze()
            img = np.transpose(img,(1,2,0))
            img = 255*(img*0.5+0.5)
            img_numpy = np.array(img, dtype = np.uint8)
            contour = targets_pixel.cpu().numpy().squeeze()
            
            plt.figure(0)
            ax = plt.gca()
            ax.imshow(img_numpy)

            plt.axis('off')
            ax.plot(contour[:,1], contour[:,0], marker = '.', c ='b')
            ax.plot(points[:,1], points[:,0], marker = '.', c ='m')
            ax.plot(points[0,1], points[0,0], markersize = marker_size, marker = '*', c = 'r')

            prediction = np.zeros([opt.img_size[0], opt.img_size[1]], dtype = np.uint8)
            points[:,[0,1]] = points[:,[1,0]]
            cv2.fillConvexPoly(prediction, points, 1)
            
            label = measure.label(prediction, connectivity=1) 
            properties = measure.regionprops(label)
            # properties.area
            area = []
            for i in range(len(properties)):
                area.append(properties[i].area)
            area_max = max(area)
            prediction_remove = morphology.remove_small_objects(prediction, min_size = (area_max - 1), connectivity=1, in_place=False)
            
            targets = targets.cpu().numpy().squeeze()
            intersection = np.logical_and(targets, prediction_remove)
            union = np.logical_or(targets, prediction_remove)
            intersection = np.sum(intersection)
            union = np.sum(union)
            targets_sum = np.sum(targets) / 255
            prediction_remove_sum = np.sum(prediction_remove)
            precision = intersection / prediction_remove_sum
            recall = intersection / targets_sum
            if precision == 0 and recall == 0:
                F_score = 0
            else:
                F_score = 2 * precision * recall / (precision + recall)
            F_score_print = round(F_score, 4)

            IoU = intersection / union


            figure_path = 'LV_small/test_images'
            path = os.path.join(os.path.dirname(__file__),"figure", figure_path)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            plt.savefig(os.path.join(path, str(batch_idx+1)+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close(0)



            contour_remove = measure.find_contours(prediction_remove, 0.5)
            size_contour = []
            for i in range(len(contour_remove)):
                a = contour_remove[i].shape
                size_contour.append(a[0])
            max_index = size_contour.index(max(size_contour))
            contour_remove = contour_remove[max_index]
            con_array = np.array(contour_remove)
            con_array = np.squeeze(contour_remove).astype(np.int)
            for row in range(con_array.shape[0]):
                p = con_array[row,:]
                p = geom.Point(p)  
                contours_line = geom.LineString(contour)
                distance = p.distance(contours_line)
                ep_distance += distance
            xb = con_array[:,0]
            yb = con_array[:,1]

            fhist = fft(xb)
            fhist1 = abs(fftshift(fhist))
            bandw = 15
            fhist[bandw:(fhist1.shape[0]-bandw)] = 0
            xb1 = ifft(fhist)
            xb1 = abs(xb1).astype(np.int)
            
            fhist = fft(yb)
            fhist1 = abs(fftshift(fhist))
            fhist[bandw:(fhist1.shape[0]-bandw)] = 0
            yb1 = ifft(fhist)
            yb1 = abs(yb1).astype(np.int)

            prediction_post = np.zeros([opt.img_size[0], opt.img_size[1]], dtype = np.uint8)
            contour_post = np.stack((yb1,xb1), axis = 0)
            contour_post = np.transpose(contour_post)
            contour_post = contour_post.astype(np.int64)
            cv2.fillConvexPoly(prediction_post, contour_post, 1)
            label = measure.label(prediction_post, connectivity=1) 
            properties = measure.regionprops(label)
            area = []
            for i in range(len(properties)):
                area.append(properties[i].area)
            area_max = max(area)
            prediction_post_remove = morphology.remove_small_objects(prediction_post, min_size = (area_max - 1), connectivity=1, in_place=False)

            intersection = np.logical_and(targets, prediction_post_remove)
            union = np.logical_or(targets, prediction_post_remove)
            intersection = np.sum(intersection)
            union = np.sum(union)

            prediction_post_remove_sum = np.sum(prediction_post_remove)
            precision_post = intersection / prediction_post_remove_sum
            recall_post = intersection / targets_sum
            if precision_post == 0 and recall_post == 0:
                F_score_post = 0
            else:
                F_score_post = 2 * precision_post * recall_post / (precision_post + recall_post)
            F_score_post_print = round(F_score_post, 4)

            plt.figure(1)
            ax = plt.gca()
            plt.axis('off')
            ax.imshow(img_numpy)
            ax.plot(contour[:,1], contour[:,0], marker = ',', c ='b', lw = line_width)
            ax.plot(yb1, xb1, marker = ',', c ='m', lw = line_width)
            ax.plot(points[0,0], points[0,1], markersize = marker_size, marker = '*', c = 'r')


            figure_path = 'LV_small/test_post'
            path = os.path.join(os.path.dirname(__file__),"figure", figure_path)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            plt.savefig(os.path.join(path, str(batch_idx+1)+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close(1)

            APD = ep_APD / (len(points)-1)
            APD_post = ep_distance / con_array.shape[0]


        test_reward_list.append(F_score)
        test_reward_post_list.append(F_score_post)
        test_precision_list.append(precision)
        test_recall_list.append(recall)
        test_precision_post_list.append(precision_post)
        test_recall_post_list.append(recall_post)
        test_APD_list.append(APD)
        test_APD_post_list.append(APD_post)
        test_iouvalue_list.append(IoU)


        print("\r[Image %d][Episode Length %d] [F_measure %.4f]" % \
                (int(batch_idx+1), int(len(points)-2), float(F_score)))

    epoch = 0
    average_F = test_reward_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_F_post = test_reward_post_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_precision = test_precision_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_recall = test_recall_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_precision_post = test_precision_post_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_recall_post = test_recall_post_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_APD = test_APD_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_APD_post = test_APD_post_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]
    average_iou2 = test_iouvalue_list[epoch*len(jpg_list_test):(epoch+1)*len(jpg_list_test)]


    all_average_F_testing.append(np.mean(average_F))
    all_average_F_testing_post.append(np.mean(average_F_post))
    all_average_precision_testing.append(np.mean(average_precision))
    all_average_recall_testing.append(np.mean(average_recall))
    all_average_precision_testing_post.append(np.mean(average_precision_post))
    all_average_recall_testing_post.append(np.mean(average_recall_post))
    all_average_APD_testing.append(np.mean(average_APD))
    all_average_APD_testing_post.append(np.mean(average_APD_post))
    all_average_iou2_testing.append(np.mean(average_iou2))
    

    print('The mean F-measure of every epoch in the testing process is:  ' + str(all_average_F_testing))
    print('The mean post-processing F-measure of every epoch in the testing process is:  ' + str(all_average_F_testing_post))
    print('The mean precision of every epoch in the testing process is:  ' + str(all_average_precision_testing))
    print('The mean post-processing precision of every epoch in the testing process is:  ' + str(all_average_precision_testing_post))
    print('The mean recall of every epoch in the testing process is:  ' + str(all_average_recall_testing))
    print('The mean post-processing recall of every epoch in the testing process is:  ' + str(all_average_recall_testing_post))
    print('The mean APD of every epoch in the testing process is:  ' + str(all_average_APD_testing))
    print('The mean post-processing APD of every epoch in the training process is:  ' + str(all_average_APD_testing_post))


    np.savetxt('LV_small_test_Fmeasure.txt' , test_reward_list)
    np.savetxt('LV_small_test_Fmeasure_post.txt' , test_reward_post_list)
    np.savetxt('LV_small_test_precision.txt' , test_precision_list)
    np.savetxt('LV_small_test_recall.txt' , test_recall_list)
    np.savetxt('LV_small_test_precision_post.txt' , test_precision_post_list)
    np.savetxt('LV_small_test_recall_post.txt' , test_recall_post_list)
    np.savetxt('LV_small_test_APD.txt' , test_APD_list)
    np.savetxt('LV_small_test_APD_post.txt' , test_APD_post_list)
    np.savetxt('LV_small_test_distance.txt' , test_distance_list)
    np.savetxt('LV_small_test_iou.txt' , test_iou_list)
    np.savetxt('LV_small_test_cluster.txt' , test_cluster_list)
    np.savetxt('LV_small_test_total.txt' , test_total_list)


    print('\033[1;31mThe txt files are successfully saved!\033[0m')


main()


