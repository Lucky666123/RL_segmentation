# RL_segmentation

This is the code for "Medical Image Segmentation with Deep Reinforcement Learning"

The proposed model consists of two neural networks. The first is FirstP-Net, whose goal is to find the first edge point and generate a probability map of the edge points positions. The second is NextP-Net, which locates the next point based on the previous edge point and image information. This model segments the image by find- ing the edge points step by step and ultimately obtaining a closed and accurate segmentation result.

![Instance Segmentation Sample](images/Fig7.png)
The ground truth (GT) boundary is plotted in blue and the magenta dots are the points found by NextP-Net. The red pentagram represents the first edge point found by FirstP-Net.

## Requirements
* Python2.7
* torch 0.4.0
* torchvision 0.2.1
* matplotlib 2.2.3
* numpy 1.16.4
* opencv-python 4.1.0.25
* scikit-image 0.14.3
* scikit-learn 0.20.4
* shapely 1.6.4.post2
* cffi
* scipy


## Installation
1. Clone this repository.

        git clone https://github.com/Mayy1994/RL_segmentation.git

2. As we use a crop and resize function like that in Fast R-CNN (https://github.com/longcw/RoIAlign.pytorch) to fix the size of the state, it needs to be built with the right -arch option for Cuda support before training.

    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | GTX 960M | sm_50 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

        cd nms/src/cuda/
        nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../

        cd roialign/roi_align/src/cuda/
        nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../../
        
3. Run train.py to train the DQN agent on 15 subjects from the ACDC dataset, or you can run val.py to test the proposed model on this dataset.

## Training curves and results
![Instance Segmentation Sample](images/Fig9.png)
The changes in three separate reward values, total reward value, F-measure accuracy and APD accuracy according to the learning iterations during the training process on ACDC dataset.

![Instance Segmentation Sample](images/Table.png)
The segmentation results of these baselines on different testing datasets.
