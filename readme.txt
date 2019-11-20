This is the code for "Medical Image Segmentation with Deep Reinforcement Learning"

1. Requirements

Python2.7
torch 0.4.0
torchvision 0.2.1
matplotlib 2.2.3
numpy 1.16.4
opencv-python 4.1.0.25
scikit-image 0.14.3
scikit-learn 0.20.4
shapely 1.6.4.post2
cffi
scipy

2. As we use a crop and resize function like that in Fast R-CNN to fix the size of the state, it needs to be built with the right -arch option for Cuda support before training.

GPU           arch
TitanX        sm_52
GTX 960M      sm_50
GTX 1070      sm_61
GTX 1080(Ti)  sm_61

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../../

3. Run train.py to train the DQN agent.
4. Run val.py to test the proposed model.
