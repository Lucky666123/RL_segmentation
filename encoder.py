# from resnet import ResNet
# from resnet import Bottleneck
from resnet18 import resnet18
from resnet18 import BasicBlock
# from resnet18 import _resnet
import torch
import torch.nn as nn

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()

        print 'Building Encoder'
        # self.resnet = ResNet(BasicBlock, layers=[2, 2, 2, 2])
        self.resnet = resnet18(pretrained=False)
        # self.resnet = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained = True, progress = True)
        # self.resnet = ResNet(Bottleneck, layers=[3, 4, 6, 3], strides=[1, 2, 1, 1], \
        #     dilations=[1, 1, 2, 4])


    def forward(self, x):
        layer4 = self.resnet(x)
        # import pdb; pdb.set_trace()

        return layer4
