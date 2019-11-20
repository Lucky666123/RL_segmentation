import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import cv2



class FirstPixel(nn.Module):
    def __init__(self, feats_dim = [46,46], feats_channels = 128):
        super(FirstPixel, self).__init__()
        self.pixel_conv = nn.Conv2d(feats_channels, 16, kernel_size = 3, padding = 1)
        self.pixel_conv2 = nn.Conv2d(16, 1, kernel_size = 1, padding = 0)
        # self.pixel_fc = nn.Linear(50*38*16, 50*38)
        # self.avgpool2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, feats, beam_size = 1):
        batch_size = feats.size(0)
        conv_pixel = self.pixel_conv(feats)
        conv_pixel = F.relu(conv_pixel)
        conv_pixel = self.pixel_conv2(conv_pixel)
        pixel_logits = conv_pixel.view(batch_size, -1)
        
        # import pdb; pdb.set_trace()
        logprobs = F.log_softmax(pixel_logits, -1)

        logprob, pred_first = torch.topk(logprobs, beam_size, dim = 1)

        return conv_pixel, pixel_logits, pred_first