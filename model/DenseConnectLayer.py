import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseConnectLayerStandard(nn.Module):
    def __init__(self, nChannels, opt):
        super(DenseConnectLayerStandard, self).__init__()
        self.opt = opt
        layers = []

        layers.append(nn.BatchNorm2d(nChannels))
        layers.append(nn.ReLU(inplace=True))
        
        if opt['bottleneck']:
            layers.append(nn.Conv2d(nChannels, 4 * opt['growthRate'], kernel_size=1, stride=1, padding=0, bias=False))
            nChannels = 4 * opt['growthRate']
            if opt['dropRate'] > 0:
                layers.append(nn.Dropout(opt['dropRate']))
            layers.append(nn.BatchNorm2d(nChannels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(nChannels, opt['growthRate'], kernel_size=3, stride=1, padding=1, bias=False))
        if opt['dropRate'] > 0:
            layers.append(nn.Dropout(opt['dropRate']))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)
