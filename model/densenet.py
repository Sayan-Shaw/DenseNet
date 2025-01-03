import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0, bottleneck=False):
        super(DenseLayer, self).__init__()
        inter_channels = growth_rate * 4 if bottleneck else growth_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.avg_pool(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0, bottleneck=False):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate, bottleneck)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes=10, growth_rate=12, depth=100, reduction=0.5, drop_rate=0, bottleneck=True):
        super(DenseNet, self).__init__()
        num_dense_blocks = (depth - 4) // (6 if bottleneck else 3)
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.blocks = nn.ModuleList()
        for i in range(3):
            block = DenseBlock(num_dense_blocks, num_channels, growth_rate, drop_rate, bottleneck)
            self.blocks.append(block)
            num_channels += num_dense_blocks * growth_rate
            if i != 2:
                out_channels = int(num_channels * reduction)
                self.blocks.append(TransitionLayer(num_channels, out_channels))
                num_channels = out_channels
        
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            out = block(out)
        out = self.relu(self.bn2(out))
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def create_densenet(opt):
    model = DenseNet(
        num_classes=opt['num_classes'],
        growth_rate=opt['growth_rate'],
        depth=opt['depth'],
        reduction=opt['reduction'],
        drop_rate=opt['drop_rate'],
        bottleneck=opt['bottleneck']
    )
    return model
