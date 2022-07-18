'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.transform_layers import NormalizeLayer
from torch.nn.utils import spectral_norm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Shortcut(nn.Module):
    def __init__(self, args, stride, in_planes, expansion, planes):
        super(Shortcut, self).__init__()
        self.identity = True
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != expansion*planes:
            self.identity = False
            self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.BatchNorm2d(expansion*planes)

    def forward(self, x):
        if self.identity:
            out = self.shortcut(x)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, args, in_planes, planes, stride=1, pooling=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Shortcut(args, stride, in_planes, self.expansion, planes)

        self.pooling = pooling

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        if self.pooling:
            out = F.avg_pool2d(out, 4)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        if 'cifar10' in args.dataset and 'cifar100' not in args.dataset:
            init_dim, last_dim = 64, 512 # last_dim = init_dim * 8
        elif 'cifar100' in args.dataset or 'timgnet' in args.dataset:
            init_dim, last_dim = 128, 1024
        else:
            raise NotImplementedError()
        super(ResNet, self).__init__()

        self.in_planes = init_dim
        self.last_dim = last_dim

        self.conv1 = conv3x3(3, init_dim)
        self.bn1 = nn.BatchNorm2d(init_dim)

        self.gate = torch.sigmoid

        self.layer1 = self._make_layer(args, block, init_dim, num_blocks[0], stride=1, pooling=False)
        self.layer2 = self._make_layer(args, block, init_dim * 2, num_blocks[1], stride=2, pooling=False) # LAYER2.0.SHORTCUT
        self.layer3 = self._make_layer(args, block, init_dim * 4, num_blocks[2], stride=2, pooling=False) # LAYER3.0.SHORTCUT
        self.layer4 = self._make_layer(args, block, last_dim, num_blocks[3], stride=2, pooling=True) # LAYER4.0.SHORTCUT

        self.head = nn.Linear(last_dim, num_classes)

    def _make_layer(self, args, block, planes, num_blocks, stride, pooling=False):
        strides = [stride] + [1]*(num_blocks-1)
        pooling_ = False
        layers = nn.ModuleList()
        for i, stride in enumerate(strides):
            if i == len(strides) - 1:
                pooling_ = pooling
            layers.append(block(args, self.in_planes, planes, stride, pooling_))
            self.in_planes = planes * block.expansion
        return layers

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for op in self.layer1:
            x = op(x)

        for op in self.layer2:
            x = op(x)

        for op in self.layer3:
            x = op(x)

        for op in self.layer4:
            x = op(x) # the output x is (100, 512, 1, 1)

        out = x.view(x.size(0), -1)
        return out

    def forward_classifier(self, x):
        out = self.head(x)
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_classifier(out)
        return out


def ResNet18(args, num_classes):
    return ResNet(args, BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
