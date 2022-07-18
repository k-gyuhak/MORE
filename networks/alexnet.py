'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # 784 -> 800 -> 800 -> last i.e. 4 layers
    def __init__(self, args, num_classes):
        super(Net, self).__init__()
        self.last_dim = 800

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2)

        self.fc1 = nn.Linear(1024, 800)
        self.fc = nn.Linear(800, num_classes)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.gate = torch.sigmoid

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)

    def forward_features(self, x):
        out = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        out = self.maxpool(self.drop1(self.relu(self.conv2(out))))
        out = self.maxpool(self.drop2(self.relu(self.conv3(out))))
        out = out.view(out.size(0), -1)
        return out

    def forward_classifier(self, x):
        out = self.drop2(self.relu(self.fc1(x)))
        out = self.fc(out)
        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_classifier(out)
        return out
