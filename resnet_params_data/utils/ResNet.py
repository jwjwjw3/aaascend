'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_planes, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        blks = []
        planes_factor = 1
        for i in range(len(num_blocks)):
            planes_factor = planes_factor * 2
            if i == 0:
                blks.extend(self._make_layer(block, in_planes*planes_factor, num_blocks[i], stride=1))
            else:
                blks.extend(self._make_layer(block, in_planes*planes_factor, num_blocks[i], stride=2))
        self.blks = nn.Sequential(*blks)
        self.linear = nn.Linear(in_planes*planes_factor*block.expansion*(4**(4-len(num_blocks))), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blks(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# -------------------------------------------------------
def ResNet10_22_4():
    return ResNet(BasicBlock, [2, 2], 4)

def ResNet10_22_6():
    return ResNet(BasicBlock, [2, 2], 6)

def ResNet10_22_8():
    return ResNet(BasicBlock, [2, 2], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet14_222_4():
    return ResNet(BasicBlock, [2, 2, 2], 4)

def ResNet14_222_6():
    return ResNet(BasicBlock, [2, 2, 2], 6)

def ResNet14_222_8():
    return ResNet(BasicBlock, [2, 2, 2], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet10_1111_4():
    return ResNet(BasicBlock, [1, 1, 1, 1], 4)

def ResNet10_1111_6():
    return ResNet(BasicBlock, [1, 1, 1, 1], 6)

def ResNet10_1111_8():
    return ResNet(BasicBlock, [1, 1, 1, 1], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet12_2111_4():
    return ResNet(BasicBlock, [2, 1, 1, 1], 4)

def ResNet12_2111_6():
    return ResNet(BasicBlock, [2, 1, 1, 1], 6)

def ResNet12_2111_8():
    return ResNet(BasicBlock, [2, 1, 1, 1], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet14_2211_4():
    return ResNet(BasicBlock, [2, 2, 1, 1], 4)

def ResNet14_2211_6():
    return ResNet(BasicBlock, [2, 2, 1, 1], 6)

def ResNet14_2211_8():
    return ResNet(BasicBlock, [2, 2, 1, 1], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet16_2221_4():
    return ResNet(BasicBlock, [2, 2, 2, 1], 4)

def ResNet16_2221_6():
    return ResNet(BasicBlock, [2, 2, 2, 1], 6)

def ResNet16_2221_8():
    return ResNet(BasicBlock, [2, 2, 2, 1], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet18_2222_4():
    return ResNet(BasicBlock, [2, 2, 2, 2], 4)

def ResNet18_2222_6():
    return ResNet(BasicBlock, [2, 2, 2, 2], 6)

def ResNet18_2222_8():
    return ResNet(BasicBlock, [2, 2, 2, 2], 8)
# -------------------------------------------------------

# -------------------------------------------------------
def ResNet34_3463_4():
    return ResNet(BasicBlock, [3, 4, 6, 3], 4)

def ResNet34_3463_6():
    return ResNet(BasicBlock, [3, 4, 6, 3], 6)

def ResNet34_3463_8():
    return ResNet(BasicBlock, [3, 4, 6, 3], 8)
# -------------------------------------------------------

# def ResNet50_8():
#     return ResNet(Bottleneck, [3, 4, 6, 3])

# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()