'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.normalize import Normalize

from torch.autograd import Variable
__all__ = ["cifar_resnet18_feat",  "cifar_resnet34_feat", "cifar_resnet50_feat", "cifar_resnet101_feat", "cifar_resnet152_feat"]

class CIFAR_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CIFAR_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CIFAR_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(CIFAR_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, D, H, W = x.shape  # batch, dim, height, width
        x = x.reshape(N, D, -1) # batch, dim, length
        x = x.permute(0,2,1) # N, L, D

        N, L, D = x.shape

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        #ids_keep = ids_shuffle[:, :len_keep]
        #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = 1-mask
        #print(mask_ratio, torch.sum(mask)/(N*L))

        #print(mask[0], mask.shape)
        mask = mask.unsqueeze(-1).repeat(1,1,D)
        #print(mask[0,:,0], mask.shape)

        x_masked = x * mask

        x_masked = x_masked.permute(0,2,1) # N, D, L
        x_masked = x_masked.reshape(N, D, H, W)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.5):
        #out = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        #print('layer1 ', out.size())
        out = self.layer2(out)
        #print('layer2 ', out.size())
        g = self.layer3(out)

        g_masked,_,_ = self.random_masking(g, mask_ratio)

        f = self.layer4(g)

        f_masked = self.layer4(g_masked)

        return g, g_masked, f, f_masked

def cifar_resnet18_feat(num_classes=128, **kwargs):
    return ResNet(CIFAR_BasicBlock, [2,2,2,2], num_classes)

def cifar_resnet34_feat(num_classes=128, **kwargs):
    return ResNet(CIFAR_BasicBlock, [3,4,6,3], num_classes)

def cifar_resnet50_feat(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,4,6,3], num_classes)

def cifar_resnet101_feat(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,4,23,3], num_classes)

def cifar_resnet152_feat(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,8,36,3], num_classes)


def test():
    net = cifar_resnet18_feat()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()