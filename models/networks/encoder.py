"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch as th
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConesEncoder(th.nn.Module):
    "multi scale encoder inspired by fpn fully resolution"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(ConesEncoder, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(ndf, affine=True)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down & smooth
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        p2 = self.smooth3(p2)
        return p2

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
