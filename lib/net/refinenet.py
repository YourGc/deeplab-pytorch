import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from lib.net.backbone import build_backbone

def conv3x3(in_planes, out_planes, stride=1):
    #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def maxpool(kernel_size,stride=2,ceil=False):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)//2,(kernel_size-1)//2),ceil_mode=ceil)

def PredConv(in_planes, out_planes, kernel_size=[1,1], stride=1):
    #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def ConvUpscaleBlock(in_channels, out_channels, kernel_size=[3, 3], stride=2,padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.ReLU(inplace=True))

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

class RCUBlock(nn.Module):
    def __init__(self, in_channels=256,out_channels=256, kernel_size=3,stride=1, bias=False, relu=True):
        super(RCUBlock,self).__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.conv1 = nn.Sequential(activation,
            nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=stride,bias=bias,padding=(kernel_size-1)//2))
        self.conv2 = nn.Sequential(activation,
            nn.Conv2d(out_channels, out_channels,kernel_size=kernel_size,stride=stride,bias=bias,padding=(kernel_size-1)//2))
        #self.shortcut = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out += shortcut
        
        return out

class ChainedResidualPooling(nn.Module):
    def __init__(self, in_channels=256,out_channels=256, pool_kernel_size = 5, conv_kernel_size=3,stride=1, bias=False, relu=True):
        super(ChainedResidualPooling,self).__init__()
        if relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.PReLU()

        self.max1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=(pool_kernel_size-1)//2)
        self.conv1 =  nn.Conv2d(in_channels, out_channels,kernel_size=conv_kernel_size,stride=stride,bias=bias,padding=(conv_kernel_size-1)//2)

        self.max2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=(pool_kernel_size-1)//2)
        self.conv2 =  nn.Conv2d(in_channels, out_channels,kernel_size=conv_kernel_size,stride=stride,bias=bias,padding=(conv_kernel_size-1)//2)


    def forward(self, x):
        net_relu =  self.activation(x)
        max1     = self.max1(net_relu)
        conv1    = self.conv1(max1)
        sun1     = conv1 + net_relu

        max2     = self.max1(net_relu)
        conv2    = self.conv1(max2)
        sun2     = conv2 + sun1

        out = sun2
        return out

class MultiResolutionFusion(nn.Module):
    def __init__(self, out_channels=256, Low_input=256, High_input=0,kernel_size=3):
        super(MultiResolutionFusion,self).__init__()
        if High_input == 0:
            self.conv_low = nn.Conv2d(Low_input, out_channels,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        else:
            self.conv_low = nn.Conv2d(Low_input, out_channels,kernel_size=kernel_size,padding=(kernel_size-1)//2)
            self.conv_high =  ConvUpscaleBlock(High_input, out_channels)

    def forward(self, Low, High = None):
        low = self.conv_low(Low)

        if not hasattr(self, 'shortcut'):
            return low
        high = self.conv_high(High)
        high = crop_like(high, low)
        out = high + low
        return out

class BoundaryRefinementBlock(nn.Module):
    def __init__(self, in_channels,out_channels,
                 kernel_size=1, padding=0, bias=False, relu=True):
        super(BoundaryRefinementBlock,self).__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias),
                nn.BatchNorm2d(out_channels), activation)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=padding,bias=bias)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        #print(out.size(),shortcut.size())
        out = out + shortcut
        return out

class RefineBlock(nn.Module):
    def __init__(self,out_channels=256, Low_input=256, High_input=0):
        super(RefineBlock,self).__init__()
        if High_input == 0:
             self.RCU_low =  nn.Sequential(RCUBlock(Low_input,out_channels), RCUBlock(out_channels, out_channels))
        else:
            self.RCU_low =  nn.Sequential(RCUBlock(Low_input,out_channels), RCUBlock(out_channels, out_channels))
            self.RCU_high =  nn.Sequential(RCUBlock(High_input,out_channels), RCUBlock(out_channels, out_channels))

        self.MRF = MultiResolutionFusion(out_channels, Low_input, High_input)
        self.CRP = ChainedResidualPooling(out_channels,out_channels)
        self.RCU_out = RCUBlock(out_channels,out_channels)


    def forward(self, Low, High):
        rcu_low  = self.RCU_low(Low)
        rcu_high = self.RCU_high(High) if hasattr(self, 'shortcut') else High

        out =  self.MRF(rcu_low,rcu_high)
        out =  self.CRP(out)
        out =  self.RCU_out(out)
        return out

class refinenet(nn.Module):
    def __init__(self, cfg, pretrained_backbone=False):
        super(refinenet, self).__init__()
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=pretrained_backbone, os=cfg.MODEL_OUTPUT_STRIDE)
        self.in_planes = [64, 256, 512, 1024, 2048]
        self.feature = 256
        #self.number_classes = cfg.MODEL_NUM_CLASSES

        self.adj4 = nn.Conv2d(self.in_planes[4], 512,kernel_size=1)
        self.adj3 = nn.Conv2d(self.in_planes[3], 256,kernel_size=1)
        self.adj2 = nn.Conv2d(self.in_planes[2], 256,kernel_size=1)
        self.adj1 = nn.Conv2d(self.in_planes[1], 256,kernel_size=1)

        self.RefineBlock4 = RefineBlock(out_channels=512, Low_input=512, High_input=0)
        self.RefineBlock3 = RefineBlock(out_channels=256, Low_input=256, High_input=512)
        self.RefineBlock2 = RefineBlock(out_channels=256, Low_input=256, High_input=256)
        self.RefineBlock1 = RefineBlock(out_channels=256, Low_input=256, High_input=256)
        self.UpBr1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                   #ConvUpscaleBlock(self.feature,self.feature),
                                   nn.Conv2d(self.feature, self.feature,kernel_size=1))
                                    #RCUBlock(self.feature,self.feature))
        self.UpBr2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(self.feature, self.feature,kernel_size=1))
                                    #RCUBlock(self.feature,self.feature))
        self.output = nn.Sequential(RCUBlock(self.feature,self.feature),
                                    RCUBlock(self.feature,self.feature),
                                    nn.Conv2d(self.feature,cfg.MODEL_NUM_CLASSES,kernel_size=1,stride=1,padding=0,bias=True))
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.backbone_layers = self.backbone.get_layers()

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          elif isinstance(m, SynchronizedBatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = x[:,:3,:,:]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        #print(x.size(),len(layers),layers[0].size(),layers[1].size(),layers[2].size(),layers[3].size())
        RCU1 = self.adj1(layers[0])
        RCU2 = self.adj2(layers[1])
        RCU3 = self.adj3(layers[2])
        RCU4 = self.adj4(layers[3])

        MR4 = self.RefineBlock4(RCU4,None)
        MR3 = self.RefineBlock3(RCU3,MR4)
        MR2 = self.RefineBlock2(RCU2,MR3)
        MR1 = self.RefineBlock1(RCU1,MR2)
        out = self.UpBr1(MR1)
        out = self.UpBr2(out)
        result = self.output(out)

        return result

