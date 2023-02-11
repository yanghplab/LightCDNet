# -*- coding: utf-8 -*-
# File   : LightCDNet.py
# Author : Yuanyuan Chen
# Email  : 1470787912@qq.com
# Date   : 07/12/2022

# This file is model part of A Lightweight Siamese Neural Network for Building Change Detection.
# Distributed under MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from thop import profile

def basic_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
class LightCDNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=False, freeze_bn=False,isdeconv=True):
        super(LightCDNet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.isdeconv=isdeconv
        self.backbone= build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.freeze_bn = freeze_bn
        if  backbone == 'resnet':
            self.lowf_conv=basic_block(256*2,256*1)
            self.highf_conv=basic_block(2048*2,2048*1)
            low_level_inplanes = 256
        elif  backbone == 'xception':
            self.lowf_conv=basic_block(128*2,128*1)
            self.highf_conv=basic_block(2048*2,2048*1)
            low_level_inplanes = 128
        elif  backbone == 'mobilenet':
            self.lowf_conv=basic_block(24*2,24*1)
            self.highf_conv=basic_block(320*2,320*1)
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block1= nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(64),
                                       nn.ReLU(),
                                       )
        if self.isdeconv:
            self.conv= nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        else:
            self.conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       )
        self._init_weight()
    def forward(self, input1,input2):
# -------------------SiameseEncoder------------------------
        #shaing weight
        FH1, FL1 = self.backbone(input1)
        FH2, FL2 = self.backbone(input2)
#----------------------------------------------------------------------

#-------------------MultitemporalFeatureFusion-------------------------
        # low level features fusion
        low_level_feat = torch.cat((FL1, FL2), dim=1)
        low_level_fusion_feat = self.lowf_conv(low_level_feat)
        low_level_fusion_feat = self.conv1(low_level_fusion_feat)
        low_level_fusion_feat = self.bn1(low_level_fusion_feat)
        low_level_fusion_feat = self.relu(low_level_fusion_feat)
        # high level features fusion
        high_level_feat = torch.cat((FH1, FH2), dim=1)
        high_level_fusion_feat = self.highf_conv(high_level_feat)
        high_level_fusion_feat = self.aspp(high_level_fusion_feat)
        high_level_fusion_feat = F.interpolate(high_level_fusion_feat, size=low_level_fusion_feat.size()[2:],
                                               mode='bilinear', align_corners=True)
        # fusion features
        fusion_feat = torch.cat((high_level_fusion_feat, low_level_fusion_feat), dim=1)
#-----------------------------------------------------------------------

#--------------------------------Decoder or upsampling--------------------------------
        fusion_feat = self.last_conv(fusion_feat)
        if self.isdeconv:
            # Deconvolution
            up1 = self.up1(fusion_feat)
            block1 = self.block1(up1)
            up2 = self.up2(block1)
            block2 = self.block2(up2)
            out = self.conv(block2) # classify
        else:
            # upsampling
            x=self.conv(fusion_feat)
            out = F.interpolate(x, size=input1.size()[2:], mode='bilinear', align_corners=True)
#------------------------------------------------------------------------
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    model = LightCDNet(backbone='mobilenet', output_stride=16,isdeconv=True)
    input1 = torch.rand(8, 3, 256, 256)
    input2 = torch.rand(8, 3, 256, 256)
    output = model(input1,input2)
    print(output.shape)
    flops, params = profile(model, inputs=(input1, input2))
    print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
    print("params=", str(params / 1e6) + '{}'.format("M"))



