import torch
import torch.nn as nn
from .cbam_module import CBAM

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=self.padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CBAMClassifierCompressed(nn.Module):
    def __init__(self):
        super(CBAMClassifierCompressed, self).__init__()
        self.Conv1 = nn.Sequential(
            DepthwiseSeparableConv(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.cbam3 = CBAM(128)

        self.Conv4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.cbam4 = CBAM(256)

        self.Conv5 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.cbam5 = CBAM(512)
        
        dummy_input = torch.zeros(1, 1, 224, 224)
        x = self.forward_conv(dummy_input)
        flattened_size = x.view(1, -1).shape[1]

        self.Linear1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, 25)

    def forward_conv(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.cbam3(x)
        x = self.Conv4(x)
        x = self.cbam4(x)
        x = self.Conv5(x)
        x = self.cbam5(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x

