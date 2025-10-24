import torch
import torch.nn as nn
from .cbam_module import CBAM

class CBAMClassifier(nn.Module):
    def __init__(self):
        super(CBAMClassifier, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.cbam3 = CBAM(128)

        self.Conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.cbam4 = CBAM(256)

        self.Conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.cbam5 = CBAM(512)

        self.Linear1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.cbam3(x)
        x = self.Conv4(x)
        x = self.cbam4(x)
        x = self.dropout(x)
        x = self.Conv5(x)
        x = self.cbam5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x

