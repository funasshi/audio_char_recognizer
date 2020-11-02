import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(11,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 11)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=2),
            nn.LocalResponseNorm(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=2),
            nn.LocalResponseNorm(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=2),
            nn.LocalResponseNorm(2),
        )

    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        return out


model = Inception(1)

import torchsummary

torchsummary.summary(model, (1,256,512))


