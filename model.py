import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels=1):
        super(Inception, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(11, 1), padding=(5, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 11), padding=(0, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1)),
            nn.LocalResponseNorm(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1)),
            nn.LocalResponseNorm(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2,padding=(0, 1)),
            nn.LocalResponseNorm(2),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.shape[0], -1)
        return out


class InceptionLSTM(nn.Module):

    def __init__(self, in_channels=1, batch=128, input_size=(128,64)):
        super(InceptionLSTM, self).__init__()
        tate,yoko=input_size
        self.inception = Inception(in_channels)
        self.lstm1 = nn.LSTM(256*(tate//(2**3))*(yoko//(2**3)), 128)
        self.lstm2 = nn.LSTM(128, 26)
        self.batch = batch

    def forward(self, x):
        out = self.inception(x)
        out = out.reshape(out.shape[0] // self.batch, self.batch, out.shape[1])
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out = nn.Softmax(2)(out)
        return out


import torchsummary

model = InceptionLSTM(input_size=(64,32))
# torchsummary.summary(model, (1, 128, 64))
a = torch.ones(256, 1, 64, 32)
print(model(a).shape)
