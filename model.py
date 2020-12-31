import torchsummary
from torch import nn
import torch


# class Inception(nn.Module):
#     def __init__(self, in_channels=1):
#         super(Inception, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=64,
#                       kernel_size=(11, 1), padding=(5, 0)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64,
#                       kernel_size=(1, 11), padding=(0, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1)),
#             nn.LocalResponseNorm(2),
#         )

#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128,
#                       kernel_size=(5, 1), padding=(2, 0)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128,
#                       kernel_size=(1, 5), padding=(0, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1)),
#             nn.LocalResponseNorm(2),
#         )

#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256,
#                       kernel_size=(3, 1), padding=(1, 0)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256,
#                       kernel_size=(1, 3), padding=(0, 1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1)),
#             nn.LocalResponseNorm(2),
#         )

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = out.reshape(out.shape[0], -1)
#         return out

class Resblock(nn.Module):
    def __init__(self, in_channels=16):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(0.5),

        )

    def forward(self, x):
        out = self.layer1(x)
        out = torch.add(out, x)
        out = nn.ReLU()(out)
        return out


class Inception(nn.Module):
    def __init__(self, in_channels=200, out_channels=3):
        super(Inception, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
        )

        self.layer2 = Resblock(out_channels)

        self.layer3 = Resblock(out_channels)
        self.layer4 = Resblock(out_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        return out


class InceptionLSTM(nn.Module):

    def __init__(self, in_channels=200, batch=128, input_size=24, out_channels=3):
        super(InceptionLSTM, self).__init__()
        self.inception = Inception(in_channels, out_channels=out_channels)
        # self.lstm1 = nn.LSTM(input_size, 46)
        # self.lstm2 = nn.LSTM(128, 46)
        # self.linear = nn.Linear(out_channels*24, 46)
        self.last_conv = nn.Conv1d(out_channels, 46, kernel_size=3, padding=1)
        self.batch = batch
        self.out_channels = out_channels
        self.linear_mnist = nn.Linear(9600, 200*47)

    def forward(self, x):
        x = self.linear_mnist(x)
        x = x.reshape(-1, 200, 47)
        out = self.inception(x)
        # out = out.permute(1, 0, 2)
        # out, _ = self.lstm1(out)
        # out, _ = self.lstm2(out)
        # out = out.permute(1, 0, 2).reshape(-1, 64*46)
        out = self.last_conv(out)
        out = nn.functional.max_pool1d(out, kernel_size=out.size()[2])
        out = out.reshape(-1, 46)
        # out = self.linear(out)
        out = nn.Softmax(1)(out)
        return out


# model = InceptionLSTM().cuda()
# torchsummary.summary(model, (200, 47))
