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
            nn.Dropout(0.25),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
        )
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.add(out, x)
        out = nn.ReLU()(out)
        # out = self.dropout2(out)
        return out


class Resblock2d(nn.Module):
    def __init__(self, in_channels=16):
        super(Resblock2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.add(out, x)
        out = nn.ReLU()(out)
        # out = self.dropout2(out)
        return out


class Inception(nn.Module):
    def __init__(self, in_channels=200, out_channels=3):
        super(Inception, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
        )

        self.layer2 = Resblock(out_channels)
        self.layer3 = Resblock(out_channels)
        self.layer4 = Resblock(out_channels)
        self.layer5 = Resblock(out_channels)
        self.layer6 = Resblock(out_channels)
        self.layer7 = Resblock(out_channels)
        self.layer8 = Resblock(out_channels)
        self.layer9 = Resblock(out_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


class Inception2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Inception2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.layer2 = Resblock2d(out_channels)
        self.layer3 = Resblock2d(out_channels)

    def forward(self, x):
        x = x.reshape(-1, 1, 199, 250)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


class InceptionLSTM(nn.Module):

    def __init__(self, in_channels=199, batch=128, input_size=250, out_channels=3):
        super(InceptionLSTM, self).__init__()
        self.inception = Inception(in_channels, out_channels=out_channels)
        self.lstm1 = nn.LSTM(input_size, input_size)
        self.lstm2 = nn.LSTM(input_size, input_size)
        self.lstm3 = nn.LSTM(input_size, input_size)

        # self.linear = nn.Linear(out_channels*24, 46)
        self.last_conv = nn.Conv1d(out_channels, 46, kernel_size=3, padding=1)
        self.batch = batch
        self.out_channels = out_channels
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        # self.linear = nn.Linear(9600, 199*9600)

    def forward(self, x):
        out = x.permute(1, 0, 2)
        out, _ = self.lstm1(out)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)

        out = out.permute(1, 0, 2)
        # x = self.linear(x)
        # x = x.reshape(-1, 200, 47)
        out = self.inception(out)
        # out = out.permute(1, 0, 2).reshape(-1, 64*46)
        out = self.last_conv(out)
        # out = nn.Dropout(0.5)(out)
        out = nn.functional.avg_pool1d(out, kernel_size=out.size()[2])
        out = out.reshape(-1, 46)
        out = nn.Softmax(1)(out)
        return out


class InceptionLSTM2d(nn.Module):

    def __init__(self, in_channels=1, batch=128, input_size=250, out_channels=3):
        super(InceptionLSTM2d, self).__init__()
        self.inception = Inception2d(in_channels, out_channels=out_channels)
        self.lstm1 = nn.LSTM(input_size, input_size)
        self.lstm2 = nn.LSTM(input_size, input_size)
        self.lstm3 = nn.LSTM(input_size, input_size)
        self.lstm4 = nn.LSTM(input_size, input_size)
        # self.linear = nn.Linear(out_channels*24, 46)
        self.last_conv = nn.Conv2d(out_channels, 46, kernel_size=3, padding=1)
        self.batch = batch
        self.out_channels = out_channels
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)

        # self.linear = nn.Linear(9600, 199*9600)

    def forward(self, x):
        out = x.permute(1, 0, 2)
        out, _ = self.lstm1(out)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out, _ = self.lstm4(out)
        out = out.permute(1, 0, 2)
        # x = self.linear(x)
        # x = x.reshape(-1, 200, 47)
        out = self.inception(out)
        # out = out.permute(1, 0, 2).reshape(-1, 64*46)
        out = self.last_conv(out)
        # out = nn.Dropout(0.5)(out)
        out = nn.functional.avg_pool2d(out, kernel_size=out.size()[2])
        out = out.reshape(-1, 46)
        out = nn.Softmax(1)(out)
        return out

# model = InceptionLSTM().cuda()
# torchsummary.summary(model, (200, 47))
