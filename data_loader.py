from torch.utils.data import DataLoader
import numpy as np
from data_augmentation import data_augmentation
from audio_preprocess import audio_to_psd
from spectrogram import audio_to_spectrogram
import torch


class Dataset:
    def __init__(self, X, y, feature, fr, aug_noise, aug_shift):
        self.X = X
        self.y = y
        self.feature = feature
        self.aug_noise = aug_noise
        self.aug_shift = aug_shift
        self.fr = fr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        x = self.X[index]
        y = self.y[index]
        x = data_augmentation(x, self.aug_noise, self.aug_shift)
        if self.feature == "spectro":
            x = x.cpu()
            x = torch.Tensor(audio_to_spectrogram(x, self.fr))
            x = x.cuda()
        elif self.feature == "stpsd":
            x = x.cpu()
            x = torch.Tensor(audio_to_psd(x, self.fr))
            x = x.cuda()

        return x, y


def audio_data_loader(X, y, batch_size, fr, shuffle=True, feature=None, aug_noise=False, aug_shift=False):
    dataset = Dataset(X, y, feature, fr, aug_noise, aug_shift)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=False)
    return dataloader
