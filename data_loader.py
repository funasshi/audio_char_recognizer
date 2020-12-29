from torch.utils.data import DataLoader
import numpy as np


class Dataset:
    def __init__(self, X=np.arange(10), y=np.arange(10)[::-1]):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.y[index]


def audio_data_loader(X, y, batch_size, shuffle=True):
    dataset = Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=True)
    return dataloader
