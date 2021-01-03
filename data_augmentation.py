import numpy as np
import torch
import random


def add_white_noise(x, rate=0.005):
    return x + rate * torch.randn(x.shape[0]).cuda()


def shift_sound(x):
    result = torch.zeros_like(x)
    start = random.randint(50, 100)
    end = random.randint(9350, 9400)
    start2 = random.randint(0, 25)
    result[start2:start2+end-start] = x[start:end]
    return result


# def stretch_sound(x, rate=1.1):
#     input_length = len(x)
#     x = librosa.effects.time_stretch(x, rate)
#     if len(x) > input_length:
#         return x[:input_length]
#     else:
#         return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def data_augmentation(sound_arr, noise=True, shift=True):
    if noise:
        sound_arr = add_white_noise(sound_arr)
    if shift:
        sound_arr = shift_sound(sound_arr)
    # stretch_sound_arr = stretch_sound(sound_arr)
    return sound_arr
