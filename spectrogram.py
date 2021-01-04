from pydub import AudioSegment
import numpy as np
# from scipy.signal.signaltools import wiener
from scipy.fft import fft, ifft
from scipy import signal
import torch
from audio_preprocess import audio_to_psd


def audio_to_spectrogram(tensor_sound, fr=4800, time_window=0.02, time_overlap=0.01, lpass: 'bool, low passの処理をするかどうか' = False, lpass_thresh=10000):
    '''
    stPSDを返す。返り値はdim=2のndarray. axis=0方向は各フレーム. axis=1方向はそのフレームでのstPSD.
    '''
    x = tensor_sound.numpy()

    # %%
    NFFT = int(fr * time_window)  # フレームの大きさ
    OVERLAP = NFFT // 2  # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
    frame_length = x.shape[0]  # wavファイルの全フレーム数
    time_song = float(frame_length) / fr  # 波形長さ(秒)
    time_unit = 1 / float(fr)  # 1サンプルの長さ(秒)

    # 💥 1.
    # FFTのフレームの時間を決めていきます
    # time_rulerに各フレームの中心時間が入っています
    start = (NFFT / 2) * time_unit
    stop = time_song
    step = (NFFT - OVERLAP) * time_unit
    time_ruler = np.arange(start, stop, step)

    # 💥 2.
    # 窓関数は周波数解像度が高いハミング窓を用います
    window = np.hamming(NFFT)

    spec = np.zeros([len(time_ruler), 1 + int(NFFT / 2)])  # 転置状態で定義初期化
    pos = 0

    for fft_index in range(len(time_ruler)):
        # 💥 1.フレームの切り出します
        frame = x[pos:pos + NFFT]
        # フレームが信号から切り出せない時はアウトです
        if len(frame) == NFFT:
            # 💥 2.窓関数をかけます
            windowed = window * frame
            # 💥 3.FFTして周波数成分を求めます
            # rfftだと非負の周波数のみが得られます
            fft_result = np.fft.rfft(windowed)
            # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
            # グラフで見やすくするために対数をとります
            fft_data = np.log(np.abs(fft_result) ** 2)
            # fft_data = np.log(np.abs(fft_result))
            # fft_data = np.abs(fft_result) ** 2
            # fft_data = np.abs(fft_result)
            # これで求められました。あとはspecに格納するだけです
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i - 1] = fft_data[i]

            # 💥 4. 窓をずらして次のフレームへ
            pos += (NFFT - OVERLAP)
    clipped_spec = spec[:, 150:400]
    mean = np.mean(clipped_spec)
    std = np.std(clipped_spec)
    reg_spec = (clipped_spec - mean) / std
    return reg_spec


def make_batch_spectrogram(X_train):
    ans = []
    for i in X_train:
        ans.append(audio_to_spectrogram(i))
    return torch.Tensor(np.array(ans))


def make_batch_stpsd(X_train):
    ans = []
    for i in X_train:
        ans.append(audio_to_stpsd(i))
    return torch.Tensor(np.array(ans))
