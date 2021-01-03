from pydub import AudioSegment
import numpy as np
# from scipy.signal.signaltools import wiener
from scipy.fft import fft,  ifft
from scipy import signal
import torch


def audio_to_spectrogram(tensor_sound, fs=4800, time_window=0.02, time_overlap=0.01, lpass: 'bool, low passの処理をするかどうか' = False, lpass_thresh=10000):
    '''
    stPSDを返す。返り値はdim=2のndarray. axis=0方向は各フレーム. axis=1方向はそのフレームでのstPSD.
    '''

    tensor_sound = tensor_sound.cpu()
    ndarray_sound = tensor_sound.numpy()
    N = len(ndarray_sound)
    dt = 1/fs

    # 窓関数に関する変数
    lw = int(time_window//dt)  # length of window, 窓巻数の配列の要素数
    lo = int(time_overlap//dt)  # length of overlapm overlapの要素数
    w = signal.hann(lw)  # hanning window

    # N, lw, loからnf (フレーム数、xをいくつに区切って窓関数を適用するか)とlp(パディングの要素数)を求める。
    nf = (N-lo) // (lw-lo)

    stpwds = []  # stpwdsに各フレームのstPSDのndarrayを格納
    df = fs / lw  # 各フレームごとのfftにおける周波数分解能
    f = np.linspace(1, lw//2, lw//2)*df - df  # 各フレームごとの周波数
    # nf個のフレームに分けて、hanning windowをかけて、FFT
    for i in range(nf):
        frame = np.zeros_like(ndarray_sound)
        s = (lw - lo) * i  # フレームの開始のindex, sよりindexが小さいところに要素がs個ある

        frame[s:s+lw] = w * ndarray_sound[s: s+lw]  # hanning windowを掛ける
        frame_f = fft(frame)
        # 正規化及び有効な範囲(fs/2以下の部分)のみを取り出す
        frame_f_regularized = 2/lw * frame_f[:N//2]
        frame_f_regularized[0] = frame_f_regularized[0]/2  # 正規化
        stpwd = np.abs(frame_f_regularized)**2 / \
            df  # 周波数成分の絶対値の2乗を周波数分解能で割る
        # f = np.linspace(1, lw//2, lw//2)*df - df
        # plt.plot(f, stpwd)
        # concatenateで2次元配列にしたいのでnewaxisを加える,
        stpwds.append(stpwd[np.newaxis, :])

    res = np.concatenate(stpwds, axis=0)

    # low pass処理
    if lpass:
        # axis1方向の、thresh_idx未満にある要素はlpass_thresh以下の周波数に対応する
        thresh_idx = np.where(f > lpass_thresh)[0][0]
        res = res[:, :thresh_idx]

    return torch.Tensor(res).cuda()
