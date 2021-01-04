# 参考url: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/
# %%
from pydub import AudioSegment
import numpy as np
# from scipy.signal.signaltools import wiener
from scipy.fft import fft, ifft
from scipy import signal
# %%


def audio_from_path(path: str):
    '''
    pathから、音声ファイルを読み込み、音声信号のndarrayとサンプリング周波数、音声の時間を返す。
    '''
    sound = AudioSegment.from_file(path, format=path.split('.')[-1])
    ndarray_sound = np.array(sound.get_array_of_samples())
    ndarray_sound = ndarray_sound / np.abs(ndarray_sound).max()  # なんとなく正規化的な
    fs = sound.frame_rate
    T = sound.duration_seconds
    return ndarray_sound, fs, T


def del_dc_component(ndarray_sound):
    '''
    音声信号のDC成分を消す
    '''
    xf = fft(ndarray_sound)
    xf[0] = 0
    xf_if = ifft(xf)
    # ifftで虚数部分はほぼ0になると思うが、ミスってた場合に備えて
    assert np.max(np.abs(np.imag(xf_if))) < 1e-5
    return np.real(xf_if)

# def low_pass(ndarray_sound, fs, T, thresh):
#     '''
#     threshより大きい周波数成分を0にする
#     '''
#     N = len(ndarray_sound)
#     df = fs/N
#     f = np.linspace(1, N, N)*df - df
#     thresh_idx =
#     xf = fft(ndarray_sound)


def audio_to_psd(tensor_sound, fs, time_window=0.02, time_overlap=0.01, lpass: 'bool, low passの処理をするかどうか' = False, lpass_thresh=10000):
    '''
    stPSDを返す。返り値はdim=2のndarray. axis=0方向は各フレーム. axis=1方向はそのフレームでのstPSD.
    '''
    ndarray_sound = tensor_sound.numpy()
    N = len(ndarray_sound)
    dt = 1 / fs

    # 窓関数に関する変数
    lw = int(time_window // dt)  # length of window, 窓巻数の配列の要素数
    lo = int(time_overlap // dt)  # length of overlapm overlapの要素数
    w = signal.hann(lw)  # hanning window

    # N, lw, loからnf (フレーム数、xをいくつに区切って窓関数を適用するか)とlp(パディングの要素数)を求める。
    if (N - lo) % (lw - lo) == 0:
        nf = (N - lo) // (lw - lo)
        lp = 0
    else:
        nf = (N - lo) // (lw - lo) + 1
        lp = (lw - lo) * nf + lo - N

    # xの後ろをlp個の0でpadding
    x_pad = np.concatenate([ndarray_sound, np.zeros(lp)], axis=0)

    stpwds = []  # stpwdsに各フレームのstPSDのndarrayを格納
    df = fs / lw  # 各フレームごとのfftにおける周波数分解能
    f = np.linspace(1, lw // 2, lw // 2) * df - df  # 各フレームごとの周波数
    # nf個のフレームに分けて、hanning windowをかけて、FFT
    for i in range(nf):
        s = (lw - lo) * i  # フレームの開始のindex, sよりindexが小さいところに要素がs個ある
        frame = w * x_pad[s: s + lw]  # hanning windowを掛ける
        frame_f = fft(frame)
        # 正規化及び有効な範囲(fs/2以下の部分)のみを取り出す
        frame_f_regularized = 2 / lw * frame_f[:lw // 2]
        frame_f_regularized[0] = frame_f_regularized[0] / 2  # 正規化
        stpwd = np.abs(frame_f_regularized)**2 / df  # 周波数成分の絶対値の2乗を周波数分解能で割る
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

    clipped_spec = res[:, 80:330]
    clipped_spec = np.log(clipped_spec)
    mean = np.mean(clipped_spec)
    std = np.std(clipped_spec)
    reg_spec = (clipped_spec - mean) / std

    return reg_spec

# %%
# -------------------------------------------------------
# # 音声信号の絶対値をとってから処理する
# # A1未満の信号がT1秒続いたら、そこを文字と文字の間隔と見なし区切る
# T1 = 0.01 # [sec] 文字と文字の区切りの間隔、なるべく小さくする
# A1 = 0.02 #
# T2 = 0.01 # [sec] A1以上の信号がT2秒未満続いた場合はパルスとして無視する, 逆にT2以上続いた場合はパルスとして無視できない

# # n-th sammpleとn+1-th sampleの間の間隔は1/F秒
# F = sound.frame_rate
# T = sound.duration_seconds
# # 文字と文字の区切り(A1未満の信号がT1秒続く)とは、A1未満の信号のsampleがN1 (=F*T1 + 1)個連続することと同じ
# # T1とT2に対応するsampleの個数
# N1 = F * T1 + 1
# N2 = E * T2 + 1
# # 以下のzerooneでは、soundのうち閾値以上の部分が1、それ以外が0となっているようなndarray
# # さらに両端にN1だけ０でパディングしてある
# zeroone = np.concatenate((np.zeros(N1), (ndarray_sound>=A1).astype(np.int), np.zeros(N1)))
# L = len(zeroone)
# # 尺取り法的な
# start = 0 # 閉区間の開始のindex
# end = 0 # 閉区間の末尾のindex, endを含む
# while end < L:
#     # 0の区間の捜査
#     while end < L and zeroone[end] == 0:
#         end += 1
#     # 0続きの区間がN1未満の場合、その区間の0を1に置き換える
#     if end - start < N1:
#         zeroone[start:end] = np.ones(end-start)
#     start = end

#     # 1の区間の捜査, whileが終わった時はzeroone[end]==1 or end == L
#     while end < L and zeroone[end] == 1:
#         end += 1
#     start = end


# #-------------------------------------------------------
# res = []

# n1 = 0 # ndarray_sound[end-1] < A1となるsample数
# n2 = 0 # ndarray_sound[end-1] >= A1となるsample数
# # 諸々の処理をするためのndarray, 最初に0を付け加えるのは一番最初にend-2を見たいから(end-1が注目してるやつで、end-2はその1個手前), 0を最初に加えなかったらendの初期値は1でend-2=-1となる
# ndarray = np.concatenate((np.zeros(1), np.abs(ndarray_sound)))
# L = len(ndarray_sound)

# while end <= L:
#     #while end <= L and n2 < N2:
#     while end <= L:
#         if ndarray[end-2] >= A1 and ndarray[end-1] >= A1:
#             n2 += 1
#         elif ndarray[end-2] >= A1 and ndarray[end-1] < A1:
#             n2 = 0
#         elif ndarray[end-2] < A1 and ndarray[end-1] >= A1:
#             n2 += 1
#         else:
#             pass # n2はそのまま

#         ###
#         if n2 >= N2:
#             break
#         else:
#             end += 1

#     # A1以上の音が続いた場合
#     if n2 == N2:
#         end -= n2


#         if ndarray[end-1]
#         n1 += 1
#         end += 1

#     if end - start > N1:
#         res.append((start, end))


#     start = end
#     end = start + 1
# -------------------------------------------------------
