import numpy as np

# 以下はファイルパスなどを適当に設定
from audio_preprocess import audio_to_psd  # ワイが作ったモジュール
from word_processor3 import make_dataset  # ふなっしーが作ったモジュール


def get_stpsd():
    '''
    make_datasetによって得られる複数の音声データ(ndarray_sounds, dim=2)とframe_rateを、
    audio_to_psd (dim=1のndarrayを引数とする)を用いてstpsdに変換する。
    返り値: dim=3のndarray, 複数の音声データのstpsd
    '''
    ndarray_sounds, labels, frame_rate = make_dataset()
    num_sounds = ndarray_sounds.shape[0]  # 音声の個数

    ls = []
    for i in range(num_sounds):
        ls.append(audio_to_psd(ndarray_sounds[i], frame_rate, T=None))

    return np.stack(ls, axis=0), labels
