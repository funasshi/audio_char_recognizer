import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


def get_audio_numpy(file_name):
    # 入力:音声ファイルのファイル名
    # 出力:音声ファイルのnumpy配列,frame_rate

    sound = AudioSegment.from_file(file_name, "m4a")
    sound_arr = np.array(sound.get_array_of_samples())  # numpy配列取得

    frame_rate = sound.frame_rate  # frame_rate取得

    sound_arr = sub_sampling(sound_arr)
    frame_rate /= 10  # サブサンプリング
    return sound_arr, int(frame_rate)


def sub_sampling(sound_arr):
    # サブサンプリング関数
    return sound_arr[::10]


def audio_split(sound_arr, frame_rate):
    # 音声をひらがな一文字ずつに分ける
    sound_arr = remove_start_end(sound_arr, frame_rate)
    return sound_arr.reshape(-1, int(frame_rate*2))


def remove_start_end(sound_arr, frame_rate, start_cut_time=1, length=20):
    # start_time: 始め何秒をcutするか
    # length:何秒間切り取るか
    end_cut_time = start_cut_time+length
    return sound_arr[int(start_cut_time * frame_rate):int(end_cut_time * frame_rate)]


def plot_hiragana_audio(hiragana):
    # 入力:表示したいひらがな(str)
    # 出力:10個の波形
    file_name = "easily_splittable_hiragana_data/" + hiragana + "1.m4a"
    sound_arr, frame_rate = get_audio_numpy(file_name)
    sound_arrs = audio_split(sound_arr, frame_rate)
    for i, sound_arr in enumerate(sound_arrs):
        plt.subplot(2, 5, i+1)
        plt.plot(sound_arr)
    plt.show()


def make_dataset():
    # 入力:表示したいひらがな(str)
    # 出力:10個の波形
    hiraganas = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
    X = None
    labels = []
    for label, hiragana in enumerate(hiraganas):
        for i in range(2):
            file_name = "easily_splittable_hiragana_data/" + \
                hiragana + str(i+1)+".m4a"
            sound_arr, frame_rate = get_audio_numpy(file_name)
            sound_arrs = audio_split(sound_arr, frame_rate)
            labels += [label]*sound_arrs.shape[0]
            if X is None:
                X = sound_arrs
            else:
                X = np.concatenate([X, sound_arrs])
    return X, labels


X, labels = make_dataset()

print(X)
