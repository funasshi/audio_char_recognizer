import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os


def get_audio_numpy(file_name):
    sound = AudioSegment.from_file(file_name, "m4a")
    sound_arr = np.array(sound.get_array_of_samples())
    sound_arr = remove_start_end(sound_arr, sound.frame_rate)
    sound_arr = clipping(sound_arr)
    sound_arr = sub_sampling(sound_arr)
    frame_rate = sound.frame_rate / 10
    x = np.arange(
        start=0, stop=sound_arr.shape[0] / frame_rate, step=1 / frame_rate)
    return sound_arr, x


def clipping(sound_arr):
    abs_sound_arr = np.abs(sound_arr)
    average = np.average(abs_sound_arr)
    sound_arr[abs_sound_arr > average * 10] = average * 10
    return sound_arr


def remove_start_end(sound_arr, frame_rate, start_cut_time=1, end_cut_time=1):
    # start_time: 始め何秒をcutするか
    # end_time:終わり何秒をcutするか
    return sound_arr[start_cut_time * frame_rate:-end_cut_time * frame_rate]


def sub_sampling(sound_arr):
    return sound_arr[::10]


def average_array(array, type, average):
    if type == "noise":
        return np.average(array[array < 3 * average])
    else:
        return np.average(array)


def average_power(sound_arr, window_size, type, fr=4800, alpha=0.1):
    strides = window_size / 3
    abs_sound_arr = np.abs(sound_arr)
    average = np.average(abs_sound_arr)
    abs_sound_arr[abs_sound_arr > average * 10] = average * 10
    average = np.average(abs_sound_arr)
    average_noise_power_list = []
    first_frame = abs_sound_arr[:int(window_size * fr)]
    Pt = average_array(first_frame, type, average)
    average_noise_power_list += [Pt] * int(strides * fr)
    for i in range(int(strides * fr), sound_arr.shape[0], int(strides * fr)):
        Pt = (1 - alpha) * Pt + alpha * \
            average_array(
                abs_sound_arr[i:int(i + window_size * fr)], type, average)
        average_noise_power_list += [Pt] * int(strides * fr)
    return np.array(average_noise_power_list[:sound_arr.shape[0]])


def get_peak(sound_arr, gamma=1.3, ):
    noise_array = average_power(sound_arr, window_size=1, type="noise")
    signal_array = average_power(sound_arr, window_size=0.04, type="signal")
    return np.array(signal_array > (noise_array * gamma))


# 長さ21.6秒
# frame rate=48000Hz
# arrayは1037312

# folder_name = "hiragana_audio_folder"
# hiragana = "そ"
# sound_arr, x = get_audio_numpy(os.path.join(folder_name,hiragana+"-舟橋①.m4a"))
# peak = get_peak(sound_arr)
# plt.plot(x, sound_arr*peak)
# plt.show()

folder_name = "hiragana_audio_folder"
hiragana = "そ"
sound_arr, x = get_audio_numpy(
    os.path.join(folder_name, hiragana + "-舟橋①.m4a"))

file_name = os.path.join(folder_name, hiragana + "-舟橋①.m4a")
sound = AudioSegment.from_file(file_name, "m4a")
sound_arr = np.array(sound.get_array_of_samples())

fk = np.fft.fft(sound_arr)
freq = np.fft.fftfreq(fk.shape[0])
plt.plot(freq, np.abs(fk))
plt.show()
