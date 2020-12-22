import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal.signaltools import wiener

def get_audio_numpy(file_name):
    # 音声ファイルをnumpyで出力。frame_rateも出力
    sound = AudioSegment.from_file(file_name, "m4a")
    frame_rate = sound.frame_rate
    sound_arr = np.array(sound.get_array_of_samples())
    sound_arr = remove_start_end(sound_arr, sound.frame_rate)
    # sound_arr = sub_sampling(sound_arr)
    # sound_arr = noise_reduction(sound_arr, frame_rate)
    # frame_rate /= 10
    # x = np.arange(start=0, stop=sound_arr.shape[0] / frame_rate, step=1 / frame_rate)
    return sound_arr,frame_rate

def remove_start_end(sound_arr, frame_rate, start_cut_time=1, end_cut_time=1):
    # start_time: 始め何秒をcutするか
    # end_time:終わり何秒をcutするか
    return sound_arr[start_cut_time * frame_rate:-end_cut_time * frame_rate]


def noise_reduction(sound_arr,window_size,frame_rate=48000, s1=0.02,overlap=0.01):
    #
    # hanning_windowのwindow_sizeは0.02s, overlapは0.01s
    noise_removed_arr=np.zeros_like(sound_arr,dtype="float64")
    hanning_window=np.hanning(int(s1*frame_rate))
    for i in range(0, sound_arr.shape[0],int(overlap*frame_rate)):
        if i+s1*frame_rate<=sound_arr.shape[0]:
            sound_arr_frame=sound_arr[i:int(i+s1*frame_rate)]
            noise_removed_arr[i:int(i+s1*frame_rate)]+=(wiener(sound_arr_frame*hanning_window,window_size)//2)
    return noise_removed_arr

file_name = "sample2.m4a"
sound_arr ,frame_rate= get_audio_numpy(file_name)


def get_entire_denoised(sound_arr,window_size):
    removed_sound_arr = np.zeros_like(sound_arr, dtype="float64")
    # processing_window=sound_arr[:48000//4]

    for i in range(0, sound_arr.shape[0], int(0.25*48000)):
        removed_sound_arr[i:int(i+0.25*48000)] = noise_reduction(sound_arr[i:int(i+0.25*48000)], window_size)
    # removed_sound_arr=wiener(sound_arr,int(sound_arr.shape[0]/1000))
    return removed_sound_arr

def split_to_window(sound_arr,window_size=0.25,frame_rate=48000):
    pro_windows=[]
    for i in range(0, sound_arr.shape[0], int(window_size * frame_rate)):
        pro_windows.append(sound_arr[i:int(i+window_size*frame_rate)])
    return pro_windows

def split_to_frame(pro_windows,window_size=0.02,overlap_size=0.01,frame_rate=48000):
    frames=[]
    for i in range(0, pro_windows.shape[0], int(overlap_size * frame_rate)):
        frames.append(pro_windows[i:int(i+window_size*frame_rate)])
    return frames

def log_ste_frame(frame):
    return 10*np.log(np.sum(frame**2)+1e-10)

def sum_log_ste_frame(frames):
    sum_=0
    log_ste_list=[]
    for frame in frames:
        log_ste=log_ste_frame(frame)
        sum_ +=log_ste
        log_ste_list.append(log_ste)
    return sum_,log_ste_list
    
def log_ste_processing(sound_arr,alpha,beta):
    log_ste_list=[]
    average_log_ste=[]
    pro_windows=split_to_window(sound_arr)
    for i, pro_window in enumerate(pro_windows):
        frames=split_to_frame(pro_window)
        sum_, log_ste=sum_log_ste_frame(frames)
        log_ste_list+=log_ste
        if i==0:
            Ep=sum_/len(frames)
        else:
            Ep=(1-alpha)*average_log_ste[-1]+alpha*sum_/len(frames)
        average_log_ste.append(Ep)
    average_log_ste=[i*beta for i in average_log_ste]
    return average_log_ste,log_ste_list


window_sizes=[5,10,15,20]
# fig = plt.figure() # Figureオブジェクトを作成

# for i,window_size in enumerate(window_sizes):
#     ax = fig.add_subplot(len(window_sizes),1,i+1) # figに属するAxesオブジェクトを作成
#     removed_sound_arr=get_entire_denoised(sound_arr,window_size)
#     ax.plot(removed_sound_arr)
# # plt.plot(sound_arr,color="red")
# # fig.plot(removed_sound_arr, color="blue")
# plt.plot(sound_arr)
# plt.show()
sound_arr=sound_arr/100
average_log_ste,log_ste_list=log_ste_processing(sound_arr,alpha=0.3,beta=0.6)
average_log_ste=np.repeat(average_log_ste, 25)
plt.plot(log_ste_list)
plt.plot(average_log_ste)

plt.show()
