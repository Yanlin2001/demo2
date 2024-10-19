import pyedflib
import numpy as np
from tqdm import tqdm
import mne
from scipy.signal import welch,stft
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
import pywt
from scipy.signal import welch

# @NOTE: 从小波分解后的子带信号中提取特征
# @IDEA: 包括标准差（STD）、功率谱密度（PSD）、频带能量和模糊熵（FuzzyEn）
# @DATA: signal.shape = (1024,) = (256 * 4,)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def extract_stft_features(data, fs, window_length_sec=4):
    """
    使用短时傅里叶变换（STFT）从 EEG 数据中提取特征，并生成频谱图。

    :param data: EEG 信号数据。
    :param fs: 采样频率。
    :param window_length_sec: STFT的每个窗口长度（秒）。
    :return: 从 STFT 提取的特征。
    """
    # @DATA: data.shape = (768,) fs = 256 window_length_sec = 3
    # 执行 STFT
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec * fs)
    
    '''    # 绘制频谱图
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        plt.show()
    '''
    # 从 STFT 提取特征
    # 计算每个频率带的平均功率
    power = np.mean(np.abs(Zxx)**2, axis=1)  # 每个频率下的平均功率

    return power
def extract_advanced_features(data, fs, window_length_sec=4):
    """
    使用短时傅里叶变换（STFT）从 EEG 数据中提取高级特征，并生成频谱图。

    :param data: EEG 信号数据。
    :param fs: 采样频率。
    :param window_length_sec: STFT的每个窗口长度（秒）。
    :return: 从 STFT 提取的特征。
    """
    # @DATA: data.shape = (768,) fs = 256 window_length_sec = 3
    # 执行 STFT
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec * fs)
    
    '''    # 绘制频谱图
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        plt.show()
    '''
    # 从 STFT 提取特征
    # 计算每个频率带的平均功率
    power = np.mean(np.abs(Zxx)**2, axis=1)  # 每个频率下的平均功率

    return power


# @NOTE: 预处理和提取特征
def preprocess_and_extract_features_mne_with_timestamps(file_name):
    """
    使用 mne 库预处理 EEG 数据，并提取基础和高级特征。
    在每个特征数组的开头加入对应的时间戳。
    """

    # 加载数据
    ## 加载原始数据
    # @DATA: raw.n_times = 1 * 60 * 60 * 256 = 921600
    raw = mne.io.read_raw_edf(file_name, preload=True)

    # 应用带通滤波器
    raw.filter(1., 50., fir_design='firwin')

    # 选择 EEG 通道
    raw.pick_types(meg=False, eeg=True, eog=False)

    # 定义短时间窗口的参数
    # @DATA: 921600
    window_length = 4  # 窗口长度（秒）
    # @DATA: sfreq = 256
    sfreq = raw.info['sfreq']  # 采样频率
    # @DATA: window_samples = 256 * 4 = 1024
    window_samples = int(window_length * sfreq)
    # 初始化一个空列表来存储特征和时间戳
    features_with_timestamps = []

    # @DATA: total_windows = 921600 / 1024 = 900
    # 获取总窗口数以便在进度条中使用
    total_windows = len(raw.times) // window_samples

    # 使用tqdm包装range对象以显示进度条
    for start in tqdm(range(0, len(raw.times), window_samples), total=total_windows, desc="Processing windows"):
        end = start + window_samples
        if end > len(raw.times):
            break

        # 提取并预处理这个窗口中的数据
        # @DATA: window_data.shape = (23, 768) times.shape = (768,) = (256 * 3,)
        window_data, times = raw[:, start:end] 
        window_data = np.squeeze(window_data) ## ???

        # 获取窗口的开始时间戳
        timestamp = raw.times[start]
        features_per_channel = []

        # 为每个通道的每个窗口提取基础和高级特征
        # @DATA: channel_data.shape = (768,)
        for channel_data in window_data:
            # @DATA: wavelet_features.shape = (18,)
            advanced_features = extract_advanced_features(channel_data, sfreq)
            # @DATA: combined_features.shape = (392,)
            combined_features = np.concatenate([[timestamp], advanced_features])

            features_with_timestamps.append(combined_features)
    # @NOTE: features_with_timestamps中每23行为一个窗口的23个通道的特征
    # @DATA: features_with_timestamps.shape = (27600,392) = (1200 * 23, 18)
    return np.array(features_with_timestamps)