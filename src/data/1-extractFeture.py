import pyedflib
import numpy as np
import tqdm
import mne
from scipy.signal import welch,stft
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean

import pywt
from scipy.signal import welch

# @NOTE: 提取基础特征
def extract_basic_features(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    mean = np.mean(signal)   #计算平均值
    std = np.std(signal)  #计算标准差
    sample_entropy = np.log(np.std(np.diff(signal)))  #计算样本熵
    fuzzy_entropy = -np.log(euclidean(signal[:-1], signal[1:]) / len(signal)) #计算模糊熵
    skewness = skew(signal)  #计算偏度
    kurt = kurtosis(signal)  #计算峰度
    return [mean, std, sample_entropy, fuzzy_entropy, skewness, kurt]

# @NOTE: 提取高级特征
def extract_advanced_features(data, fs, window_length_sec=3):
    """
    使用短时傅里叶变换（STFT）从 EEG 数据中提取高级特征。

    :param data: EEG 信号数据。
    :param fs: 采样频率。
    :param window_length_sec: STFT的每个窗口长度（秒）。
    :return: 从 STFT 提取的特征。
    """

    # 执行 STFT
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec*fs)
    
    # 从 STFT 提取特征
    # 在此，我们可以从 STFT 中提取各种特征。
    # 为简化起见，这里计算每个频率带的平均功率。
    # 可以通过计算 STFT 幅度的平方的平均值来完成。
    power = np.mean(np.abs(Zxx)**2, axis=1)  # 每个频率下的平均功率

    return power

# @NOTE: 从小波分解后的子带信号中提取特征
# @IDEA: 包括标准差（STD）、功率谱密度（PSD）、频带能量和模糊熵（FuzzyEn）
def extract_wavelet_features(signal, sfreq):
    """
    对信号进行离散小波变换 (DWT)，并提取每个子带的特征。

    :param signal: EEG 信号。
    :param sfreq: 采样频率。
    :return: 小波分解后的特征。
    """
    wavelet = 'db4'  # 使用 Daubechies 小波
    coeffs = pywt.wavedec(signal, wavelet, level=5)

    # D1-D5 是细节系数，A5 是近似系数
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 初始化一个空列表来存储子带特征
    wavelet_features = []

    # 定义一个函数来计算模糊熵
    def fuzzy_entropy(signal, m=2, r=0.2):
        N = len(signal)
        def _phi(m):
            x = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (N - m + 1)
            return C
        return -np.log(_phi(m + 1) / _phi(m))

    # 对每个子带提取特征
    for subband in [cD1, cD2, cD3, cD4, cD5, cA5]:
        # 计算标准差
        std = np.std(subband)

        # 计算功率谱密度 (PSD)
        freqs, psd = welch(subband, sfreq)
        mean_psd = np.mean(psd)

        # 计算带能量
        band_energy = np.sum(np.square(subband))

        # 计算模糊熵
        fuzzyEn = fuzzy_entropy(subband)

        # 将特征添加到列表中
        wavelet_features.extend([std, mean_psd, band_energy, fuzzyEn])

    return wavelet_features

# @NOTE: 修改的预处理和提取特征函数
def preprocess_and_extract_features_mne_with_timestamps(file_name):
    """
    使用 mne 库预处理 EEG 数据，并提取基础、小波分解和高级特征。
    在每个特征数组的开头加入对应的时间戳。
    """

    # 加载数据
    raw = mne.io.read_raw_edf(file_name, preload=True)

    # 应用带通滤波器
    raw.filter(1., 50., fir_design='firwin')

    # 选择 EEG 通道
    raw.pick_types(meg=False, eeg=True, eog=False)

    # 定义短时间窗口的参数
    window_length = 3  # 窗口长度（秒）
    sfreq = raw.info['sfreq']  # 采样频率
    window_samples = int(window_length * sfreq)

    # 初始化一个空列表来存储特征和时间戳
    features_with_timestamps = []

    # 遍历每个窗口中的数据
    for start in range(0, len(raw.times), window_samples):
        end = start + window_samples
        if end > len(raw.times):
            break

        # 提取并预处理这个窗口中的数据
        window_data, times = raw[:, start:end]
        window_data = np.squeeze(window_data)

        # 获取窗口的开始时间戳
        timestamp = raw.times[start]

        # 为每个通道的每个窗口提取基础、小波和高级特征
        for channel_data in window_data:
            # 提取基础特征
            basic_features = extract_basic_features(channel_data)

            # 提取小波特征
            wavelet_features = extract_wavelet_features(channel_data, sfreq)

            # 提取高级特征
            advanced_features = extract_advanced_features(channel_data, sfreq)

            # 组合所有特征并加入时间戳
            combined_features = np.concatenate([[timestamp], basic_features, wavelet_features, advanced_features])
            features_with_timestamps.append(combined_features)

    return np.array(features_with_timestamps)


# @NOTE: 数据读取
# preprocess_and_extract_features_mne_with_timestamps(r"C:\Users\LinziGoooosh\Downloads\chbmit\1.0.0\chb01\chb01_03.edf")