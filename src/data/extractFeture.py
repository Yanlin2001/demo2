import pyedflib
import numpy as np
from tqdm import tqdm
import mne
from scipy.signal import welch,stft
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
import pywt
from scipy.signal import welch


def extract_advanced_features(data, fs, window_length_sec=24):
    """
    使用短时傅里叶变换（STFT）从 EEG 数据中提取高级特征。

    :param data: EEG 信号数据。
    :param fs: 采样频率。
    :param window_length_sec: STFT的每个窗口长度（秒）。
    :return: 从 STFT 提取的特征。
    """
    # @DATA: data.shape = (768,) fs = 256 window_length_sec = 3
    # 执行 STFT
    # @DATA: f.shape = (385,) t.shape = (3,) Zxx.shape = (385, 3)
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec*fs)
    
    # 从 STFT 提取特征
    # 在此，我们可以从 STFT 中提取各种特征。
    # 为简化起见，这里计算每个频率带的平均功率。
    # 可以通过计算 STFT 幅度的平方的平均值来完成。
    # @DATA: power.shape = (385,)
    power = np.mean(np.abs(Zxx)**2, axis=1)  # 每个频率下的平均功率

    # @NOTE: 只有power（平均功率）这一个特征
    return power


# @NOTE: 从小波分解后的子带信号中提取特征
# @IDEA: 包括标准差（STD）、功率谱密度（PSD）、频带能量和模糊熵（FuzzyEn）
# @DATA: signal.shape = (1024,) = (256 * 4,)
def extract_wavelet_features(signal, sfreq):
    """
    对信号进行离散小波变换 (DWT)，并提取每个子带的特征。

    :param signal: EEG 信号。
    :param sfreq: 采样频率。
    :return: 小波分解后的特征。
    """
    # wavelet = 'db4'  # 使用
    # coeffs = pywt.wavedec(signal, wavelet, level=5)

    # @DATA: {依次折半} cD1.shape = (512,) cD2.shape = (256,) cD3.shape = (128,) cD4.shape = (64,) cD5.shape = (32,) cA5.shape = (16,)
    # D1-D5 是细节系数，A5 是近似系数
    (A1,D1) = pywt.dwt(signal, 'haar')
    (A2,D2) = pywt.dwt(A1, 'haar')
    (A3,D3) = pywt.dwt(A2, 'haar')
    (A4,D4) = pywt.dwt(A3, 'haar')
    (A5,D5) = pywt.dwt(A4, 'haar')
    cA5 , cD5, cD4, cD3, cD2, cD1 = A5,D5,D4,D3,D2,D1
    # cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 初始化一个空列表来存储子带特征
    wavelet_features = []

    # 对每个子带提取特征
    for subband in [cD1, cD2, cD3, cD4, cD5, cA5]:
        # 计算标准差
        std = np.std(subband)

        # 计算功率谱密度 (PSD)
        # @DATA: freqs.shape = (129,) psd.shape = (129,) 129 = 256 / 2 + 1
        # @BUG: UserWarning: nperseg = 256 is greater than input length  = 192 , using nperseg = 192 
        # @BUG: sfreq = 256是调用函数的时候传入的，但是这里的subband长度是小波变换后的长度，不是原始信号的长度
        new_sfreq = subband.shape[0]
        freqs, psd = welch(subband, subband.shape[0])
        mean_psd = np.mean(psd)

        # 计算带能量
        band_energy = np.sum(np.square(subband))

        # 将特征添加到列表中
        wavelet_features.extend([std, mean_psd, band_energy])
        # @DATA: wavelet_features.shape = (3 * 6,) = (18,)
        # wavelet_features.extend([std, mean_psd, band_energy])

    return wavelet_features

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
    window_length = 24  # 窗口长度（秒）
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
    total_windows = len(range(0, len(raw.times), window_samples))
    quarter_windows = total_windows // 4  # 25% 的窗口数

    for start in tqdm(range(0, len(raw.times), window_samples), total=total_windows, desc="Processing windows", miniters=quarter_windows):
        end = start + window_samples
        if end > len(raw.times):
            break

        # 提取并预处理这个窗口中的数据
        # @DATA: window_data.shape = (23, 768) times.shape = (768,) = (256 * 3,)
        window_data, times = raw[:, start:end] 
        window_data = np.squeeze(window_data) ## ???

        # 获取窗口的开始时间戳
        timestamp = raw.times[start]

        # 为每个通道的每个窗口提取基础和高级特征
        # @DATA: channel_data.shape = (768,)
        for channel_data in window_data:
            # @DATA: advanced_features.shape = (18,)
            raw_channel_data = channel_data
            len_raw = len(raw_channel_data)
            advanced_features = extract_advanced_features(channel_data, sfreq)
            len_a = len(advanced_features)
            # @DATA: wavelet_features.shape = (18,)
            wavelet_features = extract_wavelet_features(channel_data, sfreq)
            len_w = len(wavelet_features)
            # @DATA: combined_features.shape = (392,)
            combined_features = np.concatenate([[timestamp], raw_channel_data, advanced_features, wavelet_features])

            features_with_timestamps.append(combined_features)
    # @NOTE: features_with_timestamps中每23行为一个窗口的23个通道的特征
    # @DATA: features_with_timestamps.shape = (27600,392) = (1200 * 23, 18)
    # 返回数据长度与数据
    return [len_raw,len_a,len_w],np.array(features_with_timestamps)