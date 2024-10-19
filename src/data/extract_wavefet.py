def preprocess_and_extract_features_mne_with_timestamps(file_name):
    """
    使用 mne 库预处理 EEG 数据，并提取基础和高级特征。
    返回的数组形状为 (1200, 23, 18)，其中 1200 是窗口数，23 是通道数，18 是每个通道提取的特征数。
    """

    # 加载数据
    raw = mne.io.read_raw_edf(file_name, preload=True)

    # 应用带通滤波器
    raw.filter(1., 50., fir_design='firwin')

    # 选择 EEG 通道
    raw.pick_types(meg=False, eeg=True, eog=False)

    # 定义短时间窗口的参数
    window_length = 4  # 窗口长度（秒）
    sfreq = raw.info['sfreq']  # 采样频率
    window_samples = int(window_length * sfreq)
    
    # 初始化列表来存储特征和时间戳
    all_features = []
    timestamps = []

    total_windows = len(raw.times) // window_samples

    # 使用tqdm包装range对象以显示进度条
    for start in tqdm(range(0, len(raw.times), window_samples), total=total_windows, desc="Processing windows"):
        end = start + window_samples
        if end > len(raw.times):
            break

        # 提取并预处理这个窗口中的数据
        window_data, times = raw[:, start:end]
        window_data = np.squeeze(window_data)

        # 获取窗口的开始时间戳
        timestamp = raw.times[start]
        timestamps.append(timestamp)

        # 存储当前窗口的特征
        window_features = []

        # 为每个通道提取基础和高级特征
        for channel_data in window_data:
            wavelet_features = extract_wavelet_features(channel_data, sfreq)
            window_features.append(wavelet_features)

        # 将每个窗口的通道特征存储为 2D 数组 (23, 18)
        all_features.append(window_features)

    # 将所有窗口的特征转换为 3D 数组 (1200, 23, 18)
    all_features = np.array(all_features)

    return np.array(timestamps), all_features
