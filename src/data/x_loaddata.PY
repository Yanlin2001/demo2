import glob
import os.path
import numpy as np
from src.data.extractFeture import preprocess_and_extract_features_mne_with_timestamps
from src.data.extractTarget import extractTarget
from tqdm import tqdm

def extract_data_and_labels(edf_file_path, summary_file_path):
    # 提取特征
    # @DATA: X.shape = (20700,19) = (900 * 23, 19) = (3600/4 * 23, 18+1)
    X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path)
    
    # 提取标签
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    
    # 处理没有发作时间的情况
    if seizure_start_time is None or seizure_end_time is None:
        y = np.zeros(X.shape[0])  # 如果没有发作时间，所有标签都设为0
    else:
        y = np.array([1 if seizure_start_time <= row[0] <= seizure_end_time else 0 for row in X])
    
    # RESHAPE X
    X1d = X[:, 1:]
    X2d = reshape_X_by_timestamp(X)
    y2d = merge_y_by_consistency(y)
    
    return X1d, X2d, y, y2d

def load_data(subject_id,base_path):

    # @NOTE: 拼接所有edf文件路径
    # 构建edf文件路径
    edf_file_paths = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
    summary_file_path = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
    all_X1d = []
    all_X2d = []
    all_y1d = []
    all_y2d = []
    
    # 使用tqdm包装迭代器以显示进度条
    for edf_file_path in tqdm(edf_file_paths, desc="Loading EDF files"):
        X1d, X2d, y, y2d = extract_data_and_labels(edf_file_path, summary_file_path)
        all_X1d.append(X1d)
        all_X2d.append(X2d)
        all_y1d.append(y)
        all_y2d.append(y2d)
    return all_X1d,all_X2d,all_y1d,all_y2d