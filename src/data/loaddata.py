import glob
import os.path
import numpy as np
from src.data.extractFeture import preprocess_and_extract_features_mne_with_timestamps
from src.data.extractTarget import extractTarget
from tqdm import tqdm

# @TODO: 对低级特征、高级特征、时域信息	频域信息、时频域信息分别提取并打上标签
'''def extract_data_and_labels(edf_file_path, summary_file_path):

    # 提取特征
    # @DATA: X.shape = (n_samples, n_features) = (27600,392) = (1200 * 23, 392) = (3600/3 * 23, 391+1)
    X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path)
    # 提取标签
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    # @DATA: y.shape = (n_samples,) = (27600,) = (1200 * 23,)
    y = np.array([1 if seizure_start_time <= row[0] <= seizure_end_time else 0 for row in X])

    # @DATA: X.shape = (27600,392) = (1200 * 23, 391)
    #从X数组中移除第一列Time
    X = X[:,1:]
    return X,y'''



def extract_data_and_labels(edf_file_path, summary_file_path):
    # 提取特征
    # @DATA: X.shape = (20700,19) = (900 * 23, 19) = (3600/4 * 23, 18+1)
    len_info, X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path)
    
    # 提取标签
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    
    # 处理没有发作时间的情况
    if seizure_start_time is None or seizure_end_time is None:
        y = np.zeros(X.shape[0])  # 如果没有发作时间，所有标签都设为0
    else:
        y = np.array([1 if seizure_start_time <= row[0] <= seizure_end_time else 0 for row in X])
    
    # RESHAPE X
    X = X[:,1:]
    
    return X, y, len_info

def load_data(subject_id,base_path):
    """
    加载给定主题的数据。
    会读取给定chb主题的所有edf文件，并从每个文件中提取特征。
    返回一个包含所有数据的列表，以及一个包含所有标签的列表。
    其中，每个数据都是一个形状为 (n_samples, n_features) 的数组，每个标签都是一个形状为 (n_samples,) 的数组。
    """
    # @NOTE: 拼接所有edf文件路径
    # 构建edf文件路径
    edf_file_paths = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
    summary_file_path = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
    all_X = []
    all_y = []
    
    # 使用tqdm包装迭代器以显示进度条
    len_info = None
    for edf_file_path in tqdm(edf_file_paths, desc="Loading EDF files"):
        X, y , len_info = extract_data_and_labels(edf_file_path, summary_file_path)
        all_X.append(X)
        all_y.append(y)
    return all_X, all_y, len_info

'''    for edf_file_path in edf_file_path:
        X, y = extract_data_and_labels(edf_file_path, summary_file_path)
        all_X.append(X)
        all_y.append(y)'''

#使用方法：
# subject_id = 1
# base_path = "data"
# all_X,all_y = load_data(subject_id,base_path)

#对于all_y每个数据，统计1和0的个数并打印
# total_n_count = 0
# total_p_count = 0
# for y in all_y:
#     p_count = 0
#     n_count = 0
#     for lable in y:
#         if lable == 1:
#             p_count += 1
#         else:
#             n_count += 1
#     total_n_count += n_count
#     total_p_count += p_count
# print("total_p_count/total_count:",total_p_count/(total_n_count+total_p_count))

## total_p_count/total_count: 0.018808777429467086