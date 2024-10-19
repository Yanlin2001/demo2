# @NOTE: 原始数据读取在extractFeture.py中
#from data import extractFeture
import time
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv1D, MaxPooling1D
import pickle
# 记录数据预处理开始时间
data_preprocess_start_time = time.time()

subject_ids = [1, 2, 3]
# 打印当前目录
print(os.getcwd())
base_path = "data" # 数据存放路径
# 初始化 all_X 和 all_y 为空列表，用于存储所有主题的 X 和 y 数据
all_X = []
all_y = []

for subject_id in subject_ids:
    data_file = f"{base_path}/data_subject_{subject_id}.pkl"
    
    # 检查数据是否已存在
    if os.path.exists(data_file):
        print(f"Loading data from {data_file} for subject {subject_id}...")
        with open(data_file, 'rb') as f:
            X, y,len_info = pickle.load(f)
    else:
        print(f"Data file not found, loading and saving data for subject {subject_id}...")
        X, y,len_info = load_data(subject_id, base_path)

        # 保存数据
        with open(data_file, 'wb') as f:
            pickle.dump((X, y, len_info), f)
        print(f"Data saved to {data_file}.")

    # 将数据追加到all_X和all_y
    len_raw = len_info[0]
    len_a = len_info[1]
    len_w = len_info[2]
    all_X.append(X)
    all_y.append(y)
    len_per_subject = len(y[0])



# @TODO:23个通道的特征只是简单的拼接在一起，没有用到通道之间的关系
# 合并 all_X 和 all_y, vstack是垂直合并, concatenate是水平合并
# @DATA: X.shape = (27600,18), y.shape = (27600,) # (27600,18) -> (1200 , 23, 18) -> (1200 , 23 , 6 , 3)


X = np.vstack(all_X)
y = np.concatenate(all_y)
#X = X.reshape(-1, len_w)
X = X.reshape(-1, len_raw + len_a + len_w)
y = y.reshape(-1)


# 记录数据预处理结束时间
data_preprocess_end_time = time.time()

# 计算数据预处理耗时
data_preprocess_time = data_preprocess_end_time - data_preprocess_start_time
print(f"Data preprocess time: {data_preprocess_time:.2f} seconds")

# 记录过采样开始时间
oversampling_start_time = time.time()
# 初始化 SMOTE 实例
smote = SMOTE()

# @NOTE: SMOTE: Synthetic Minority Over-sampling Technique
# 1. 找到标签 y 中的少数类（例如 y=1）
# 2. 在少数类样本的特征空间中，通过现有样本之间的插值生成新的样本
# 3. 生成的新样本与少数类样本相似，从而增加少数类样本数量，平衡类别分布

# 应用 SMOTE 过采样
# @NOTE: 过采样平衡数据
# @NOTE: 过采样：对少数类样本进行插值，增加样本数量，使得少数类样本与多数类样本数量接近相等（不超原数据两倍）
# @NOTE: y = 1/0
X_resampled, y_resampled = smote.fit_resample(X, y)
print(X_resampled.shape, y_resampled.shape) # (38189, 391) (38189,)
# X2d_resampled, y2d_resampled = smote.fit_resample(X2d, y2d)
# 记录过采样结束时间
oversampling_end_time = time.time()

# 计算过采样耗时
oversampling_time = oversampling_end_time - oversampling_start_time
print(f"Oversampling time: {oversampling_time:.2f} seconds")

# 分割处理后的数据集
# @TODO:了解数据内容与格式
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
# @DATA: X_train.shape = (*,391), X_test.shape = (*,391), y_train.shape = (*,), y_test.shape = (*,)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (38189, 391) (16367, 391) (38189,) (16367,)
# 规一化数据
# @NOTE: 为了提高模型的收敛速度，需要对数据进行规一化处理
# X_train = (X_train - X_train.mean()) / X_train.std()
# X_test = (X_test - X_test.mean()) / X_test.std()



import tensorflow as tf
from models import CustomModel  # 导入自定义模型

# Define parameters
batch_size = 32
len_raw = 50       # Sequence length for LSTM input
input_size_raw = 10 # Feature size for LSTM input
len_a = 100        # Input size for CNN
lstm_units = 64    # Number of LSTM units
cnn_filters = 32   # Number of filters for CNN
cnn_kernel_size = 3
fc_units = 128     # Number of units in fully connected layer
# Instantiate the model
model = CustomModel(lstm_units=lstm_units, cnn_filters=cnn_filters, cnn_kernel_size=cnn_kernel_size, fc_units=fc_units)

# Define inputs for len_raw and len_a parts
x_raw = tf.random.normal((batch_size, len_raw, input_size_raw))  # LSTM input: [batch_size, len_raw, input_size_raw]
x_a = tf.random.normal((batch_size, len_a, 1))  # CNN input: [batch_size, len_a, 1]

# Forward pass
output = model(x_raw, x_a)
print(output.shape)  # Output shape: [batch_size, fc_units]

# Compile the model
model.compile(optimizer='adam', loss='mse')


# 记录训练开始时间
print("Training...")
start_time = time.time()

# 训练决策树分类器
# @IDEA:换成1DCNN
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# 记录训练结束时间
end_time = time.time()

# 计算训练耗时
train_time = end_time - start_time
print(f"Training time: {train_time:.2f} seconds")

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"F1 分数: {f1}")
