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

subject_ids = [1, 3, 5]
# 打印当前目录
print(os.getcwd())
base_path = "data" # 数据存放路径
# 初始化 all_X 和 all_y 为空列表，用于存储所有主题的 X 和 y 数据
all_X = []
all_y = []

len_raw = None
len_a = None
len_w = None
len_per_subject = None
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

# 单独设置测试集（不加载和保存）
test_X = []
test_y = []
test_subject_ids = [2]
for subject_id in test_subject_ids:
    print(f"Loading data for test subject {subject_id}...")
    X_test, y_test,len_info = load_data(subject_id, base_path)
    len_raw = len_info[0]
    len_a = len_info[1]
    len_w = len_info[2]
    test_X.append(X_test)
    test_y.append(y_test)

test_X = np.vstack(test_X)
test_y = np.concatenate(test_y)
test_X = test_X.reshape(-1, len_raw + len_a + len_w)
test_y = test_y.reshape(-1)




# @TODO:23个通道的特征只是简单的拼接在一起，没有用到通道之间的关系
# 合并 all_X 和 all_y, vstack是垂直合并, concatenate是水平合并
# @DATA: X.shape = (27600,18), y.shape = (27600,) # (27600,18) -> (1200 , 23, 18) -> (1200 , 23 , 6 , 3)


# 输出每个元素的形状
print("all_X shapes:", [len(X) for X in all_X])
print("all_y shapes:", [len(y) for y in all_y])

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
# @NOTE: SMOTE: Synthetic Minority Over-sampling Technique
# 1. 找到标签 y 中的少数类（例如 y=1）
# 2. 在少数类样本的特征空间中，通过现有样本之间的插值生成新的样本
# 3. 生成的新样本与少数类样本相似，从而增加少数类样本数量，平衡类别分布

# 应用 SMOTE 过采样
# @NOTE: 过采样平衡数据
# @NOTE: 过采样：对少数类样本进行插值，增加样本数量，使得少数类样本与多数类样本数量接近相等（不超原数据两倍）
# @NOTE: y = 1/0

#X_resampled, y_resampled = smote.fit_resample(X, y)
#test_X_resampled, test_y_resampled = smote.fit_resample(test_X, test_y)
print(X_resampled.shape, y_resampled.shape) # (38189, 391) (38189,)
# X2d_resampled, y2d_resampled = smote.fit_resample(X2d, y2d)
# 记录过采样结束时间
print("Number of positives in test_y:", sum(test_y == 1))
oversampling_end_time = time.time()

# 计算过采样耗时
oversampling_time = oversampling_end_time - oversampling_start_time
print(f"Oversampling time: {oversampling_time:.2f} seconds")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from models import CustomModel  # 导入模型定义

# 假设你有数据 (123004, 1427)，并已按照前面步骤拆分好
#len_raw = 1024
#len_a = 385
#len_w = 18


# 检查 GPU 是否可用
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", physical_devices)

if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available. Running on CPU.")



data = X_resampled
y = y_resampled

# 最大-最小规范化

#data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
#test_X = (test_X - np.min(test_X, axis=0)) / (np.max(test_X, axis=0) - np.min(test_X, axis=0))
# data 与 test_X 一起规范化，以确保它们在相同的范围内
min_all = np.min(np.vstack((data, test_X)), axis=0)
max_all = np.max(np.vstack((data, test_X)), axis=0)
data = (data - min_all) / (max_all - min_all)
test_X = (test_X - min_all) / (max_all - min_all)

# 分割数据 (与之前一致)
data_raw = data[:, :len_raw]
data_a = data[:, len_raw:len_raw + len_a]
test_data_raw = test_X[:, :len_raw]
test_data_a = test_X[:, len_raw:len_raw + len_a]

# Reshape for LSTM and CNN inputs
data_raw_lstm_input = np.expand_dims(data_raw, axis=-1)  # (123004, 1024, 1)
data_a_cnn_input = np.reshape(data_a, (data_a.shape[0], len_a // 5, 5))  # (123004, 385, 5)
test_data_raw_lstm_input = np.expand_dims(test_data_raw, axis=-1)  # (123004, 1024, 1)
test_data_a_cnn_input = np.reshape(test_data_a, (test_data_a.shape[0], len_a // 5, 5))  # (123004, 385, 5)

# Step 1: 划分训练集和验证集
x_raw_train, x_raw_val, x_a_train, x_a_val, y_train, y_val = train_test_split(
    data_raw_lstm_input, data_a_cnn_input, y, test_size=0.3, random_state=42
)

# Step 2: 定义模型
lstm_units = 64
cnn_filters = 32
cnn_kernel_size = 3
fc_units = 128
# Instantiate the custom model
model = CustomModel(lstm_units=lstm_units, cnn_filters=cnn_filters, cnn_kernel_size=cnn_kernel_size, fc_units=fc_units)

# Step 3: 编译模型，使用 binary_crossentropy 作为损失函数，accuracy 作为评估指标
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
# Step 4: 训练模型并加入验证集
history = model.fit(
    x=[x_raw_train, x_a_train],
    y=y_train,
    validation_data=([x_raw_val, x_a_val], y_val),  # 验证数据
    epochs=100,  # 根据需求调整
    batch_size=32  # 根据需求调整
)

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# 测试集上的预测
test_pred = model.predict([test_data_raw_lstm_input, test_data_a_cnn_input])
test_pred = np.round(test_pred).flatten()  # 四舍五入
val_pred = model.predict([x_raw_val, x_a_val])
val_pred = np.round(val_pred).flatten()  # 四舍五入

# 计算准确度（Accuracy）
test_accuracy = accuracy_score(test_y, test_pred)
val_accuracy = accuracy_score(y_val, val_pred)

# 计算灵敏度（Sensitivity，使用召回率 recall_score）
test_sensitivity = recall_score(test_y, test_pred)
val_sensitivity = recall_score(y_val, val_pred)

false_negatives = np.where((test_y == 1) & (test_pred == 0))

# 输出标签为正但预测为负的索引
print("False Negatives indices in test set:", false_negatives[0])

# 输出 False Negatives 的实际标签和预测值
print("True labels (test_y):", test_y[false_negatives])
print("Predicted labels (test_pred):", test_pred[false_negatives])

# 计算特异度（Specificity，需要从混淆矩阵中计算）
tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
vtn, vfp, vfn, vtp = confusion_matrix(y_val, val_pred).ravel()
test_specificity = tn / (tn + fp)
val_specificity = vtn / (vtn + vfp)

# 计算F1值
test_f1 = f1_score(test_y, test_pred)
val_f1 = f1_score(y_val, val_pred)

# 输出结果
print(f'Accuracy: {test_accuracy}')
print(f'Sensitivity: {test_sensitivity}')
print(f'Specificity: {test_specificity}')
print(f'F1 Score: {test_f1}')

print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Sensitivity: {val_sensitivity}')
print(f'Validation Specificity: {val_specificity}')
print(f'Validation F1 Score: {val_f1}')



# Step 5: 打印训练历史或可视化
import matplotlib.pyplot as plt

# 绘制训练和验证的损失曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练和验证的准确率曲线
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()