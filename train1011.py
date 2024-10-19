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
# 记录数据预处理开始时间
data_preprocess_start_time = time.time()


subject_id = 1
# 打印当前目录
print(os.getcwd())
base_path = "src/data"
# @DATA: 27600 = 1200 * 23 = 3600 / 3 * 23 （1200个窗口，每个窗口23个通道）
# @DATA: all_x_shape = (27600,392), all_y_shape = (27600,)
all_X,all_y = load_data(subject_id,base_path)
'''

base_path = "data"
subject_ids = range(1, 3)  # subject_id 的范围是 [1, 23]
all_data = []  # 用于存储所有数据的列表

for subject_id in subject_ids:
    print(f"Loading data for subject {subject_id}")
    all_X, all_y = load_data(subject_id, base_path)
    all_data.append((all_X, all_y))
'''

# @TODO:23个通道的特征只是简单的拼接在一起，没有用到通道之间的关系
# 合并 all_X 和 all_y, vstack是垂直合并, concatenate是水平合并
# @DATA: X.shape = (27600,18), y.shape = (27600,) # (27600,18) -> (1200 , 23, 18) -> (1200 , 23 , 6 , 3)
X = np.vstack(all_X)
# @DATA: y.shape = (27600,)
y = np.concatenate(all_y)

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

# 记录训练开始时间
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
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
import time

# 记录训练开始时间
start_time = time.time()

num_classes = 2  # 二分类问题
# 构建1DCNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # num_classes是你的类别数量

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 记录训练结束时间
end_time = time.time()

# 计算训练耗时
train_time = end_time - start_time
print(f"Training time: {train_time:.2f} seconds")

# 对测试集进行预测
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()  # 获取预测的类别

# 评估模型
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred_classes, average='weighted')  # 使用'weighted'平均以考虑类别不平衡
print(f"F1 分数: {f1}")'''