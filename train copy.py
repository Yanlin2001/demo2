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

subject_ids = [1,2,3]
# 打印当前目录
print(os.getcwd())
base_path = "src/data" # 数据存放路径
# 初始化 all_X 和 all_y 为空列表，用于存储所有主题的 X 和 y 数据
all_X = []
all_y = []

# 遍历所有主题
for subject_id in subject_ids:
    print(f"Loading data for subject {subject_id}...")
    # 加载数据
    X, y = load_data(subject_id, base_path)
    all_X.append(X)
    all_y.append(y)

# all_X1d, all_x2d, all_y = load_data(subject_id,base_path)
# 定义保存文件的路径
data_file = f"{base_path}/data_subject_{subject_id}.pkl"

# 检查数据是否已存在
if os.path.exists(data_file):
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        all_X1d, all_x2d, all_y1d, all_y2d = pickle.load(f)
else:
    print(f"Data file not found, loading and saving data for subject {subject_id}...")
    all_X1d, all_x2d, all_y1d, all_y2d = load_data(subject_id, base_path)

    # 保存数据
    with open(data_file, 'wb') as f:
        pickle.dump((all_X1d, all_x2d, all_y1d, all_y2d), f)
    print(f"Data saved to {data_file}.")


# @TODO:23个通道的特征只是简单的拼接在一起，没有用到通道之间的关系
# 合并 all_X 和 all_y, vstack是垂直合并, concatenate是水平合并
# @DATA: X.shape = (27600,18), y.shape = (27600,) # (27600,18) -> (1200 , 23, 18) -> (1200 , 23 , 6 , 3)
X1d = np.vstack(all_X1d)
X2d = np.vstack(all_x2d)
# @DATA: y.shape = (27600,)
y1d = np.concatenate(all_y1d)
y2d = np.concatenate(all_y2d)
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
X_resampled, y_resampled = smote.fit_resample(X1d, y1d)
# X2d_resampled, y2d_resampled = smote.fit_resample(X2d, y2d)
# 记录过采样结束时间
oversampling_end_time = time.time()

# 计算过采样耗时
oversampling_time = oversampling_end_time - oversampling_start_time
print(f"Oversampling time: {oversampling_time:.2f} seconds")

# 分割2d数据集
X2d_train, X2d_test, y_train, y_test = train_test_split(X2d, y2d, test_size=0.3, random_state=0)
# @DATA: X2d_train.shape = (*,23,18), X2d_test.shape = (*,23,18), y_train.shape = (*,), y_test.shape = (*,)

# @IDEA:搭建简易双层CNN模型-datashape(900,23,18
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(23, 18, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
# @DATA: X2d_train.shape = (*,23,18), y_train.shape = (*,)
# @DATA: X2d_test.shape = (*,23,18), y_test.shape = (*,)
model.fit(X2d_train, y_train, epochs=10, batch_size=32, validation_data=(X2d_test, y_test))

# 分割处理后的数据集
# @TODO:了解数据内容与格式
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
# @DATA: X_train.shape = (*,391), X_test.shape = (*,391), y_train.shape = (*,), y_test.shape = (*,)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (38189, 391) (16367, 391) (38189,) (16367,)
# 规一化数据
# @NOTE: 为了提高模型的收敛速度，需要对数据进行规一化处理
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# @IDEA:搭建多层1DCNN模型-datashape(20700,18)
from keras.layers import BatchNormalization
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(18, 1)))
model.add(BatchNormalization())  # 添加批归一化层
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
# @DATA: X_train.shape = (*,18), y_train.shape = (*,)
# @DATA: X_test.shape = (*,18), y_test.shape = (*,)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))



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
