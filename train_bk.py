# @NOTE: 原始数据读取在extractFeture.py中
#from data import extractFeture

import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data

subject_id = 1
base_path = "data"
# @NOTE: all_x_shape = (20700,391), all_y_shape = (20700,)
all_X,all_y = load_data(subject_id,base_path)

# 合并 all_X 和 all_y, vstack是垂直合并, concatenate是水平合并
# @NOTE: X.shape = (20700,391), y.shape = (20700,)
X = np.vstack(all_X)
# @NOTE: y.shape = (20700,)
y = np.concatenate(all_y)

# 初始化 SMOTE 实例
smote = SMOTE()

# 应用 SMOTE 过采样
# @NOTE: 过采样平衡数据
# @NOTE: X_resampled.shape = (40894,391), y_resampled.shape = (40894,)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 分割处理后的数据集
# @TODO:了解数据内容与格式
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
# @NOTE: X_train.shape = (28625,391), X_test.shape = (12269,391), y_train.shape = (28625,), y_test.shape = (12269,)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (28625, 391) (12269, 391) (28625,) (12269,)
# 训练决策树分类器
# @IDEA:换成CNN
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"F1 分数: {f1}")