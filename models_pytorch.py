import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, lstm_units, cnn_filters, cnn_kernel_size, fc_units):
        super(CustomModel, self).__init__()
        
        # LSTM 部分用于处理 len_raw 输入
        # LSTM 的输入尺寸为 [batch_size, len_raw, lstm_units]，输出 hidden_size 为 lstm_units
        self.lstm = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, batch_first=True, num_layers=2)

        # CNN 部分用于处理 len_a 输入
        # 输入尺寸应为 [batch_size, 1, len_a]，这里通道数为1
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.cnn2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.flatten = nn.Flatten()
        
        # 全连接层
        # 输入为 LSTM 和 CNN 输出的拼接结果，大小为 lstm_units + cnn_filters
        self.fc1 = nn.Linear(in_features=lstm_units + cnn_filters, out_features=fc_units)
        self.fc2 = nn.Linear(in_features=fc_units, out_features=1)
        
        # 输出层激活函数为 Sigmoid，用于二分类问题
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x_raw, x_a):
        # LSTM 部分 (处理 len_raw 输入)
        x_raw, _ = self.lstm(x_raw)  # 输出形状: [batch_size, len_raw, lstm_units]
        x_raw = x_raw[:, -1, :]  # 只使用最后一个时间步的输出

        # CNN 部分 (处理 len_a 输入)
        x_a = self.relu(self.cnn1(x_a))  # 通过第一个卷积层并应用 ReLU 激活函数
        x_a = self.relu(self.cnn2(x_a))  # 通过第二个卷积层并应用 ReLU 激活函数
        x_a = self.flatten(x_a)  # 展平 CNN 输出，输出形状为: [batch_size, flatten_size]
        
        # 将 LSTM 和 CNN 的输出拼接在一起
        combined = torch.cat((x_raw, x_a), dim=1)

        # 全连接层
        x = self.relu(self.fc1(combined))  # 通过第一个全连接层并应用 ReLU 激活函数
        x = self.sigmoid(self.fc2(x))  # 通过第二个全连接层并应用 Sigmoid 激活函数

        return x
