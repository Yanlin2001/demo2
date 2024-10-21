# models.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class CustomModel(Model):
    def __init__(self, lstm_units, cnn_filters, cnn_kernel_size, fc_units):
        super(CustomModel, self).__init__()
        
        # LSTM part for len_raw input
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units, return_sequences=False)  # only output the last time step
        
        # CNN part for len_a input
        self.cnn1 = layers.Conv1D(cnn_filters, cnn_kernel_size, activation='relu')
        self.cnn2 = layers.Conv1D(cnn_filters, cnn_kernel_size, activation='relu')
        self.flatten = layers.Flatten()
        
        # Fully connected layers
        self.fc1 = layers.Dense(fc_units, activation='relu')
        # binary classification output
        self.fc2 = layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        # LSTM part (for len_raw input)
        x_raw, x_a, x_w = inputs
        x_raw = self.lstm1(x_raw)
        x_raw = self.lstm2(x_raw)  # shape: [batch_size, lstm_units]

        # CNN part (for len_a input)
        x_a = self.cnn1(x_a)
        x_a = self.cnn2(x_a)
        x_a = self.flatten(x_a)  # shape: [batch_size, flatten_size]
        
        # Concatenate LSTM and CNN outputs
        combined = tf.concat([x_raw, x_a, x_w], axis=1)

        # Fully connected layers
        x = self.fc1(combined)
        x = self.fc2(x)

        return x
