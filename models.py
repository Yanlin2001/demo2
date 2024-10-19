# models.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class CustomModel(Model):
    def __init__(self, lstm_units, cnn_filters, cnn_kernel_size, fc_units):
        super(CustomModel, self).__init__()
        
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
        x_raw, x_a = inputs

        # CNN part (for len_a input)
        x_a = self.cnn1(x_a)
        x_a = self.cnn2(x_a)
        x_a = self.flatten(x_a)  # shape: [batch_size, flatten_size]
        

        # Fully connected layers
        x = self.fc1(x_a)
        x = self.fc2(x)

        return x
