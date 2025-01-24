import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import interpolate

from tqdm import tqdm
import matplotlib.pyplot as plt
from collections.abc import Iterable
from datetime import date
import os
import random
import math

class CNNFeatureExtractor(nn.Module):
    def __init__(self, kernel_size, out_channels=1, stride=1):
        super(CNNFeatureExtractor, self).__init__()

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=1,  # Single-channel input
            out_channels=out_channels,  # Number of output channels
            kernel_size=(kernel_size, kernel_size),  # Kernel size
            stride=(stride, stride),  # Stride
            padding=0  # No padding
        )
        
        # Define a flattening layer
        self.flatten = nn.Flatten()  # Flattens the output to a 1D vector

    def forward(self, x):
        batch_size, seq_len, height, width = x.size()  # Input: [Batch size, sequence length, Height, Width]
        x = x.view(batch_size * seq_len, 1, height, width)  # Reshape to [B * seq_len, 1, Height, Width]
        x = self.conv(x)  # Apply convolution: [B * seq_len, out_channels, new_height, new_width]
        x = x.view(batch_size * seq_len, -1)  # Flatten to [B * seq_len, new_height * new_width]
        x = x.view(batch_size, seq_len, -1)  # Reshape back to [Batch size, sequence length, feature_dim]
        return x


class RNNModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM layer
        x = self.dropout(x)  # Apply dropout to the LSTM outputs
        x = self.fc(x)  # Fully connected layer for predictions
        return x



# Combine CNN and RNN into a single model
class WindTurbineLoadPredictor(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, kernel_size, out_channels, stride, input_height, input_width, dropout_prob, angle_dim=1):
        super(WindTurbineLoadPredictor, self).__init__()
        self.cnn = CNNFeatureExtractor(kernel_size, out_channels, stride)

        # Dynamically calculate the CNN output feature dimension
        dummy_input = torch.zeros(1, 1, input_height, input_width)  # Example input: [Batch size, Channels, Height, Width]
        dummy_output = self.cnn.conv(dummy_input)  # Apply only the CNN convolution
        cnn_feature_dim = dummy_output.numel()  # Total elements in the CNN output per sample

        # Initialize the RNN with the calculated feature_dim + angle_dim
        self.rnn = RNNModel(cnn_feature_dim + angle_dim, hidden_dim, output_dim, num_layers, dropout_prob)

    def forward(self, x, x1):
        cnn_output = self.cnn(x)  # Extract spatial features
        seq_len = cnn_output.size(1)

        # Expand and concatenate x1 (scalar feature)
        x1 = x1.unsqueeze(-1).expand(-1, seq_len, -1)  # Shape: [Batch, seq_len, 1]
        combined_features = torch.cat((cnn_output, x1), dim=-1)  # Concatenate along the feature axis

        x = self.rnn(combined_features)  # Model temporal dependencies
        return x


print("Defining model completed.")