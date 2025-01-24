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

# Create model
input_height = 11            # Height of the 2D input field
input_width = 14            # Width of the 2D input field
hidden_dim = 128            # Hidden dimension of LSTM
output_dim = 1              # Output dimension (load value)
num_layers = 3              # Number of LSTM layers
dropout_prob = 0.2          # Dropout_prob of LSTM
kernel_size = 3             # Kernel size for CNN
out_channels = 1            # Output channels of CNN
stride = 1                  # Stride of CNN

learning_rate = 0.0005      # Learning rate optimizer
num_epochs = 20             # Amount of epochs

model = WindTurbineLoadPredictor(hidden_dim=hidden_dim, 
                                 output_dim=output_dim, 
                                 num_layers=num_layers,
                                 kernel_size=kernel_size,
                                 out_channels=out_channels,
                                 stride=stride,
                                 input_height=input_height,
                                 input_width=input_width,
                                 dropout_prob=dropout_prob)


# Clear cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device =", device)

# Train the model
trained_model, train_losses, validation_losses = train(model, 
                                                       dataloaders, 
                                                       loss_fn, 
                                                       optimizer, 
                                                       num_epochs, 
                                                       device=device, 
                                                       early_stopping_value=-1, 
                                                       print_freq=10)

# Save the trained model
model_save_path = '/kaggle/working/wind_turbine_model.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot the training and validation losses
plot_losses(train_losses, validation_losses)

# Evaluate the model and plot inference
val_loss = evaluate(trained_model, dataloaders['validation'], loss_fn, device=device)
plot_predictions(model=trained_model, dataloader=dataloaders['validation'], num_plots=2)

