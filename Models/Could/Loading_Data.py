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


# Define the directories containing the two linked file sets
input_data = '/kaggle/input/could-dataset2/dataset2/Inputs'
output_data = '/kaggle/input/could-dataset2/dataset2/Outputs'

# Step 1: List and sort the files to ensure consistency
input_files = sorted([f for f in os.listdir(input_data) if f.endswith('.npy')])
output_files = sorted([f for f in os.listdir(output_data) if f.endswith('.npy')])

# Ensure both sets have the same number of files
assert len(input_files) == len(output_files), "The two file sets must have the same number of files."

# Step 2: Shuffle and split into train, validate, and test
X_train, X_temp, y_train, y_temp = train_test_split(input_files, output_files, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verify the split
print(f"Total files: {len(input_files) + len(output_files)}")
print(f"Train files: {len(X_train)+len(y_train)}")
print(f"Validate files: {len(X_val) + len(y_val)}")
print(f"Test files: {len(X_test)+len(y_test)}")
print("Size of train set =", len(X_train))
print("Size of val set =", len(X_val))
print("Size of test set =", len(X_test))

# Which load to train on
load_x_or_y = 'y' 

# Load the data
train_data = X_train
train_labels = y_train
validation_data = X_val
validation_labels = y_val
test_data = X_test
test_labels = y_test

# Compute scaling factors
scaling_factors = compute_global_scaling_factors((os.path.join(input_data, f) for f in input_files), 
                                                 (os.path.join(output_data, f) for f in output_files))
print("Scaling factors:")
for i in scaling_factors:
    print(f'  {i} = {scaling_factors[i]:.3f}')
print()

# Create datasets
datasets = {
    'train': WindTurbineDataset(
        [os.path.join(input_data, f) for f in X_train],
        [os.path.join(output_data, f) for f in y_train],
        scaling_factors = scaling_factors,
        load_x_or_y = load_x_or_y
    ),
    'validation': WindTurbineDataset(
        [os.path.join(input_data, f) for f in X_val],
        [os.path.join(output_data, f) for f in y_val],
        scaling_factors = scaling_factors,
        load_x_or_y = load_x_or_y
    ),
    'test': WindTurbineDataset(
        [os.path.join(input_data, f) for f in X_test],
        [os.path.join(output_data, f) for f in y_test],
        scaling_factors = scaling_factors,
        load_x_or_y = load_x_or_y
    )
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=8, shuffle=True),
    'validation': DataLoader(datasets['validation'], batch_size=8, shuffle=False),
    'test': DataLoader(datasets['test'], batch_size=8, shuffle=False)
}

# Print information about dataset and loaders
print("Shapes of data and loaders")
for settype in dataloaders:
    Turbsim_shape, angles_shape, target_shape = datasets[settype].shapes()
    print(f"  {settype}: Turbsim shape: {list(Turbsim_shape)}, Angles shape: {list(angles_shape)}, Target shape: {list(target_shape)}")
print()

Turbsim, angles, targets = next(iter(DataLoader(datasets['train'], batch_size=8, shuffle=True)))
Turbsim_mean, Turbsim_std = Turbsim.mean(), Turbsim.std()
Turbsim_min, Turbsim_max = Turbsim.min(), Turbsim.max()
angles_mean, angles_std = angles.mean(), angles.std()
angles_min, angles_max = angles.min(), angles.max()
target_mean, target_std = targets.mean(), targets.std()
target_min, target_max = targets.min(), targets.max()

print(f"Statistics for a random sample out of train dataset:")
print(f"  Turbsim - Mean: {Turbsim_mean:.3f}, Std: {Turbsim_std:.3f}, Min: {Turbsim_min:.3f}, Max: {Turbsim_max:.3f}")
print(f"  Angles - Mean: {angles_mean:.3f}, Std: {angles_std:.3f}, Min: {angles_min:.3f}, Max: {angles_max:.3f}")
print(f"  Targets - Mean: {target_mean:.3f}, Std: {target_std:.3f}, Min: {target_min:.3f}, Max: {target_max:.3f}")