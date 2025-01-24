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

class WindTurbineDataset(Dataset):
    def __init__(self, input_files, target_files, scaling_factors, load_x_or_y, input_transform=None, target_transform=None):
        """
        Dataset class for Wind Turbine data with lazy loading.

        Parameters:
        -----------
        input_files: list of str, List of file paths for input data.
        target_files: list of str, List of file paths for target data.
        input_transform: callable, optional, A function or transformation to apply to the inputs.
        target_transform: callable, optional, A function or transformation to apply to the targets.
        """
        self.input_files = input_files
        self.target_files = target_files
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.scaling_factors = scaling_factors
        self.load_x_or_y = load_x_or_y

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load the input and target data for the given index
        angle_data = self._load_angles(self.target_files[idx])
        target_data = self._load_targets(self.target_files[idx])
        self.target_length = target_data.shape[0]
        Turbsim_data = self._load_inputs(self.input_files[idx])
        
        # Apply global normalization
        Turbsim_data = self._normalize(Turbsim_data, self.scaling_factors["Turbsim_min"], self.scaling_factors["Turbsim_max"], to_range=(0, 1))
        angle_data = self._normalize(angle_data, self.scaling_factors["angle_min"], self.scaling_factors["angle_max"], to_range=(0, 1))

        return Turbsim_data, angle_data, target_data


    def _load_inputs(self, file_path):
        # Input data from shape (1200, 41, 33) to (15000, 41, 33) 
        data = np.load(file_path)  # Load the .npy file
        target_length = self.target_length 
        current_length = data.shape[0]
    
        # Interpolation to the target length (15000 steps)
        time_steps_input = np.arange(current_length)  # Original time steps (0, 1, 2, ..., 1199)
        time_steps_target = np.linspace(0, current_length - 1, target_length)  # New time steps (stretch to 15000)

        # Interpolate over the time axis (first dimension)
        interpolation_function = interpolate.interp1d(time_steps_input, data, kind='linear', axis=0)
        inputs_stretched = interpolation_function(time_steps_target)  # Result will have shape [15000, 41, 33]
        return torch.tensor(inputs_stretched).float()  # Convert to PyTorch tensor
        

    def _load_targets(self, file_path):
        data = np.load(file_path)  # Load the .npy file
        if load_x_or_y == 'x' or load_x_or_y == 'X':
            data = data[:, 1]          # Keep only the RootMxb1 column
        elif load_x_or_y == 'y' or load_x_or_y == 'Y':
            data = data[:, 2]          # Keep only the RootMyb1 column
        return torch.tensor(data)  # Convert to PyTorch tensor
    
    def _load_angles(self, file_path):
        data = np.load(file_path)  # Load the .npy file
        data = data[:, 3]          # Keep only the Angles column
        return torch.tensor(data)  # Convert to PyTorch tensor

    @staticmethod
    def _normalize(data, min_val, max_val, to_range=(0, 1)):
        range_min, range_max = to_range
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized * (range_max - range_min) + range_min

    def shapes(self):
        sample_Turbsim, sample_angle, sample_target = self[0]
        return sample_Turbsim.shape, sample_angle.shape, sample_target.shape

def _denormalize(normalized_data, min_val, max_val, from_range=(-1, 1)):
    range_min, range_max = from_range
    original = (normalized_data - range_min) / (range_max - range_min + 1e-8)
    return original * (max_val - min_val) + min_val

print("Defining Dataloader completed.")

def train_one_epoch(model: nn.Module, 
                    dataset: Dataset, 
                    criterion: nn.modules.loss, 
                    optimizer: optim, 
                    device: torch.device=torch.device('cpu')
                    ) -> float:
    """
    Trains the model for one epoch.

    Parameters:
    -----------
    model: torch.nn.Module, The model to train
    dataset: Dataset, The training dataset
    criterion: torch.nn.modules.loss, The loss function
    optimizer: torch.optim, The optimizer
    device: torch.device, The device to use for training

    Returns:
    --------
    epoch_loss: float, The loss for the epoch
    """
    model.train()
    epoch_loss = 0
    for Turbsim, angle, target in dataset:
        Turbsim, angle, target = Turbsim.to(device), angle.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Turbsim, angle)
        output = _denormalize(output, scaling_factors['target_min'], scaling_factors['target_max'], from_range=(-1, 1)) #rescale predictions
        target = target.unsqueeze(-1)  # Add a dimension at the end to make the shape (batch_size, seq_len, 1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if device != torch.device('cpu'):
            torch.cuda.empty_cache()
    return epoch_loss


def train(model: nn.Module, 
          dataloader: DataLoader, 
          criterion: nn.modules.loss, 
          optimizer: optim, 
          n_epochs: int,
          device: torch.device=torch.device('cpu'),
          early_stopping_value: int=-1,
          print_freq: int=10
          ) -> tuple[nn.Module, list, list]:
    """
    Parameters:
    -----------
    model: torch.nn.Module, The model to train
    dataloader: torch.utils.data.DataLoader, The dataloader containing the training and validation data
    criterion: torch.nn.modules.loss, The loss function
    optimizer: torch.optim, The optimizer
    n_epochs: int, The number of epochs to train for
    device: torch.device, The device to use for training
    early_stopping: int, Whether to use early stopping or not
        Default: -1 = No early stopping applied
        Other values: The number of epochs to wait to see an improvement in the loss
    print_freq: int,The frequency to print the loss
        Default: 10
    
    Returns:
    --------
    model: torch.nn.Module, The trained model
    train_loss_history: list, The history of training losses
    val_loss_history: list, The history of validation losses
    """
    model.to(device)
    model.train()

    best_loss = float('inf')
    train_loss_history: list = []
    val_loss_history: list = []
    print("Training started")

    # Train the model
    for epoch in range(n_epochs):
        # Train for one epoch and append the loss to the loss history
        
        train_epoch_loss = train_one_epoch(model, dataloader['train'], criterion, optimizer, device)
        train_loss_history.append(train_epoch_loss)

        # Evaluate the model on the validation set
        val_epoch_loss = evaluate(model, dataloader['validation'], criterion, device)
        val_loss_history.append(val_epoch_loss)

        # Print the loss
        print(f'Epoch: {epoch+1}/{n_epochs} Training Loss = {train_epoch_loss:.3f}, Validation Loss = {val_epoch_loss:.3f}')

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            
    return model, train_loss_history, val_loss_history


def evaluate(model: nn.Module, 
             dataset: Dataset, 
             criterion: nn.modules.loss, 
             device: torch.device=torch.device('cpu')
             ) -> float:
    """
    Evaluates the model.

    Parameters:
    -----------
    model: torch.nn.Module, The model to evaluate
    dataloader: torch.utils.data.DataLoader, The dataloader containing the evaluation data
    criterion: torch.nn.modules.loss, The loss function
    device: torch.device, The device to use for evaluation
    
    Returns:
    --------
    loss: float, The loss for the evaluation
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for Turbsim, angles, target in dataset:
            Turbsim, angles, target = Turbsim.to(device), angles.to(device), target.to(device)
            output = model(Turbsim, angles)
            output = _denormalize(output, scaling_factors['target_min'], scaling_factors['target_max'], from_range=(-1, 1)) # Rescale predictions
            target = target.unsqueeze(-1)  # Add a dimension at the end to make the shape (batch_size, seq_len, 1)
            loss += criterion(output, target).item()
    return loss

print("Defining Train and Evaluation functions completed.")

def plot_losses(train_loss_history: Iterable, val_loss_history: Iterable) -> None:
    """
    Plots the training and validation losses.

    Parameters:
    -----------
    train_loss_history: Iterable, The history of training losses
    val_loss_history: Iterable, The history of validation losses
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_loss_history, label='Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')
    ax[0].legend()

    ax[1].plot(val_loss_history, label='Validation Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Validation Loss')
    ax[1].legend()

    fig.show()


def plot_predicted_vs_true(model, input_file, device='cuda' if torch.cuda.is_available() else 'cpu', only_plot_pred="no"):
    """
    Predicts and plots the predicted vs. true load values for a given simulation.

    Args:
        model: Trained PyTorch model for prediction.
        input_file (str): Path to the input simulation .npy file.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()
    
    # Derive the target file path from the input file name
    target_file = input_file.replace('_in_', '_out_')
    target_file = target_file.replace('Input', 'Output')

    # Load input and target data using the WindTurbineDataset
    dataset = WindTurbineDataset([input_file], [target_file], scaling_factors = scaling_factors, load_x_or_y=load_x_or_y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for Turbsim, angles, true_outputs in dataloader:
            Turbsim = Turbsim.to(device).float()
            angles = angles.to(device)
            true_outputs = true_outputs.squeeze(0).cpu().numpy()  # Shape: (seq_len, 1)
            true_outputs = true_outputs[:, 25:]  # Disregard first second

            
            # Model prediction
            predictions  = model(Turbsim, angles).squeeze(0).cpu().numpy()  # Shape: (seq_len, 1)
            predictions = _denormalize(predictions, scaling_factors['target_min'], scaling_factors['target_max'], from_range=(-1, 1))
            predictions = predictions[:, 25:]  # Disregard first second
            
            # Create time steps
            time_steps = np.arange(predictions.shape[0])/25

            # Plotting
            plt.figure(figsize=(12, 6))
            if only_plot_pred == "no":
                plt.plot(time_steps, true_outputs, label='True RootMxb1', color='green', linestyle='--')
            plt.plot(time_steps, predictions, label='Predicted RootMxb1', color='red')
            
            plt.xlabel("Time (seconds)")
            plt.ylabel("Load Value")
            plt.title("Predicted vs. True Load Values")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            break  # Only one batch since batch_size=1

def plot_predictions(model, dataloader, num_plots=1, num_parts=1, device='cuda' if torch.cuda.is_available() else 'cpu', only_plot_pred="no"):
    """
    Plots randomly selected predicted vs. true values for a given dataloader using a pretrained model.

    Args:
        model: Pretrained PyTorch model.
        dataloader: DataLoader containing the data to evaluate.
        num_plots (int): Number of plots to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').
        only_plot_pred (str): If "yes", only plot the predictions.
    """
    model.to(device)
    model.eval()

    # Gather all data from the dataloader for random sampling
    all_predictions, all_true_values = [], []
    with torch.no_grad():
        for batch in dataloader:
            Turbsim, angles, true_outputs = batch
            Turbsim = Turbsim.to(device).float()
            angles = angles.to(device)
            true_outputs = true_outputs.cpu().numpy()  # Convert to NumPy
            true_outputs = true_outputs[:, 25:]  # Disregard first second
            seq_len = len(true_outputs[0])

            # Get model predictions
            predictions = model(Turbsim, angles)  # Convert to NumPy
            predictions = _denormalize(predictions, scaling_factors['target_min'], scaling_factors['target_max'], from_range=(-1, 1)).cpu().numpy()
            predictions = predictions[:, 25:]  # Disregard first second
            
            # Collect predictions and true values
            all_predictions.extend(predictions)  # Shape: (batch_size, seq_len, 1)
            all_true_values.extend(true_outputs)  # Shape: (batch_size, seq_len, 1)

    # Randomly select `num_plots` indices
    num_samples = len(all_predictions)
    if num_samples < num_plots:
        print(f"Warning: Only {num_samples} samples available, but {num_plots} plots requested.")
        num_plots = num_samples

    selected_indices = random.sample(range(num_samples), num_plots)

    # Generate plots
    for plot_idx, idx in enumerate(selected_indices):
        prediction = all_predictions[idx]   # Shape: (seq_len,)
        true_output = all_true_values[idx]   # Shape: (seq_len,)
        time_steps = np.arange(seq_len)/25  # Shape: (seq_len,)

        # Determine the number of time steps per part
        part_len = seq_len // num_parts  # Drop leftover steps
    
        # Split and plot each part
        for part_idx in range(num_parts):
            start_idx = part_idx * part_len
            end_idx = start_idx + part_len
    
            plt.figure(figsize=(12, 6))
            if only_plot_pred == "no":
                plt.plot(
                    time_steps[start_idx:end_idx],
                    true_output[start_idx:end_idx],
                    label='True Values',
                    color='green',
                    linestyle='--'
                )
            plt.plot(
                time_steps[start_idx:end_idx],
                prediction[start_idx:end_idx],
                label='Predicted Values',
                color='red'
            )
    
            plt.xlabel("Time (seconds)")
            plt.ylabel("Load Value")
            plt.title(f"Prediction vs. True Values (Plot {plot_idx + 1}, Part {part_idx + 1}/{num_parts})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_average_field_over_time(dataset):
    """
    Plots the average of a 41x33 field for each timestep across all tensors in a PyTorch dataset.

    Parameters:
    - dataset: A PyTorch dataset containing tensors of shape [1500, 41, 33].
    """
    all_averages = []  # To store average values for each timestep across all tensors

    # Iterate through the dataset
    for item in dataset:
        # Ensure the item is a PyTorch tensor and has the expected shape
        if isinstance(item, torch.Tensor) and item.dim() == 3 and item.shape[1:] == (41, 33):
            # Compute the average for each timestep
            averages = item.mean(dim=(1, 2))  # Average across dimensions 1 (41) and 2 (33)
            all_averages.append(averages)

    # Concatenate averages from all tensors
    all_averages = torch.cat(all_averages).cpu().numpy()

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(all_averages, label="Average Field Value")
    plt.xlabel("Timestep")
    plt.ylabel("Average Value")
    plt.title("Average of 41x33 Field Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

print("Defining Plotting completed.")

def compute_global_scaling_factors(input_files, target_files):
    """
    Compute global min and max values for input and target data by iterating over all files.
    
    Parameters:
    -----------
    input_files: list of str, List of file paths for input data.
    target_files: list of str, List of file paths for target data.
    
    Returns:
    --------
    dict
        Dictionary containing global min and max values for inputs and targets.
    """
    # Initialiseer min en max waarden
    input_min = float('inf')
    input_max = float('-inf')
    target_min = float('inf')
    target_max = float('-inf')
    angle_min = float('inf')
    angle_max = float('-inf')

    if load_x_or_y == 'x' or load_x_or_y == 'X':
        target_column = 1         # Keep only the RootMxb1 column
    elif load_x_or_y == 'y' or load_x_or_y == 'Y':
        target_column = 2         # Keep only the RootMxb1 column
    
    # Loop door alle bestanden
    for input_file, target_file in zip(input_files, target_files):
        # Laad input data en bereken min/max
        input_data = np.load(input_file)  
        input_min = min(input_min, input_data.min())
        input_max = max(input_max, input_data.max())
        
        # Laad target data en bereken min/max
        target_data = np.load(target_file)  # Bijvoorbeeld van vorm (tijd, kolommen)
        target_values = target_data[:, target_column]  # (RootMyb1)
        target_min = min(target_min, target_values.min())
        target_max = max(target_max, target_values.max())
        angle_values = target_data[:, 3]  # Angles
        angle_min = min(angle_min, angle_values.min())
        angle_max = max(angle_max, angle_values.max())

    # Geef de globale min/max waarden terug
    return {
        'Turbsim_min': input_min,
        'Turbsim_max': input_max,
        'angle_min': angle_min,
        'angle_max': angle_max,        
        'target_min': target_min,
        'target_max': target_max
    }

print("Defining other functions completed.")
