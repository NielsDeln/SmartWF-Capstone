import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from datetime import date


class EarlyStopping:
    def __init__(self, patience: int=5, delta: float=0) -> None:
        """
        Initializes the EarlyStopping class.

        Parameters:
        -----------
        patience: int
            The number of epochs to wait to see an improvement in the loss
            Default: 5
        delta: float
            Minimum change in the monitored quantity to qualify as an improvement
            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.best_score: float = float('inf')
        self.early_stop: bool = False
        self.counter: int = 0

    def __call__(self, val_loss: float, model: nn.Module, save_directory:str) -> None:
        """
        Calls the EarlyStopping class.

        Parameters:
        -----------
        val_loss: float
            The validation loss
        model: torch.nn.Module
            The model to save if an improvement is seen
        save_directory: str
            The directory to save the model
        """
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), f'{save_directory}/model_{date.today()}_val_loss{val_loss:.3f}.pt')

    def load_best_model(self, model: nn.Module, val_loss, save_directory) -> None:
        model.load_state_dict(torch.load(f'{save_directory}/model_{date.today()}_val_loss{val_loss:.3f}.pt', weights_only=True))


def train_one_epoch(model: nn.Module, 
                    dataloader: DataLoader, 
                    criterion: nn.modules.loss, 
                    optimizer: optim, 
                    device: torch.device=torch.device('cpu')
                    ) -> float:
    """
    Trains the model for one epoch.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to train
    dataloader: torch.nn.utils.DataLoader
        The training dataloader
    criterion: torch.nn.modules.loss
        The loss function
    optimizer: torch.optim
        The optimizer
    device: torch.device
        The device to use for training

    Returns:
    --------
    epoch_loss: float
        The loss for the epoch
    """
    model.train()
    epoch_loss = 0
    for data, target, _ in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if device == torch.device('cuda:0'):
            torch.cuda.empty_cache()
    return epoch_loss


def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader,
          criterion: nn.modules.loss, 
          optimizer: optim, 
          n_epochs: int,
          save_directory: str,
          device: torch.device=torch.device('cpu'),
          early_stopping: int=-1,
          print_freq: int=10
          ) -> tuple[nn.Module, list, list]:
    """
    Trains the model.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to train
    train_dataloader: torch.utils.data.DataLoader
        The dataloader containing the training data
    val_dataloader: torch.utils.data.DataLoader
        The dataloader containing the validation data
    criterion: torch.nn.modules.loss
        The loss function
    optimizer: torch.optim
        The optimizer
    n_epochs: int
        The number of epochs to train for
    save_directory: str
        The directory to save the model
    device: torch.device
        The device to use for training
    early_stopping: int
        Whether to use early stopping or not
        Default: -1 = No early stopping applied
        Other values: The number of epochs to wait to see an improvement in the loss
    print_freq: int
        The frequency to print the loss
        Default: 10
    
    Returns:
    --------
    model: torch.nn.Module
        The trained model
    train_loss_history: list
        The history of training losses
    val_loss_history: list
        The history of validation losses
    """
    model.to(device)

    train_loss_history: list = []
    val_loss_history: list = []

    # Initialize the early stopping if specified
    if early_stopping >= 0 and isinstance(early_stopping, int):
        stop_condition = EarlyStopping(patience=early_stopping, delta=0)

    # Train the model
    for epoch in range(n_epochs):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch+1}/{n_epochs}:\n------------')
        # Train for one epoch and append the loss to the loss history
        train_epoch_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        train_loss_history.append(train_epoch_loss)

        # Evaluate the model on the validation set
        val_epoch_loss = evaluate(model, val_dataloader, criterion, device)
        val_loss_history.append(val_epoch_loss)

        # Print the loss
        if epoch % print_freq == 0:
            print(f'Training Loss {train_epoch_loss}, Validation Loss {val_epoch_loss}')

        # Save the model if best loss is seen
        if early_stopping >= -1 and isinstance(early_stopping, int):
            stop_condition(val_epoch_loss, model, save_directory)
        elif early_stopping <= -1 or not isinstance(early_stopping, int):
            raise ValueError(f'Early stopping must be an integer in the range [-1, {n_epochs})')
        
        if stop_condition.early_stop:
            print(f'Early stopping at epoch {epoch+1}. \nModel at epoch {epoch} will be loaded.')
            break
    
    stop_condition.load_best_model(model, np.min(val_loss_history), save_directory)
    return model, train_loss_history, val_loss_history


def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.modules.loss, 
             device: torch.device=torch.device('cpu')
             ) -> float:
    """
    Evaluates the model.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to evaluate
    dataloader: torch.utils.data.DataLoader
        The dataloader containing the evaluation data
    criterion: torch.nn.modules.loss
        The loss function
    device: torch.device
        The device to use for evaluation
    
    Returns:
    --------
    loss: float
        The loss for the evaluation
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
    return loss


def plot_losses(train_loss_history: Iterable, val_loss_history: Iterable) -> None:
    """
    Plots the training and validation losses.

    Parameters:
    -----------
    train_loss_history: Iterable
        The history of training losses
    val_loss_history: Iterable
        The history of validation losses
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


def plot_inference(model: nn.Module, 
                   dataloader: DataLoader, 
                   labels_mean: float, 
                   labels_stdev: float,
                   num_inf: int=1,
                   device: torch.device=torch.device('cpu')
                   ) -> None:
    """
    Plots the inference of the model.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to evaluate
    dataloader: torch.utils.data.DataLoader
        The dataloader to plot inference on
    labels_mean: float
        The mean of the training labels used to denormalize the data
    labels_stdev: float
        The standard deviation of the training labels used to denormalize the data
    num_inf: int
        The number of inferences to plot
        default: 1
    device: torch.device
        The device to use for evaluation
    """
    if num_inf < 1 or not isinstance(num_inf, int):
        raise ValueError('Number of inferences must be an integer greater than 0') 

    model.eval()
    model.to(device)
    count = 0
    with torch.no_grad():
        for data, target, time in dataloader:
            data = data.to(device)
            output = model(data)
            
            # Take only the first item in the batch and remove normalization
            time = time[0].to('cpu')
            target = target[0].to('cpu') * labels_stdev + labels_mean
            output = output[0].to('cpu') * labels_stdev + labels_mean
            plt.figure()
            plt.plot(time, target, label='True')
            plt.plot(time, output, label='Predicted')
            plt.xlabel('Time (s)')
            plt.ylabel('Bending moment (kN-m)')
            plt.legend()
            plt.show()

            # Break loop if number of inferences is reached
            count += 1
            if count >= num_inf:
                break