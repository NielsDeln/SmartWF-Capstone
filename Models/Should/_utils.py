import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from datetime import date

import rainflow


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

    def __call__(self, val_loss: float, model: nn.Module, save_directory, dataloader) -> None:
        """
        Calls the EarlyStopping class.

        Parameters:
        -----------
        val_loss: float
            The validation loss
        model: torch.nn.Module
            The model to save if an improvement is seen
        """
        if self.patience > -1:
            if val_loss > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_loss
                self.counter = 0
                self.save_best_model(model, save_directory, dataloader)
    
    def save_best_model(self, model: nn.Module, save_directory, dataloader) -> None:
        torch.save(model.state_dict(), f'{save_directory}/model_{date.today()}_{dataloader.dataset.load_axis}.pt')

    def load_best_model(self, model: nn.Module, save_directory, dataloader) -> None:
        model.load_state_dict(torch.load(f'{save_directory}/model_{date.today()}_{dataloader.dataset.load_axis}.pt', weights_only=True))


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
        output = output * dataloader.dataset.label_std + dataloader.dataset.label_mean
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if device == torch.device('cuda:0'):
            torch.cuda.empty_cache()
    return epoch_loss


def train(model: nn.Module, 
          dataloaders: dict[str, DataLoader], 
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
    dataloaders: dict
        The dictionary containing the training and validation dataloaders
    criterion: torch.nn.modules.loss
        The loss function
    optimizer: torch.optim
        The optimizer
    n_epochs: int
        The number of epochs to train for
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
    val_del_history: list = []

    # Initialize the early stopping if specified
    if early_stopping < -1 or not isinstance(early_stopping, int):
        raise ValueError(f'Early stopping must be an integer in the range [-1, {n_epochs})')
    elif early_stopping >= -1 and isinstance(early_stopping, int):
        stop_condition = EarlyStopping(patience=early_stopping, delta=0)

    # Train the model
    for epoch in range(n_epochs):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch+1}/{n_epochs}:\n------------')
        # Train for one epoch and append the loss to the loss history
        train_epoch_loss = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        train_loss_history.append(train_epoch_loss)

        # Evaluate the model on the validation set
        val_epoch_loss, val_del_error = evaluate(model, dataloaders['validation'], criterion, device)
        val_loss_history.append(val_epoch_loss)
        val_del_history.append(val_del_error)

        # Print the loss
        if epoch % print_freq == 0:
            print(f'Training Loss MSE: {train_epoch_loss}\nValidation Loss MSE: {val_epoch_loss}\nvalidation DEL error: {val_del_error}%')

        # Save the model if best loss is seen
        stop_condition(val_epoch_loss, model, save_directory, dataloaders['validation'])
        
        if stop_condition.early_stop:
            print(f'Early stopping at epoch {epoch+1}. \nModel at epoch {epoch+1-early_stopping} will be loaded.')
            break
        
    if not stop_condition.early_stop:
        stop_condition.save_best_model(model, save_directory, dataloaders['validation'])
    stop_condition.load_best_model(model, save_directory, dataloaders['validation'])
    return model, train_loss_history, val_loss_history, val_del_history


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
    model.to(device)
    loss = 0
    del_errors = []
    with torch.no_grad():
        for data, target, time in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output * dataloader.dataset.label_std + dataloader.dataset.label_mean
            loss += criterion(output, target).item()
            del_errors.extend(calculate_del_error(output, target, time, m=10))
    return loss, np.mean(del_errors)


def calculate_del_error(batch_output: torch.Tensor, batch_target: torch.Tensor, batch_time: torch.Tensor, m=10) -> float:
    """
    Calculates the damage equivalent load error.

    Parameters:
    -----------
    output: torch.Tensor
        The batch of outputs from the model
    target: torch.Tensor
        The batch of target data
    time: troch.Tensor
        The time data
    m: int
        The Wohler exponent
        defualt: 10
    
    Returns:
    --------
    del_error: list
        The damage equivalent load error in percentage
    """
    del_errors = []
    
    for item_target, item_output, item_time in zip(batch_target, batch_output, batch_time):
        target_del = calculate_del(item_target, item_time, m)
        output_del = calculate_del(item_output, item_time, m)
        del_error = np.abs(output_del - target_del)/target_del * 100
        del_errors.append(del_error)
        
    return del_errors


def calculate_del(data: torch.Tensor, time: torch.Tensor, m: int, Teq: int=1) -> float:
    """
    Calculates the damage equivalent load.

    Parameters:
    -----------
    data: torch.Tensor
        The data to calculate the DEL for
    time: torch.Tensor
        The time data
    m: int
        The Wohler exponent
    Teq: int
        The equivalent time
        default: 1
    
    Returns:
    --------
    DEL: float
        The damage equivalent load
    """
    data = data.to('cpu').numpy().flatten()
    time = time.to('cpu').numpy().flatten()
    cycles = rainflow.count_cycles(data, nbins=100)

    neq = time[-1]/Teq
    DELi = 0
    for rng, count in cycles:
        DELi += rng**m * count / neq
        
    return DELi**(1/m)


def plot_losses(train_loss_history: Iterable[int], 
                val_loss_history: Iterable[int], 
                val_del_history: Iterable[int],
               ) -> None:
    """
    Plots the training and validation losses.

    Parameters:
    -----------
    train_loss_history: Iterable[int]
        The history of training losses
    val_loss_history: Iterable[int]
        The history of validation losses
    val_del_history: Iterable[int]        
        The history of the validation damage equivalent load error
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_loss_history, label='Training Loss')
    ax[0].plot(val_loss_history, label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(val_del_history, label='Validation DEL')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('DEL Error')
    ax[1].set_title('DEL Mean Absolute Error')
    ax[1].legend()

    fig.show()


def plot_inference(model: nn.Module, 
                   dataloader: DataLoader,
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
            target = target[0].to('cpu')
            output = output[0].to('cpu') * dataloader.dataset.label_std + dataloader.dataset.label_mean
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