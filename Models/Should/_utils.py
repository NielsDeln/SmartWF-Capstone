import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
from collections.abc import Iterable
from datetime import date


def train_one_epoch(model: nn.Module, 
                    dataset: Dataset, 
                    criterion: nn.modules.loss, 
                    optimizer: optim, 
                    device: str='cpu'
                    ) -> float:
    """
    Trains the model for one epoch.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to train
    dataloader: torch.utils.data.DataLoader
        The dataloader containing the training data
    criterion: torch.nn.modules.loss
        The loss function
    optimizer: torch.optim
        The optimizer
    device: str
        The device to use for training

    Returns:
    --------
    epoch_loss: float
        The loss for the epoch
    """
    model.train()
    epoch_loss = 0
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if device != 'cpu':
            torch.cuda.empty_cache()
    return epoch_loss


def train(model: nn.Module, 
          dataloader: DataLoader, 
          criterion: nn.modules.loss, 
          optimizer: optim, 
          n_epochs: int,
          device: str='cpu',
          early_stopping: int=-1,
          print_freq: int=10
          ) -> tuple[nn.Module, list, list]:
    """
    Trains the model.

    Parameters:
    -----------
    model: torch.nn.Module
        The model to train
    dataloader: torch.utils.data.DataLoader
        The dataloader containing the training data
    criterion: torch.nn.modules.loss
        The loss function
    optimizer: torch.optim
        The optimizer
    n_epochs: int
        The number of epochs to train for
    device: str
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
    best_loss: float
        The best loss during training
    """
    model.to(device)

    best_loss = float('inf')
    train_loss_history: list = []
    val_loss_history: list = []

    # Train the model
    for epoch in tqdm(n_epochs):
        # Train for one epoch and append the loss to the loss history
        train_epoch_loss = train_one_epoch(model, dataloader['train'], criterion, optimizer, device)
        train_loss_history.append(train_epoch_loss)

        # Evaluate the model on the validation set
        val_epoch_loss = evaluate(model, dataloader['val'], criterion, device)
        val_loss_history.append(val_epoch_loss)

        # Print the loss
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}: Training Loss {train_epoch_loss}, Validation Loss {val_epoch_loss}')

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), f'/Models/Should/Trained_Models/best_model{date.today()}.pt')
        
        # if early_stopping >= 0 and train_epoch_loss > best_loss:
        #     best_loss = train_epoch_loss
        #     break  


def evaluate(model: nn.Module, 
             dataset: Dataset, 
             criterion: nn.modules.loss, 
             device: str='cpu'
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
    device: str
        The device to use for evaluation
    
    Returns:
    --------
    loss: float
        The loss for the evaluation
    """
    model.eval()
    loss = 0
    for data, target in dataset:
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