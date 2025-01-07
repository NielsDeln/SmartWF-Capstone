import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

def train_one_epoch(model: nn.Module, 
                    dataloader: DataLoader, 
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
    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
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
    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()
    return loss