import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd


class Should_LSTM(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int=1, 
                 dropout: float=0., 
                 regularization: str=None
                 ) -> None:
        """
        Initializes the Should_LSTM class.

        Parameters:
        -----------
        input_size: int
            The number of expected features in the input x
        hidden_size: int
            The number of features in the hidden state h
        num_layers: int
            Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        dropout: float
            If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        regularization: str
            Regularization method to use. Default: None
        """
        super(Should_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        if regularization is not None:
            self.regularization = self.add_regularization(regularization)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x, h0=None, c0=None) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Parameters:
        -----------
        x: torch.Tensor
            Input data
        h0: torch.Tensor
            Initial hidden state
        c0: torch.Tensor
            Initial cell state
        
        Returns:
        --------
        out: torch.Tensor
            Output data
        """
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        return out

    def add_regularization(self, regularization: str):
        if regularization == 'l1':
            return nn.L1Loss()
        elif regularization == 'l2':
            return nn.MSELoss()
        else:
            return None