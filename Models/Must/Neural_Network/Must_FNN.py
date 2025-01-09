import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd


''''
class Should_model(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int=1, 
                 dropout: float=0., 
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
        """
        super(Should_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

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
'''

class Must_Model(nn.Module):
  """
  PyTorch neural network. Network layers are defined in __init__ and forward
  pass implemented in forward.
  """
  
  def __init__(self, 
               N_in, 
               hidden_dim, 
               N_out):
    """
    Args:
      N_in: number of features/dimensions of input layer
      hidden_dim: number of features/dimensions of hidden dimension
      N_out: number of features/dimensions of output layer
    """
    super(Must_Model, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(N_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_out),
        )

  def forward(self, x):

    y = self.linear_relu_stack(x)
    return y