import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd


class Should_RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(Should_RNN, self).__init__()

    def forward(self, x):
        pass

