from Could_functions import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(32 * (input_hight // 4) * (input_width // 4), feature_dim)  # Adjust for input size

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.conv(x)  # Apply CNN
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected to feature vector
        x = x.view(batch_size, seq_len, -1)  # Reshape to sequence format
        return x

class RNNModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM layer
        x = self.fc(x)  # Fully connected layer for predictions
        return x

# Combine CNN and RNN into a single model
class WindTurbineLoadPredictor(nn.Module):
    def __init__(self, input_channels, feature_dim, hidden_dim, output_dim, num_layers):
        super(WindTurbineLoadPredictor, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, feature_dim)
        self.rnn = RNNModel(feature_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        x = self.cnn(x)  # Extract spatial features
        x = self.rnn(x)  # Model temporal dependencies
        return x



# Example usage
input_channels = 3          # Number of channels in 2D wind speed field
input_hight = 33            # Height of the 2D input field
input_width = 41            # Width of the 2D input field
feature_dim = 64            # Feature dimension from CNN
hidden_dim = 128            # Hidden dimension of LSTM
output_dim = 2              # Output dimension (load value)
num_layers = 2              # Number of LSTM layers
seq_len = 150               # Length of the input sequence
batch_size = 1              # Batch size


'''
# Example usage
# Instantiate the model
model = WindTurbineLoadPredictor(input_channels, feature_dim, hidden_dim, output_dim, num_layers)

# Example input (batch_size, seq_len, input_channels, height, width)
input_data = torch.randn(batch_size, seq_len, input_channels, input_hight, input_width)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, output_dim)

#Calculate total number of learnable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")


# Example Training
# Hyperparameters
num_epochs = 5
learning_rate = 0.001
batch_size = 8

# Example data (replace with your actual data)
num_samples = 40
inputs = torch.randn(num_samples, seq_len, input_channels, input_hight, input_width)  # Random data
targets = torch.randn(num_samples, seq_len, output_dim)  # Random targets

# DataLoader
dataset = WindTurbineDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, and optimizer
model = WindTurbineLoadPredictor(input_channels, feature_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, dataloader, criterion, optimizer, num_epochs)
'''