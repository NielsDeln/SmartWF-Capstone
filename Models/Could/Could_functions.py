import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Test function
def hello():
    print('Hello world!')

# Example dataset class
class WindTurbineDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs  # Shape: (num_samples, seq_len, channels, height, width)
        self.targets = targets  # Shape: (num_samples, seq_len, output_dim)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs_batch, targets_batch) in enumerate(dataloader):
            # Move data to device (if using GPU)
            inputs_batch = inputs_batch.to(torch.float32)
            targets_batch = targets_batch.to(torch.float32)

            # Forward pass
            predictions = model(inputs_batch)

            # Compute loss
            loss = criterion(predictions, targets_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "trained_models/wind_turbine_model.pth")
    print("Model saved and training completed")
