import os
import pandas as pd
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset


class Should_Dataset(Dataset):
    def __init__(self, dataset_path, data, labels, transforms=None) -> None:
        """
        Initializes the Should_Dataset class.

        Parameters:
        -----------
        dataset_path: str
            The path to the dataset
        data: Iterable[str]
            List of file names
        labels: Iterable[str]
            List of labels
        transforms: callable
            A function/transform that takes input sample and its target as entry and returns a transformed version
        """
        self.dataset_path = dataset_path
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input, labels = load_input_output_tensor(self.dataset_path, self.data, index)

        if self.transforms is not None:
            return self.transforms(input, labels)
        return input, labels


def load_input_output_tensor(dataset_path: str, data: Iterable[str], idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the input and output tensors from the output file

    parameters:
    -----------
    dataset_path: str
        The path to the dataset
    data: Iterable[str]
        List of file names
    idx: int
        The index of the file to load
    
    returns:
    --------
    input: torch.Tensor
        The input tensor
    output: torch.Tensor
        The output tensor
    """
    file_path = os.path.join(dataset_path, data[idx])

    df = pd.read_csv(file_path,
                     delim_whitespace=True, 
                     header=None, 
                     skiprows=6)

    df = df.apply(pd.to_numeric, errors='coerce')

    wind_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
    wind_speeds = df.iloc[152:, wind_index].mean(axis=1)
    moments = df.iloc[152:, 24:30]

    input = torch.tensor(wind_speeds.values)
    output = torch.tensor(moments.values)

    return input, output

if __name__ == '__main__':
    dataset_path = "/Users/niels/Desktop/TU Delft/SmartWF-Capstone/Models/Should/"
    test_dataset = Should_Dataset(dataset_path, ["test.csv"], ["test.csv"])