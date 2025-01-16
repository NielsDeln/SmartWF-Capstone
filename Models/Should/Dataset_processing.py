import os
import pandas as pd
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class Should_Dataset(Dataset):
    def __init__(self, dataset_path, data, load_axis, transforms=None) -> None:
        """
        Initializes the Should_Dataset class.

        Parameters:
        -----------
        dataset_path: str
            The path to the dataset
        data: Iterable[str]
            List of file names
        load_axis: str
            The axis to load the data
        transforms: callable
            A function/transform that takes input sample and its target as entry and returns a transformed version
        """
        self.dataset_path = dataset_path
        self.data = data
        self.load_axis = load_axis
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input, labels = load_input_output_tensor(self.dataset_path, self.data, index, self.load_axis)

        if self.transforms is not None:
            return self.transforms(input, labels)
        return input, labels


def load_input_output_tensor(dataset_path: str, data: Iterable[str], idx: int, load_axis: str) -> tuple[torch.Tensor, torch.Tensor]:
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
    load_axis: str
        The axis to load the data
    
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
                     skiprows=3)

    input = torch.tensor(df.iloc[::, 1].to_numpy())
    if load_axis == 'Mxb1':
        output = torch.tensor(df.iloc[::, 2].to_numpy())
    elif load_axis == 'Myb1':
        output = torch.tensor(df.iloc[::, 3].to_numpy())
    else:
        raise ValueError(f'load_axis must be either Mxb1 or Myb1, got {load_axis}')

    return input, output


def split_dataset(data_list: Iterable, test_size: float, validation_size: float, random_state: int=42) -> tuple[Iterable[str], ...]:
    """
    Split the dataset into train, validation, and test set

    parameters:
    -----------
    dataset_path: str
        The path to the dataset
    test_size: float
        The size of the test set
    validation_size: float
        The size of the validation set

    returns:
    --------
    train_data: Iterable[str]
        The train data files
    validation_data: Iterable[str]
        The validation data files
    test_data: Iterable[str]
        The test data files
    """
    test_validation_size = test_size + validation_size
    train_data, test_validation_data = train_test_split(data_list, 
                                                        test_size=test_validation_size, 
                                                        random_state=random_state,
                                                        shuffle=True,
                                                        )
    
    validation_data, test_data = train_test_split(test_validation_data, 
                                                  test_size=test_size/test_validation_size, 
                                                  random_state=random_state,
                                                  shuffle=True,
                                                  )

    return train_data, test_data, validation_data


if __name__ == '__main__':
    dataset_path = "/Users/niels/Desktop/TU Delft/Dataset/chunks/"
    test_dataset = Should_Dataset(dataset_path, os.listdir(dataset_path), 'Mxb1')