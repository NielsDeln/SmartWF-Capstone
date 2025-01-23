import os
import numpy as np
import pandas as pd
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class Should_Dataset(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 data: Iterable[str], 
                 load_axis: str, 
                 label_mean: float | None=None, 
                 label_std: float | None=None, 
                 transforms=None,
                ) -> None:
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
        label_mean: float | None
            The mean value for label normalization
        label_std: float | None
            The standard deviation value for label normalization
        transforms: callable
            A function/transform that takes input sample and its target as entry and returns a transformed version
        """
        self.dataset_path = dataset_path
        self.data = data
        self.transforms = transforms
        if load_axis != 'Mxb1' and load_axis != 'Myb1':
            raise ValueError(f'load_axis must be either Mxb1 or Myb1, got {load_axis}')
        else:
            self.load_axis = load_axis
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input, labels, time = load_input_output_tensor(self.dataset_path, self.data, index, self.load_axis)
        input[::, 1] = np.sin(input[::, 1]*np.pi/180) # convert azimuth axis to sinusodial function
        
        if self.transforms is not None:
            return self.transforms(input, labels), time
        return input, labels, time


def load_input_output_tensor(dataset_path: str, data: Iterable[str], idx: int, load_axis: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the input and output tensors from the output file

    Parameters:
    -----------
    dataset_path: str
        The path to the dataset
    data: Iterable[str]
        List of file names
    idx: int
        The index of the file to load
    load_axis: str
        The axis to load the data
    
    Returns:
    --------
    input: torch.Tensor
        The input tensor
    output: torch.Tensor
        The output tensor
    time: torch.Tensor
        The time tensor
    """
    file_path = os.path.join(dataset_path, data[idx])

    df = pd.read_csv(file_path,
                     delim_whitespace=True, 
                     header=None, 
                     skiprows=3)

    input = torch.reshape(torch.tensor(df.iloc[::, [1, 2]].to_numpy(), dtype=torch.float32), (15001, 2))
    time = torch.reshape(torch.tensor(df.iloc[::, 0].to_numpy(), dtype=torch.float32), (15001, 1))
    if load_axis == 'Mxb1':
        output = torch.reshape(torch.tensor(df.iloc[::, 3].to_numpy(), dtype=torch.float32), (15001, 1))
    elif load_axis == 'Myb1':
        output = torch.reshape(torch.tensor(df.iloc[::, 4].to_numpy(), dtype=torch.float32), (15001, 1))
    else:
        raise ValueError(f'load_axis must be either Mxb1 or Myb1, got {load_axis}')

    return input, output, time


def split_dataset(data_list: Iterable, test_size: float, validation_size: float, random_state: int=42) -> tuple[Iterable[str], ...]:
    """
    Split the dataset into train, validation, and test set

    Parameters:
    -----------
    data_list: Iterable
        List of data files
    test_size: float
        The size of the test set
    validation_size: float
        The size of the validation set
    random_state: int
        The random seed for shuffling the data
        default: 42

    Returns:
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



def calculate_average_and_std(dataset_path: str, data: Iterable[str], load_axis: str) -> tuple[float, float]:
    """
    Calculate the average and standard deviation of all labels in the dataset.

    Parameters:
    -----------
    dataset_path: str
        The path to the dataset
    data: Iterable[str]
        List of file names
    load_axis: str
        The axis of which the loads will be predicted, calculate the average value over this axis

    Returns:
    --------
    average_value: float
        The average value of the specified column over all files
    std_value: float
        The standard deviation of the specified column over all files
    """
    total_sum = 0.0
    total_count = 0
    all_values = []
    skipped_files = []
    error_count = 0

    for file_name in data:
        file_path = os.path.join(dataset_path, file_name)
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None, skiprows=3)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            skipped_files.append(file_name)
            error_count += 1
            continue
        df = pd.read_csv(file_path, sep='\s+', header=None, skiprows=3)
        
        if load_axis == 'Mxb1':
            column_values = df.iloc[:, 3].to_numpy()
        elif load_axis == 'Myb1':
            column_values = df.iloc[:, 4].to_numpy()
        else:
            raise ValueError(f'load_axis must be either Mxb1 or Myb1, got {load_axis}')
        total_sum += column_values.sum()
        total_count += len(column_values)
        all_values.extend(column_values)

    average_value = total_sum / total_count
    std_value = np.std(all_values)

    print(f'Number of files skipped: {error_count}')
    print(f'Calculating average and standard deviation is complete')
    print(f'training set average: {average_value}, training set standard deviation: {std_value}')
    return average_value, std_value