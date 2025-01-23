import pandas as pd
import os
from collections.abc import Iterable

def preprocess_dataset(dataset_path: str, columns_keep: Iterable[int]=[0, 1, 12, 24, 25]) -> None:
    """
    This function takes a dataset and preprocesses it to keep only selected columns.
    The chunks are saved in the same directory as the original dataset.
    
    Parameters:
    -----------
    dataset_path (str): 
        The path to the dataset.
    columns_keep (Iterable[int]):
        The columns to keep in the dataset.
        Default: [0, 1, 12, 24, 25]
    """
    # Load the dataset
    datapoints = os.listdir(dataset_path)

    save_path = os.path.join(dataset_path, 'preprocessed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in datapoints:
        # Ignore all items that do not end with .out
        if not file.endswith('.out'):
            continue

        # Load the dataset
        file_path = os.path.join(dataset_path, file)
        df_data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=6, low_memory=False)

        # Split the dataset into a header and other data
        df_header = df_data.iloc[:2, columns_keep]

        # Disregard the first minute of data and keep only the relevant columns and concat them together
        df_data = df_data.iloc[1502:, columns_keep]
        df_data = pd.concat([df_header, df_data], axis=0)

        # Save the data
        filename = os.path.splitext(file)[0] + '_processed.csv'
        df_data.to_csv(os.path.join(save_path, filename), sep=' ', index=False)

if __name__ == "__main__":
    for dataset_path in [r'C:\Users\niels\Downloads\Dataset\Must_Should_Dataset_rep_4']:
        preprocess_dataset(dataset_path)
    print('Dataset chunking complete.')