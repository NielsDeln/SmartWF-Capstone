import pandas as pd
import numpy as np
import os

def chunk_dataset(dataset_path, chunk_size):
    """
    This function takes a dataset and splits it into chunks of size chunk_size.
    The chunks are saved in the same directory as the original dataset.
    
    Parameters:
    dataset_path (str): The path to the dataset.
    chunk_size (int): The size of the chunks.
    
    Returns:
    None
    """
    # Load the dataset
    datapoints = os.listdir(dataset_path)

    save_path = os.path.join(dataset_path, 'chunks')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for datapoint in datapoints:
        # Skip hidden files
        if datapoint.startswith('.'):
            continue
        
        # Split the datapoint name
        datapoint_info = datapoint.split('_')

        # Load the dataset
        file_path = os.path.join(dataset_path, datapoint)
        df_data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=6)

        # Define the columns to keep
        columns_keep = [0, 1, 24, 25]

        # Split the dataset into a header and other data
        df_header = df_data.iloc[:2, columns_keep]

        # Disregard the first minute of data and keep only the relevant columns
        df_data = df_data.iloc[1503:, columns_keep]

        for i in range(len(df_data)//chunk_size):
            filename = datapoint_info[:3].join('_') + datapoint_info[3] + '_split' + str(i).zfill(2) + '.csv'
            df_chunk = df_data.iloc[i*chunk_size:(i+1)*chunk_size, ::]
            df_chunk = pd.concat([df_header, df_chunk], axis=0)
            df_chunk.to_csv(os.path.join(save_path, filename), sep=' ', index=False)

if __name__ == "__main__":
    chunk_dataset('/Users/niels/Desktop/TU Delft/Dataset', 1500)