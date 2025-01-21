import pandas as pd
import os

def chunk_dataset(dataset_path: str, chunk_size: int) -> None:
    """
    This function takes a dataset and splits it into chunks of size chunk_size.
    The chunks are saved in the same directory as the original dataset.
    
    Parameters:
    -----------
    dataset_path (str): 
        The path to the dataset.
    chunk_size (int): 
        The size of the chunks.
    """
    # Load the dataset
    datapoints = os.listdir(dataset_path)

    save_path = os.path.join(dataset_path, 'chunks')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in datapoints:
        # Ignore all items that do not end with .out
        if not file.endswith('.out'):
            continue

        # Load the dataset
        file_path = os.path.join(dataset_path, file)
        df_data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=6)

        # Define the columns to keep
        columns_keep = [0, 1, 12, 24, 25]

        # Split the dataset into a header and other data
        df_header = df_data.iloc[:2, columns_keep]

        # Disregard the first minute of data and keep only the relevant columns
        df_data = df_data.iloc[1502:, columns_keep]

        for i in range(len(df_data)//chunk_size):
            filename = os.path.splitext(file)[0] + '_split' + str(i+1).zfill(2) + '.csv'
            df_chunk = df_data.iloc[i*chunk_size:(i+1)*chunk_size, ::]
            df_chunk = pd.concat([df_header, df_chunk], axis=0)
            df_chunk.to_csv(os.path.join(save_path, filename), sep=' ', index=False)

if __name__ == "__main__":
    chunk_dataset('c:/Users/niels/Downloads/Dataset/Must_Should_Dataset_rep_2/Outputs', 1500)
    print('Dataset chunking complete.')