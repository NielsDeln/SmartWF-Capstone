import openfast_toolbox
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

input_directory = '../../../Could_Dataset/Inputs'

data_list = []
i = 0
# Iterate through files in the directory
for file in os.listdir(input_directory)[:100]:  # List all files in the directory
    if file.endswith('.bts'):  # Check if the file has a .bts extension
        file_path = os.path.join(input_directory, file)  # Create the full path
        TS = openfast_toolbox.io.TurbSimFile(file_path)  # Process the .bts file
        plane = TS['u']
        plane = plane.astype(np.float32)  # Convert to float32
        data_list.append(plane)
        i += 1
        print(f'file number: {i}')
        
# Stack the data and save it in a compressed format
data = np.stack(data_list, axis=0)
np.savez_compressed('w5_w9_could_dataset_numpy_float32.npz', data)

