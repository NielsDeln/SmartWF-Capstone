import openfast_toolbox
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

input_directory = '../../../Could_Dataset/Inputs'

#data = torch.zeros((50, 3, 1320, 41, 33))
#i = 0
index = ['vector', 'time', 'horizontal gridpoints', 'vertical gridpoints']
# Iterate through files in the directory
for file in os.listdir(input_directory[:100]):  # List all files in the directory
    if file.endswith('.bts'):  # Check if the file has a .bts extension
        file_path = os.path.join(input_directory, file)  # Create the full path
        TS = openfast_toolbox.io.TurbSimFile(file_path)  # Process the .bts file
        #data[i] = torch.from_numpy(TS['u'])
        plane = TS['u']
        df = pd.DataFrame(plane, index=index)
        #plane_tensor = torch.from_numpy(plane)
        #data[i] = plane_tensor
        break  # Remove this if you want to process all .bts files