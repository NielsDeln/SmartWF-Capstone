import openfast_toolbox
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

input_directory = '../../../Could_Dataset/Outputs'

data_list = []
i = 0
# Iterate through files in the directory
for file in os.listdir(input_directory)[:51]:  # List all files in the directory
    if file.endswith('.out'):
        file_path = os.path.join(input_directory, file)  # Create the full path
        outputfile = openfast_toolbox.FASTOutputFile(file_path)
        df = outputfile.toDataFrame()
        colNames = df.columns 
        renameDict = {}
        unitsDict = {}
        for colName in colNames : 
            name,unit = colName.split("_")
            unit = unit.strip('[]')
            renameDict[colName] = name 
            unitsDict[name] = unit 
        df = df.rename(columns=renameDict)
        data = df[['Time', 'RootMxb1', 'RootMyb1']]
        data_list.append(data)
        i += 1
        print(f'file number: {i}')
        
# Stack the data and save it in a compressed format
data = np.stack(data_list, axis=0)
np.savez_compressed('../../../Could_Dataset/w5_w9_could_dataset_numpy_outputs_small.npz', data)

