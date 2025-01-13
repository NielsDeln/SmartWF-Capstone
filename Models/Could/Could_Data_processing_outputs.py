import openfast_toolbox
import os
import numpy as np

# Input and output directories
input_directory = 'vtest/Outputs'
output_directory = 'numpy_data/Outputs'
columns_to_save = ['Time', 'RootMxb1', 'RootMyb1']

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through files in the input directory
for file in os.listdir(input_directory):  # List all files in the directory
    if file.endswith('.out'):  # Check if the file has a .out extension
        file_path = os.path.join(input_directory, file)  # Create the full path
        outputfile = openfast_toolbox.FASTOutputFile(file_path)
        df = outputfile.toDataFrame()
        
        # Process column names
        colNames = df.columns
        renameDict = {}
        unitsDict = {}
        for colName in colNames: 
            name, unit = colName.split("_")
            unit = unit.strip('[]')
            renameDict[colName] = name
            unitsDict[name] = unit
        df = df.rename(columns=renameDict)
        
        # Extract desired columns
        data = df[columns_to_save].to_numpy(dtype=np.float32)
        
        # Save the NumPy array with the same filename but a .npy extension
        output_file = os.path.join(output_directory, file.replace('.out', '.npy'))
        np.save(output_file, data)
        
        print(f"Processed and saved: {file} -> {output_file}")
