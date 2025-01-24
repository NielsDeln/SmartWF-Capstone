import os
import numpy as np
import openfast_toolbox

#stop #remove before running

# Directories
openfast_output_dir = 'dataset2_original/Outputs'
processed_data_dir = 'dataset2/Outputs'

# Ensure the output directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Step 1: Convert OpenFAST output to NumPy arrays and process
columns_to_save = ['Time', 'RootMxb1', 'RootMyb1', 'B1Azimuth', 'B1Pitch']
frequency = 25  # Herz
time_to_remove = 60  # seconds
rows_to_remove = time_to_remove * frequency


count=0
for file in os.listdir(openfast_output_dir):
    if file.endswith('.out'):
        file_path = os.path.join(openfast_output_dir, file)
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

        # Remove initial rows (reduce time dimension)
        if data.shape[0] >= rows_to_remove:
            processed_data = data[rows_to_remove:]
        else:
            print(f"File {file} has fewer than {rows_to_remove} rows. Skipping.")   
            continue

        output_file = os.path.join(processed_data_dir, file.replace('.out', '.npy'))        
        np.save(output_file, processed_data)
        print(f"Processed and reduced: {file} -> {processed_data_dir}")
        count+=1

print("Processing {count} files completed.")
