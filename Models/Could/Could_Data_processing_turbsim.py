import os
import numpy as np
import openfast_toolbox

#stop #remove before running

# Directories
input_directory = 'dataset2_original/Inputs'
output_directory = 'dataset2/Inputs'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Constants
frequency = 2  # Hz
time_to_remove = 60  # seconds
rows_to_remove = time_to_remove * frequency

count=0
# Step 1: Convert .bts files to .npy and process
print("Starting BTS to NPY conversion and processing...")
for file in os.listdir(input_directory):
    if file.endswith('.bts'):
        file_path = os.path.join(input_directory, file)
        TS = openfast_toolbox.io.TurbSimFile(file_path)
        plane = TS['u'].astype(np.float32)

        # Remove initial rows (reduce time dimension)
        if plane.shape[1] >= rows_to_remove:
            reduced_data = plane[:, rows_to_remove:]
            reduced_data = reduced_data[0, :, ::3, ::3]
        else:
            print(f"File {file} has fewer than {rows_to_remove} rows. Skipping.")
            continue
        
        output_file = os.path.join(output_directory, file.replace('.bts', '.npy'))
        np.save(output_file, reduced_data)

        print(f"Processed and reduced: {file} -> {output_directory}")
        count+=1



print("Processing {count} files completed.")
