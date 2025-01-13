import openfast_toolbox
import os
import numpy as np

input_directories =   ['v5-10/Inputs',
                    'v10-15/Inputs',
                    'v15-20/Inputs',
                    'v20-25/Inputs']
output_directory = 'numpy_data/Inputs'
count = 0

# Iterate through files in the directories
for input_directory in input_directories:
    for file in os.listdir(input_directory):  # List all files in the directory
        if file.endswith('.bts'):  # Check if the file has a .bts extension
            file_path = os.path.join(input_directory, file)  # Create the full path
            TS = openfast_toolbox.io.TurbSimFile(file_path)  # Process the .bts file
            plane = TS['u']
            plane = plane.astype(np.float32)  # Convert to float32
            
            # Save the NumPy array with the same filename but a .npy extension
            output_file = os.path.join(output_directory, file.replace('.bts', '.npy'))
            np.save(output_file, plane)
            
            count += 1
            print(f"Processed and saved: {file} -> {output_file}")

print(f'Finished converting {count} files')
