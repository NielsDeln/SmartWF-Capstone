"""  - Open and OpenFAST .out file - Convert it to a pandas dataframe - Compute damage equivalent load for a given Wohler exponent 
""" 
import os 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import re

from openfast_toolbox import FASTOutputFile
from openfast_toolbox.postpro import equivalent_load

columns = ['Windspeed', 'STDeV', 'Leq_x', 'Leq_y', 'Leq_res']
must_df = pd.DataFrame(columns = columns)

#Data directionary (For know this is an example of Julia's computer until we have all the data in a map)
# Julia
output_dir = r"C:\Users\Jwoon\Desktop\All must data"
# Hugo
output_dir = r"C:\Users\HugoP\OneDrive - Delft University of Technology\Year3\Q2 Capstone AI project\General\Should_dataset\Must_Should_Dataset_rep_3"

counter = 0
# Loop through all files in the directory
for filename in os.listdir(output_dir):
        if counter < 5000:
            print(filename)
            # Build the full file path
            file_path = os.path.join(output_dir, filename)
            if file_path.endswith('.txt'):
                continue
            elif file_path.endswith('.ini'):
                 continue
            df = FASTOutputFile(file_path).toDataFrame()

            match = re.match(r"w(\d+\.\d+)_s(\d+\.\d+)_", filename)

            m = 10 # Wohler slope 
            Leq_x = equivalent_load(df['Time_[s]'], df['RootMxb1_[kN-m]'], m=m) 
            Leq_y = equivalent_load(df['Time_[s]'], df['RootMyb1_[kN-m]'], m=m) 
            df['ResBndMnt'] = np.sqrt((df['RootMxb1_[kN-m]']**2) + (df['RootMyb1_[kN-m]']**2))
            leq_rest = equivalent_load(df['Time_[s]'], df['ResBndMnt'], m=m)
            new_row = {
            'Windspeed': match.group(1),
            'STDeV': match.group(2),
            'Leq_x': Leq_x,
            'Leq_y': Leq_y,
            'Leq_res': leq_rest
            }
            
            must_df.loc[len(must_df)] = [float(match.group(1)), float(match.group(2)), Leq_x, Leq_y, leq_rest]
            counter += 1

    
must_df = must_df.sort_values(by='Windspeed', ascending=True)
print(must_df)

must_df.to_csv(path_or_buf='..\DEL_must_model_rep_3.csv', sep='\t', index=True, header=True)