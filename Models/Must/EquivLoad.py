"""  - Open and OpenFAST binary file - Convert it to a pandas dataframe - Compute damage equivalent load for a given Wohler exponent 
""" 
import os 
import numpy as np
import matplotlib.pyplot as plt 

from openfast_toolbox import FASTOutputFile
from openfast_toolbox.postpro import equivalent_load

#Data directionary (For know this is an example of Julia's computer until we have all the data in a map)
output_dir = r"C:\Users\Jwoon\Desktop\STDeV 1.0 outputs"

# Loop through all files in the directory
for filename in os.listdir(output_dir):

        # Build the full file path
        file_path = os.path.join(output_dir, filename)
        df = FASTOutputFile(file_path).toDataFrame()

        m = 1 # Wohler slope 
        Leq = equivalent_load(df['Time_[s]'], df['RootMyc1_[kN-m]'], m=m) 
        print('Leq ',Leq)

# Leq = equivalent_load(df['Time_[s]'], df['RootMyc1_[kN-m]'], m=m, method='fatpack') # requires package fatpack

if __name__ == '__main__': 
    plt.show() 
if __name__ == '__test__': 
    np.testing.assert_almost_equal(Leq , 284.30398, 3) 
