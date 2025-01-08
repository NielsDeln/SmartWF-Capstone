"""  - Open and OpenFAST binary file - Convert it to a pandas dataframe - Compute damage equivalent load for a given Wohler exponent 
""" 
import os 
import numpy as np
import matplotlib.pyplot as plt 

from openfast_toolbox import FASTOutputFile
from openfast_toolbox.postpro import equivalent_load

# Read an openFAST output file 
fastoutFilename = "../../Must_Should_Dataset/Outputs/w14.5000_s0.50_0_ms_out.out"
df = FASTOutputFile(fastoutFilename).toDataFrame()

# Compute equivalent load for one signal and Wohler slope 
m = 1 # Wohler slope 
Leq = equivalent_load(df['Time_[s]'], df['RootMyc1_[kN-m]'], m=m) 
print('Leq ',Leq)

# Leq = equivalent_load(df['Time_[s]'], df['RootMyc1_[kN-m]'], m=m, method='fatpack') # requires package fatpack

if __name__ == '__main__': 
    plt.show() 
if __name__ == '__test__': 
    np.testing.assert_almost_equal(Leq , 284.30398, 3) 
