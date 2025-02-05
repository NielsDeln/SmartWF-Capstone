import os
import re

folder_inputs = r"C:\Users\HugoP\OneDrive - Delft University of Technology\Year3\Q2 Capstone AI project\General\Should_dataset\MustShould_Dataset_w15-25_s1-2.5_peter\Inputs\\"
folder_outputs = r"C:\Users\niels\Downloads\Dataset\Must_Should_Dataset_rep_4\\"
# Change the folder to folder_outputs or folder_inputs
folder = folder_outputs

repetition_new = 4
# count increase by 1 in each iteration
# iterate all files from a directory

for filename in os.listdir(folder):
    # Construct old file name
    source = folder + filename
    # Extract numbers from the old file name
    match = re.match(r"w(\d+\.\d+)_s(\d+\.\d+)_(\d+)", filename)
    windspeed = match.group(1)
    STDeV = match.group(2)
    repetition = match.group(3)
    print('Old Name:', filename)
    # Construct new file name
    if folder == folder_inputs:
        new_filename = f"w{windspeed}_s{STDeV}_{repetition_new}_ms_in.dat"
    elif folder == folder_outputs:
        new_filename = f"w{windspeed}_s{STDeV}_{repetition_new}_ms_out.out"
    else:
        raise ValueError('Something goes wrong')
    print('New Name:', new_filename)
    
    # Adding the count to the new file name and extension
    destination = folder + new_filename   

    # Renaming the file
    os.rename(source, destination)

print('All Files Renamed')
print('New Names are')
# verify the result
res = os.listdir(folder)
print(res)