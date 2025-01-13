from openfast_toolbox import FASTOutputFile
import matplotlib.pyplot as plt
import numpy as np
import os


path = r"C:\Users\HugoP\Desktop\SmartWF-My local datasets\Something went wrong"
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".out") and entry.is_file():
            print(entry.name, entry.path)

# directory_in_str = r"C:\Users\HugoP\Desktop\SmartWF-My local datasets\Something went wrong"
# directory = os.fsencode(directory_in_str)
    
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".asm") or filename.endswith(".py"): 
#         print(os.path.join(directory, filename))
#         continue
#     else:
#         continue

# outputfile_location = 
# outputfile = FASTOutputFile(outputfile_location)
# df = outputfile.toDataFrame()

# #Rename columns and store units 
# colNames = df.columns 
# renameDict = {}
# unitsDict = {}
# for colName in colNames : 
#     name,unit = colName.split("_")
#     unit = unit.strip('[]')
#     renameDict[colName] = name 
#     unitsDict[name] = unit 
# df = df.rename(columns=renameDict)
# df['ResBndMnt'] = np.sqrt((df.RootMxb1 **2) + (df.RootMyb1 **2))
# unitsDict['ResBndMnt'] = unitsDict['RootMxb1']

'''
Loop over all output files
    copy output file to temporary file
    delete columns except needed columns
    safe temporary file with name: 'originalname_stripped'
'''