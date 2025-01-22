'''
We have a dataset with the following distributions:
    
    6 samples for Windspeed[15-25], STDeV[1.00-2.50]
    1 sample for all other combinations in range Windspeed[5-25], STDeV[0.25-2.5]

We want to make a dataset where Windspeed-STDeV combination is represented in the train-validation-test dataset
'''

import os
import sys
import glob

# Construct the path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# print(path)
# Add the path to sys.path
sys.path.append(path)
# Change the working directory
os.chdir(path)

import pandas as pd
import matplotlib as plt
import tkinter as tk
from tkinter import scrolledtext

from sklearn.model_selection import train_test_split
from Models.Must.Traditional_AI_techniques.Plot_data import *

# Construct one complete dataset
path = r"C:\Users\HugoP\Desktop\SmartWF-Capstone\Models\Must\\" # use your path
all_files = glob.glob(os.path.join(path , "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep="\t")
    li.append(df)

must_df = pd.concat(li, axis=0, ignore_index=True)

# Split dataset in unique Windspeed-STDeV combinations and repeated combinations
selection_multiple = ((must_df['Windspeed'] >= 15) & (must_df['STDeV'] >= 1.00))
must_df_multiple = must_df[selection_multiple]
must_df_unique = must_df[~selection_multiple]

print((must_df_multiple.groupby(['Windspeed', 'STDeV']).size().reset_index(name='count')['count']==5).all())
print((must_df_unique.groupby(['Windspeed', 'STDeV']).size().reset_index(name='count')['count']==1).all())

# Split repeated combinations (2-2-2 distribution)
must_df_multiple['stratify_col'] = must_df_multiple['Windspeed'].astype(str) + '_' + must_df_multiple['STDeV'].astype(str)
X_multiple = must_df_multiple[['Windspeed', 'STDeV', 'stratify_col']]
y_multiple = must_df_multiple['Leq_y']

X_train_multiple, X_test_val_multiple, y_train_multiple, y_test_val_multiple = train_test_split(X_multiple, y_multiple, test_size=4/6, stratify=must_df_multiple['stratify_col'])
X_val_multiple, X_test_multiple, y_val_multiple, y_test_multiple = train_test_split(X_test_val_multiple, y_test_val_multiple, test_size=0.5, stratify=X_test_val_multiple['stratify_col'])

# Drop stratify column
X_train_multiple = X_train_multiple.drop(columns=['stratify_col'])
X_test_val_multiple = X_test_val_multiple.drop(columns=['stratify_col'])
X_val_multiple = X_val_multiple.drop(columns=['stratify_col'])
X_test_multiple = X_test_multiple.drop(columns=['stratify_col'])

# Split unique combinations
X_unique = must_df_unique[['Windspeed', 'STDeV']]
y_unique = must_df_unique['Leq_y']

X_train_unique, X_test_val_unique, y_train_unique, y_test_val_unique = train_test_split(X_unique, y_unique,test_size=4/6)
X_val_unique, X_test_unique, y_val_unique, y_test_unique = train_test_split(X_test_val_unique, y_test_val_unique, test_size=0.5)

# Combine to final datasets
X_train = pd.concat((X_train_unique, X_train_multiple))
X_val = pd.concat((X_val_unique, X_test_unique))
X_test = pd.concat((X_test_unique, X_test_multiple))

y_train = pd.concat((y_train_unique, y_train_multiple))
y_val = pd.concat((y_val_unique, y_val_multiple))
y_test = pd.concat((y_test_unique, y_test_multiple))

train = pd.concat((X_train, y_train))
val = pd.concat((X_val, y_val))
test = pd.concat((X_test, y_test))

"""# To test whether Test wheter train_test_split happend correctly:
def display_output_in_window(output):
    window = tk.Tk()
    window.title("Output Window")
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=200, height=60)
    text_area.pack(padx=10, pady=10)
    text_area.insert(tk.INSERT, output)
    text_area.configure(state='disabled')

# Analyse which samples are in the must_dataframe
Analyse = must_df.groupby(['Windspeed', 'STDeV']).size().reset_index(name='count')
Analyse = Analyse.sort_values(['STDeV','count']).to_string() + "\n\n"
display_output_in_window(Analyse)


# test_sets = {"Training":X_train_multiple, "Validation": X_val_multiple, "Testing": X_test_multiple}
# for key, set in test_sets.items():
#     set_df = pd.DataFrame(set, columns=['Windspeed', 'STDeV'])
#     set_df = set_df.groupby(['Windspeed', 'STDeV']).size().reset_index(name='count')
#     output = ""
#     output += f"Length {key} set: {len(set_df)}\n"
#     output += set_df.to_string()
#     output += f"All samples occur 2 times: {(set_df['count']==2).all()}\n"
#     output += "Combinations which don't occur 1 or 6 times\n"
#     output += set_df[(set_df['count'] != 1) & (set_df['count'] != 6)].to_string() + "\n\n"
#     output += "Combinations which do occur 1 or 6 times\n"
#     output += set_df[(set_df['count'] == 1) & (set_df['count'] == 6)].to_string() + "\n\n"
#     display_output_in_window(output)
# window = tk.Tk()
# window.mainloop()
# plt.show()"""