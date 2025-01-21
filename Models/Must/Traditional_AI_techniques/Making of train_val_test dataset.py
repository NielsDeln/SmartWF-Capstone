'''
We have a dataset with the following distributions:
    
    6 samples for Windspeed[15-25], STDeV[1.00-2.50]
    1 sample for all other combinations in range Windspeed[5-25], STDeV[0.25-2.5]

We want to make a dataset where Windspeed-STDeV combination is represented in the train-validation-test dataset
'''

import os
import sys
# Construct the path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# print(path)
# Add the path to sys.path
sys.path.append(path)
# Change the working directory
os.chdir(path)

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error


from Models.Must.Traditional_AI_techniques.Plot_data import *
must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model_repetition_0.csv', sep='\t')

X = must_df[['Windspeed', 'STDeV']].to_numpy()
y = must_df['Leq_y'].to_numpy()
must_df['stratify_col'] = must_df['Windspeed'].astype(str) + '_' + must_df['STDeV'].astype(str)

# print(must_df['stratify_col'])
# print(must_df['stratify_col'].unique())
# print(len(must_df['stratify_col'].unique()))

# # X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,test_size=2/6, stratify=must_df['stratify_col'] )

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,test_size=2/6)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5)

# # Drop the stratification column if not needed
# train = train.drop(columns=['stratify_col'])
# test = test.drop(columns=['stratify_col'])

test_sets = [X_train, X_val, X_test]
for set in test_sets:

    set_df = pd.DataFrame(set, columns=['Windspeed', 'STDeV'])
    set_df = set_df.groupby(['Windspeed', 'STDeV']).size().reset_index(name='count')
    print(set_df)
    print("Length set:", len(set_df))
    print("Combinations which don't occur 1 or 6 times")
    print(set_df[(set_df['count']!= 1) & (set_df['count']!= 6)])


