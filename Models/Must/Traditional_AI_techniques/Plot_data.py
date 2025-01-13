import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as plt

'''must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model.csv', sep='\t')

unique_std = must_df['STDeV'].unique()

for value in unique_std:
    df_subset = must_df[must_df['STDeV'] == value]
    print(df_subset)

    plt.plot(df_subset['Windspeed'], df_subset['Leq_x'])
    plt.ylabel('leq_x')
    plt.xlabel('Windspeed')
    plt.title(f'Windspeed vs leq_x, std: {value}')
    plt.show()

    plt.plot(df_subset['Windspeed'], df_subset['Leq_y'])
    plt.ylabel('leq_xy')
    plt.xlabel('Windspeed')
    plt.title(f'Windspeed vs leq_y, std: {value}')
    plt.show()'''

def plot_label_pred(ground_truth, predictions, title:str):
    # INPUT:
        # ground truth in fromat: [Wind_speed, STDeV, Leq]
        # predictions in format: [Windspeed, STDeV, predictions]
        
    
    # Labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # First scatter plot
    xs1 = ground_truth['Windspeed']
    ys1 = ground_truth['STDeV']
    zs1 = ground_truth.iloc[:,2]
    ax.scatter(xs1, ys1, zs1, marker='s', label='Data')

    # PREDICTIONS
    xs2 = predictions['Windspeed']
    ys2 = predictions['STDeV']
    zs2 = predictions.iloc[:,2]
    ax.scatter(xs2, ys2, zs2, marker='o', label='Predictions')

    # Set labels and title
    ax.set_xlabel('Windspeed')
    ax.set_ylabel('STDev')
    ax.set_zlabel('Leq')
    ax.set_title(f'3D Scatter Plots \nLabel and prediction\n{title}')
    
    ax.view_init(elev=20, azim=-122, roll=0)
    ax.legend()

def plot_rel_err(ground_truth, predictions, title:str):
    # INPUT:
        # ground truth in fromat: [Wind_speed, STDeV, Leq]
        # predictions in format: [Windspeeds, STDeV, predictions]
        
    
    # # Scatterplot with all predictions combined
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Relative Error
    xs1 = predictions['Windspeed']
    ys1 = predictions['STDeV']
    
    label = ground_truth.iloc[:,2]
    prediction = predictions.iloc[:,2]
    zs2 = (prediction - label)/label

    ax.scatter(xs1, ys1, zs2, marker='o', label='Error')

    # Set labels and title
    ax.set_xlabel('Windspeed')
    ax.set_ylabel('STDev')
    ax.set_zlabel('Relative Error')
    ax.set_title(f'3D Scatter Plots \nRelative Error\n{title}')
    
    ax.view_init(elev=20, azim=-122, roll=0)
    ax.legend()

