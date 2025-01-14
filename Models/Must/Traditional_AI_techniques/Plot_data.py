import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    plt.show()
'''

def plot_label_pred(ground_truth, predictions, title:str):
    """
    Plots the ground truth and predictions in a 3D scatter plot.

    Parameters:
    ground_truth (pd.DataFrame): DataFrame containing the ground truth values with columns ['Windspeed', 'STDeV', 'Leq'].
    predictions (pd.DataFrame): DataFrame containing the predicted values with columns ['Windspeed', 'STDeV', 'predictions'].
    title (str): Title for the plot.
    Returns:
    None
    """        
    # Labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
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

def plot_err(ground_truth, 
                 predictions,
                 title:str, 
                 error_type:str='absolute'):
    """
    Plots the relative error between ground truth and predictions in a 3D scatter plot.

    Parameters:
    ground_truth (pd.DataFrame): DataFrame containing the ground truth values with columns ['Windspeed', 'STDeV', 'Leq'].
    predictions (pd.DataFrame): DataFrame containing the predicted values with columns ['Windspeed', 'STDeV', 'predictions'].
    title (str): Title for the plot.
    Returns:
    None
    """
    # Scatterplot with all predictions combined
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Relative Error
    xs1 = predictions['Windspeed']
    ys1 = predictions['STDeV']
    
    label = ground_truth.iloc[:,2]
    prediction = predictions.iloc[:,2]
    if error_type == 'absolute':
        zs2 = (prediction - label)
    elif error_type == 'relative':
        zs2 = (prediction - label)/label
    elif error_type == 'percentage':
        zs2 = (prediction - label)/label * 100
    
    ax.scatter(xs1, ys1, zs2, marker='o', label='Error')

    # Set labels and title
    ax.set_xlabel('Windspeed')
    ax.set_ylabel('STDev')
    ax.set_zlabel('Relative Error')
    ax.set_title(f'3D Scatter Plots, {error_type}, {title}')
    
    ax.view_init(elev=20, azim=-122, roll=0)
    ax.legend()

def plot_label_pred_2D(ground_truth, 
                       predictions, 
                       title:str=None,
                       W_min=5, 
                       W_max=25, 
                       STDeV:list|str|int=all, 
                       ):
    if STDeV == all:
            STDeV = list(np.arange(0.25,2.75,0.25))
    elif type(STDeV) == int:
            if STDeV not in set(np.arange(0.25,2.75,0.25)):
                raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
            STDeV = [STDeV]
    elif type(STDeV) == list:
            for STDeV_value in STDeV:  
                if STDeV_value not in set(np.arange(0.25,2.5,0.25)):
                    raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
    else:
            raise ValueError('STDeV must be either a list, an integer or "all"')

    fig, axs = plt.subplots(len(STDeV) // 2 + (len(STDeV) % 2 > 0), 2, figsize=(20, 5 * (len(STDeV) // 2 + (len(STDeV) % 2 > 0))))
    axs = axs.flatten()
    
    for i in range(len(STDeV)):  
        ax = axs[i]
        # Here we plot the 2D scatter plot for the specific STDeV and W_speed value
        print(len(ground_truth['STDeV']))
        print(len(ground_truth['Windspeed']))
        print(STDeV[i])
        Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i]) & 
                    (ground_truth['Windspeed'] >= W_min) & 
                    (ground_truth['Windspeed'] <= W_max)]
        
        # Labels
        xs1 = Data_selection['Windspeed']
        zs1 = Data_selection.iloc[:,2]
        ax.scatter(xs1, zs1, marker='s', label='Data')

        pred_selection = predictions[(predictions['STDeV'] == STDeV[i]) & 
                    (predictions['Windspeed'] >= W_min) & 
                    (predictions['Windspeed'] <= W_max)]
        # PREDICTIONS
        xs2 = pred_selection['Windspeed']
        zs2 = pred_selection.iloc[:,2]
        ax.scatter(xs2, zs2, label='Predictions')

        # Set labels and title
        ax.set_xlabel('Windspeed')
        ax.set_ylabel('Leq')
        ax.set_title(f'STDev={STDeV[i]}')
        ax.legend()
        ax.grid()

    
    # Adjust layout
    [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
    # plt.tight_layout()
    plt.suptitle(f'2D Scatter Plot \nLabel and prediction\nW_speeds: [{W_min},{W_max}]', y=1.05)

def plot_err_2D(ground_truth, predictions, title:str=None,W_min=5, W_max=25, STDeV:list|str|int=all, error_type='relative'):
    if STDeV == all:
        STDeV = list(np.arange(0.25, 2.75, 0.25))
    elif type(STDeV) == int:
        if STDeV not in set(np.arange(0.25, 2.75, 0.25)):
            raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
        STDeV = [STDeV]
    elif type(STDeV) == list:
        for STDeV_value in STDeV:
            if STDeV_value not in set(np.arange(0.25, 2.5, 0.25)):
                raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
    else:
        raise ValueError('STDeV must be either a list, an integer or "all"')

    fig, axs = plt.subplots(len(STDeV) // 2 + (len(STDeV) % 2 > 0), 2, figsize=(20, 5 * (len(STDeV) // 2 + (len(STDeV) % 2 > 0))))
    axs = axs.flatten()

    for i in range(len(STDeV)):
        ax = axs[i]
        Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i]) &
                                    (ground_truth['Windspeed'] >= W_min) &
                                    (ground_truth['Windspeed'] <= W_max)]

        pred_selection = predictions[(predictions['STDeV'] == STDeV[i]) &
                                    (predictions['Windspeed'] >= W_min) &
                                    (predictions['Windspeed'] <= W_max)]

        xs1 = pred_selection['Windspeed']
        ys1 = pred_selection['STDeV']
        label = Data_selection.iloc[:, 2]
        prediction = pred_selection.iloc[:, 2]

        if error_type == 'absolute':
            zs2 = (prediction - label)
        elif error_type == 'relative':
            zs2 = (prediction - label) / label
        elif error_type == 'percentage':
            zs2 = (prediction - label) / label * 100

        ax.scatter(xs1, zs2, marker='o', label='Error')

        ax.set_xlabel('Windspeed')
        ax.set_ylabel(f'{error_type} Error')
        ax.set_title(f'STDev={STDeV[i]}')
        ax.legend()
        ax.grid()

    [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
    plt.suptitle(f'2D Scatter Plot, {error_type}, W_speeds: [{W_min},{W_max}]', y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.96])