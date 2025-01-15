import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
# Construct the path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# print(path)
# Add the path to sys.path
sys.path.append(path)
# Change the working directory
os.chdir(path)

def plot_data(must_df):
    unique_std = must_df['STDeV'].unique()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5 * len(unique_std)))
    axs = axs.flatten()

    for i, value in enumerate(unique_std):
        df_subset = must_df[must_df['STDeV'] == value]

        # First subplot with Leq_x and Leq_y
        ax1 = axs[0]
        ax1.plot(df_subset['Windspeed'], df_subset['Leq_x'], label=f'leq_x_s{value}')
        ax1.plot(df_subset['Windspeed'], df_subset['Leq_y'], label=f'leq_y_s{value}')
        ax1.set_ylabel('Leq_x and Leq_y')
        ax1.set_xlabel('Windspeed')
        ax1.set_title(f'Windspeed vs Leq_x and Leq_y, std: {value}')
        ax1.legend()
        ax1.grid()

        # Second subplot with Leq_res
        ax2 = axs[1]
        ax2.plot(df_subset['Windspeed'], df_subset['Leq_res'], label=f'leq_res_s{value}', color='r')
        ax2.set_ylabel('Leq_res')
        ax2.set_xlabel('Windspeed')
        ax2.set_title(f'Windspeed vs Leq_res, std: {value}')
        ax2.legend()
        ax2.grid()
    plt.tight_layout()
    plt.savefig()

def plot_label_pred_3D(ground_truth, predictions=None, title:str=None):
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

    if predictions is not None:
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

def plot_err_3D(ground_truth, 
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
    
    ax.scatter(xs1, ys1, zs2, marker='o', label=f'{error_type} Error')

    # Set labels and title
    ax.set_xlabel('Windspeed')
    ax.set_ylabel('STDev')
    ax.set_zlabel(f'{error_type} Error')
    ax.set_title(f'3D Scatter Plots, {error_type} error, {title}')
    
    ax.view_init(elev=20, azim=-122, roll=0)
    ax.legend()

def plot_label_pred_2D(ground_truth, 
                       predictions=None, 
                       title:str=None,
                       W_min=5, 
                       W_max=25, 
                       STDeV:list|str|int=all, 
                       ):
    # Here we plot the 2D scatter plot for the specific STDeV and W_speed value

    if STDeV == all:
            # STDeV = list(np.arange(0.25,2.75,0.25))
            STDeV = ground_truth['STDeV'].unique()
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
            if len(STDeV) > 6:
                raise ValueError('The number of STDeV values must be 6 or less.')

    for i in range(0, len(STDeV), 6):
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()
        for j in range(6):
            if i + j >= len(STDeV):
                break
            ax = axs[j]
            # Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j]) & 
            #             (ground_truth['Windspeed'] >= W_min) & 
            #             (ground_truth['Windspeed'] <= W_max)]
            Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j])]
            # Labels
            xs1 = Data_selection['Windspeed']
            zs1 = Data_selection.iloc[:,2]
            ax.scatter(xs1, zs1, marker='s', label='Data')

            if predictions is not None:
                # pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j]) & 
                #             (predictions['Windspeed'] >= W_min) & 
                #             (predictions['Windspeed'] <= W_max)]
                pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j])]
                # PREDICTIONS
                xs2 = pred_selection['Windspeed']
                zs2 = pred_selection.iloc[:,2]
                ax.scatter(xs2, zs2, label='Predictions')
            # Set labels and title
            ax.set_xlabel('Windspeed')
            ax.set_ylabel('Leq')
            ax.set_title(f'STDev={STDeV[i + j]}')
            ax.legend()
            ax.grid()
        # Adjust layout
        [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
        fig.subplots_adjust(hspace=0.4)
        plt.suptitle(f'2D Scatter Plot \nLabel and prediction\n{title}\nW_speeds: [{W_min},{W_max}]')

def plot_err_2D(ground_truth, predictions, title:str=None,W_min=5, W_max=25, STDeV:list|str|int=all, error_type='relative'):
    if STDeV == all:
        # STDeV = list(np.arange(0.25, 2.75, 0.25))
        STDeV = ground_truth['STDeV'].unique()
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

    for i in range(0, len(STDeV), 6):
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()
        for j in range(6):
            if i + j >= len(STDeV):
                break
            ax = axs[j]
            # Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i]) &
            #                             (ground_truth['Windspeed'] >= W_min) &
            #                             (ground_truth['Windspeed'] <= W_max)]

            # pred_selection = predictions[(predictions['STDeV'] == STDeV[i]) &
            #                             (predictions['Windspeed'] >= W_min) &
            #                             (predictions['Windspeed'] <= W_max)]
            Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i])]

            pred_selection = predictions[(predictions['STDeV'] == STDeV[i])]

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

            ax.scatter(xs1, zs2, marker='o', label=f'{error_type} Error')

            ax.set_xlabel('Windspeed')
            ax.set_ylabel(f'{error_type} Error')
            ax.set_title(f'STDev={STDeV[i + j]}')
            ax.legend()
            ax.grid()

        [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
        fig.subplots_adjust(hspace=0.4)
        plt.suptitle(f'2D Scatter Plot, {error_type} error, W_speeds: [{W_min},{W_max}]')

def plot_mean_error(ground_truth, 
                    predictions, 
                    title:str=None, 
                    variant='Windspeed',
                    W_min=5, 
                    W_max=25, 
                    STDeV:list|str|int=all,
                    error_type='relative'):
    
    label = ground_truth.iloc[:, 2]
    prediction = predictions.iloc[:, 2]
    if error_type == 'absolute':
        zs2 = abs((prediction - label))
    elif error_type == 'relative':
        zs2 = abs((prediction - label))/ label
    elif error_type == 'percentage':
        zs2 = abs((prediction - label))/ label * 100

    predictions['error'] = zs2
    mean_error = predictions.groupby(variant)['error'].mean()
    plt.figure()
    plt.plot(mean_error.index, mean_error.values, label=f'Mean {error_type} Error')
    plt.xlabel(variant)
    plt.ylabel(f'{error_type} Error')
    plt.title(f'{title}')
    plt.legend()
    plt.grid()

# must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model.csv', sep='\t')
# # Plot Leq_x
# y = must_df['Leq_x'].to_numpy()
# X = must_df[['Windspeed', 'STDeV']].to_numpy()
# all_data = pd.DataFrame(np.column_stack((X[:,:2], y)), columns=['Windspeed', 'STDeV', 'Leq_x'])
# plot_label_pred_3D(all_data, title='All Data, Leq_x')
# plot_label_pred_2D(all_data, title='All Data, Leq_x', STDeV=all)
# plt.show()

# # Plot Leq_y
# y = must_df['Leq_y'].to_numpy()
# X = must_df[['Windspeed', 'STDeV']].to_numpy()
# all_data = pd.DataFrame(np.column_stack((X[:,:2], y)), columns=['Windspeed', 'STDeV', 'Leq_y'])
# plot_label_pred_3D(all_data, title='All Data, Leq_y')
# plot_label_pred_2D(all_data, title='All Data, Leq_y', STDeV=all)
# plt.show()

# # Plot Leq_res
# y = must_df['Leq_res'].to_numpy()
# X = must_df[['Windspeed', 'STDeV']].to_numpy()
# all_data = pd.DataFrame(np.column_stack((X[:,:2], y)), columns=['Windspeed', 'STDeV', 'Leq_res'])
# plot_label_pred_3D(all_data, title='All Data, Leq_res')
# plot_label_pred_2D(all_data, title='All Data, Leq_res', STDeV=all)
# plt.show()