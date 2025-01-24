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
    '''
    This function plots data from a DataFrame containing wind speed and Leq_x, Leq_y, Leq_res values.
    Parameters:
    must_df (pd.DataFrame): A DataFrame containing the following columns:
        - 'STDeV': Standard deviation values used to subset the data.
        - 'Windspeed': Wind speed measurements.
        - 'Leq_x': Sound level measurements in the x direction.
        - 'Leq_y': Sound level measurements in the y direction.
        - 'Leq_res': Resultant sound level measurements.
    Returns:
    None: This function does not return any value. It generates and displays plots.
    The function creates two subplots:
    1. Windspeed vs Leq_x and Leq_y for each unique standard deviation value.
    2. Windspeed vs Leq_res for each unique standard deviation value.
    '''
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
    ax.scatter(xs1, ys1, zs1, marker='x',c='blue', label='Target')

    if predictions is not None:
        # PREDICTIONS
        xs2 = predictions['Windspeed']
        ys2 = predictions['STDeV']
        zs2 = predictions.iloc[:,2]
        ax.scatter(xs2, ys2, zs2, marker='x',c='black', label='Prediction')

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
    target = ground_truth.iloc[:, 2]
    prediction = predictions.iloc[:, 2]

    # Calculate average for each unique combination
    target_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean()

    # Store each average at the corresponding 'Windspeed'-'STDeV' combination. (sometimes multiple times)
    ground_truth['average_Leq'] = ground_truth.apply(lambda row: target_average.loc[(row['Windspeed'], row['STDeV'])], axis=1)
    # print(ground_truth)
    if predictions is not None:
        predictions_average = predictions.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()
    
    if error_type == 'absolute':
        error = abs((prediction - target))
    elif error_type == 'relative':
        error = abs((prediction - target))/ target
    elif error_type == 'percentage':
        error = abs((prediction - target))/ target * 100
    elif error_type == 'pred_wrt_mean':
        error = abs(prediction - ground_truth['average_Leq']) / ground_truth['average_Leq']
    else:
        raise "This error_type is not possible"
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Relative Error
    xs1 = predictions['Windspeed']
    ys1 = predictions['STDeV']
    ax.scatter(xs1, ys1, error, marker='o', label=f'{error_type} Error')

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
    """
    Plots a 2D scatter plot of ground truth and prediction data based on specified standard deviation (STDeV) 
    and wind speed (Windspeed) ranges.
    Parameters:
    -----------
    ground_truth : DataFrame
        The ground truth data containing 'STDeV' and 'Windspeed' columns.
    predictions : DataFrame, optional
        The prediction data containing 'STDeV' and 'Windspeed' columns. Can also be None, if predictions shouldn't be plotted
    title : str, optional
        The title of the plot. Default is None.
    W_min : int, optional
        The minimum wind speed value to filter the data. Default is 5.
    W_max : int, optional
        The maximum wind speed value to filter the data. Default is 25.
    STDeV : list, str, or int, optional
        The standard deviation values to filter the data. It can be a list of values, a single integer, 
        or the string 'all' to include all values in the range [0.25, 2.5] with steps of 0.25. Default is 'all'.
    Raises:
    -------
    ValueError
        If STDeV is not in the range [0.25, 2.5] with steps of 0.25, or if it is not a list, integer, or 'all'.
        If the number of STDeV values exceeds 6.
    Returns:
    --------
    None
        The function generates and displays a 2D scatter plot showing the relation between 
    """
        
    # Here we plot the Leq in a 2D scatter plot for specific STDeV and W_speed values
    if STDeV == all:
            STDeV = list(np.arange(0.25,2.75,0.25))
            # STDeV = ground_truth['STDeV'].unique()
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
            GT_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j])]
            GT_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j]) & 
                                        (ground_truth['Windspeed'] >= W_min) & 
                                        (ground_truth['Windspeed'] <= W_max)]
            # Labels
            xs1 = GT_selection['Windspeed']
            zs1 = GT_selection.iloc[:,2]
            ax.scatter(xs1, zs1, marker='x',c='blue', label='Target')

            if predictions is not None:
                pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j]) & 
                                             (predictions['Windspeed'] >= W_min) & 
                                             (predictions['Windspeed'] <= W_max)]
                # pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j])]
                # PREDICTIONS
                xs2 = pred_selection['Windspeed']
                zs2 = pred_selection.iloc[:,2]
                ax.scatter(xs2, zs2, marker='x',c='black', label='Prediction')
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
    """
    Plots 2D scatter plots of prediction errors against windspeed for different standard deviation (STDeV) values.
    Parameters:
    -----------
    ground_truth : pd.DataFrame
        DataFrame containing the ground truth values with columns 'Windspeed', 'STDeV', and 'Leq'.
    predictions : pd.DataFrame
        DataFrame containing the predicted values with columns 'Windspeed', 'STDeV', and 'Leq'.
    title : str, optional
        Title for the entire plot (default is None).
    W_min : int, optional
        Minimum windspeed value to consider for plotting (default is 5).
    W_max : int, optional
        Maximum windspeed value to consider for plotting (default is 25).
    STDeV : list, str, or int, optional
        List of standard deviation values to consider for plotting, or 'all' to include all values, or a single float value (default is 'all').
    error_type : str, optional
        Type of error to plot. Options are 'absolute', 'relative', 'percentage', or 'pred_wrt_mean' (default is 'relative').
    Raises:
    -------
    ValueError
        If STDeV is not in the range [0.25, 2.5] with steps of 0.25, or if STDeV is not a list, integer, or 'all'.
        If error_type is not one of the specified options.
    Returns:
    --------
    None
        The function generates and displays scatter plots but does not return any value.
    Goal:
    -----
    The function aims to visualize the prediction errors for different windspeed and standard deviation combinations, 
    helping to analyze the performance of the prediction model.
    """
    
    if STDeV == all:
        STDeV = list(np.arange(0.25, 2.75, 0.25))
        # STDeV = ground_truth['STDeV'].unique()
    elif type(STDeV) == float:
        if STDeV not in set(np.arange(0.25, 2.75, 0.25)):
            raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
        STDeV = [STDeV]
    elif type(STDeV) == list:
        for STDeV_value in STDeV:
            if STDeV_value not in set(np.arange(0.25, 2.5, 0.25)):
                raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
    else:
        raise ValueError('STDeV must be either a list, an integer or "all"')
    
    target = ground_truth.iloc[:, 2]
    prediction = predictions.iloc[:, 2]

    # Calculate average for each unique combination
    target_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean()

    # Store each average at the corresponding 'Windspeed'-'STDeV' combination. (sometimes multiple times)
    ground_truth['average_Leq'] = ground_truth.apply(lambda row: target_average.loc[(row['Windspeed'], row['STDeV'])], axis=1)
    # print(ground_truth)
    if predictions is not None:
        predictions_average = predictions.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()
    
    if error_type == 'absolute':
        error = abs((prediction - target))
    elif error_type == 'relative':
        error = abs((prediction - target))/ target
    elif error_type == 'percentage':
        error = abs((prediction - target))/ target * 100
    elif error_type == 'pred_wrt_mean':
        # This is the error of each prediction wrt the average of target values with similar inputs
        error = abs(prediction - ground_truth['average_Leq']) / ground_truth['average_Leq']
    else:
        raise "This error_type is not possible"
    
    predictions['error'] = error
    error_average = predictions.groupby(['Windspeed', 'STDeV'])['error'].mean().reset_index()
    
    for i in range(0, len(STDeV), 6):
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()
        for j in range(6):
            if i + j >= len(STDeV):
                break
            ax = axs[j]
            Data_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i+j]) &
                                        (ground_truth['Windspeed'] >= W_min) &
                                        (ground_truth['Windspeed'] <= W_max)]

            pred_selection = predictions[(predictions['STDeV'] == STDeV[i+j]) &
                                        (predictions['Windspeed'] >= W_min) &
                                        (predictions['Windspeed'] <= W_max)]
            
            xs1 = pred_selection['Windspeed']
            ys1 = pred_selection['STDeV']
            error = pred_selection['error']
            ax.scatter(xs1, error, marker='o', label=f'{error_type} Error')

            ax.set_xlabel('Windspeed')
            ax.set_ylabel(f'{error_type} Error')
            ax.set_title(f'STDev={STDeV[i + j]}')
            ax.legend()
            ax.grid()

        [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
        fig.subplots_adjust(hspace=0.4)
        plt.suptitle(f'2D Scatter Plot, {error_type} error, W_speeds: [{W_min},{W_max}]\n{title}')


# Plot the error for the mean of the data. Can be used to plot mean of Windspeed along STDeV or vice versa.
def plot_mean_error(ground_truth,
                    predictions, 
                    title:str=None, 
                    variant='Windspeed',
                    W_min=5, 
                    W_max=25, 
                    STDeV:list|str|int=all,
                    error_type='relative'):
    
    """
    Plots the mean error between ground truth and predictions for a given variant.
    Parameters:
    -----------
    ground_truth : pd.DataFrame
        DataFrame containing the ground truth values. Must include columns 'Windspeed', 'STDeV', and 'Leq'.
    predictions : pd.DataFrame
        DataFrame containing the predicted values. Must include columns 'Windspeed', 'STDeV', and 'Leq'.
    title : str, optional
        Title of the plot. Default is None.
    variant : str, optional
        The variable to group by for plotting. Default is 'Windspeed'.
    W_min : int, optional
        Minimum windspeed value to consider. Default is 5.
    W_max : int, optional
        Maximum windspeed value to consider. Default is 25.
    STDeV : list, str, or int, optional
        Standard deviation values to consider. Default is all.
    error_type : str, optional
        Type of error to calculate. Options are 'absolute', 'relative', 'percentage', 'pred_wrt_mean'. Default is 'relative'.
    Returns:
    --------
    None
        This function does not return any value. It generates and displays a plot.
    Goal:
    -----
    The function calculates the mean error between the ground truth and predictions for a specified variant and plots it.
    """

    label = ground_truth.iloc[:, 2]
    prediction = predictions.iloc[:, 2]
    # Calculate average for each unique combination
    target_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean()

    # Store each average at the corresponding 'Windspeed'-'STDeV' combination. (sometimes multiple times)
    ground_truth['average_Leq'] = ground_truth.apply(lambda row: target_average.loc[(row['Windspeed'], row['STDeV'])], axis=1)

    if error_type == 'absolute':
        error = abs((prediction - label))
    elif error_type == 'relative':
        error = abs((prediction - label))/ label
    elif error_type == 'percentage':
        error = abs((prediction - label))/ label * 100
    elif error_type == 'pred_wrt_mean':
        # This is the error of each prediction wrt the average of target values with similar inputs
        error = abs(prediction - ground_truth['average_Leq']) / ground_truth['average_Leq']
    else:
        raise "This error_type is not possible"

    predictions['error'] = error
    mean_error = predictions.groupby(variant)['error'].mean()
    plt.figure()
    plt.plot(mean_error.index, mean_error.values, label=f'Average {error_type} Error')
    plt.xlabel(variant)
    plt.ylabel(f'Mean {error_type} Error')
    plt.title(f'{title}')
    plt.legend()
    plt.grid()


def plot_label_pred_2D_mean(ground_truth,
                            predictions=None, 
                            title:str=None,
                            W_min=5, 
                            W_max=25, 
                            STDeV:list|str|int=all, 
                            ):
    
    """
    Plots 2D scatter plots of ground truth and prediction data with mean values.
    Parameters:
    -----------
    ground_truth : pandas.DataFrame
        DataFrame containing the ground truth data with columns 'Windspeed', 'STDeV', and 'Leq'.
    predictions : pandas.DataFrame, optional
        DataFrame containing the prediction data with columns 'Windspeed', 'STDeV', and 'Leq'. Default is None.
    title : str, optional
        Title for the plot. Default is None.
    W_min : int, optional
        Minimum windspeed value for filtering the data. Default is 5.
    W_max : int, optional
        Maximum windspeed value for filtering the data. Default is 25.
    STDeV : list, str, or int, optional
        List of STDeV values to filter the data. Can also be 'all' to include all values or a single float value. Default is 'all'.
    Raises:
    -------
    ValueError
        If STDeV is not in the range [0.25, 2.5] with steps of 0.25 or if the number of STDeV values exceeds 6.
    Returns:
    --------
    None
        The function generates and displays scatter plots but does not return any value.
    Goal:
    -----
    To visualize the relationship between windspeed and Leq for different STDeV values, comparing ground truth and prediction data, and their respective averages.
    """
    
    if STDeV == all:
            STDeV = list(np.arange(0.25,2.75,0.25))
            # STDeV = ground_truth['STDeV'].unique()
    elif isinstance(STDeV, float):
            if STDeV not in set(np.arange(0.25,2.75,0.25)):
                raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
            STDeV = [STDeV]
    elif isinstance(STDeV, list):
            for STDeV_value in STDeV:  
                if STDeV_value not in set(np.arange(0.25,2.5,0.25)):
                    raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
    else:
            raise ValueError('STDeV must be either a list, an float or "all"')
            if len(STDeV) > 6:
                raise ValueError('The number of STDeV values must be 6 or less.')

    ground_truth_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()
    if predictions is not None:
        predictions_average = predictions.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()

    # print(predictions_average)
    for i in range(0, len(STDeV), 6):
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()
        for j in range(6):
            if i + j >= len(STDeV):
                break
            ax = axs[j]

            
            # GT_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j])]
            GT_selection = ground_truth[(ground_truth['STDeV'] == STDeV[i + j]) & 
                                        (ground_truth['Windspeed'] >= W_min) & 
                                        (ground_truth['Windspeed'] <= W_max)]
            
            # GT_average_selection = ground_truth_average[(ground_truth_average['STDeV'] == STDeV[i + j])]
            GT_average_selection = ground_truth_average[(ground_truth_average['STDeV'] == STDeV[i + j]) & 
                                                        (ground_truth_average['Windspeed'] >= W_min) & 
                                                        (ground_truth_average['Windspeed'] <= W_max)]
            # Labels
            xs1 = GT_selection['Windspeed']
            ys1 = GT_selection.iloc[:,2]
            ax.scatter(xs1, ys1, marker='x',c='blue', label='Target')

            # Average of labels
            xs2 = GT_average_selection['Windspeed'].unique()
            ys2 = GT_average_selection.iloc[:,2]
            ax.scatter(xs2,ys2, marker='>',c='orange', label='Targets average')

            # Predictions
            if predictions is not None:
                pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j]) & 
                            (predictions['Windspeed'] >= W_min) & 
                            (predictions['Windspeed'] <= W_max)]
                # pred_selection = predictions[(predictions['STDeV'] == STDeV[i + j])]

                # pred_selection_average = predictions_average[(predictions_average['STDeV'] == STDeV[i + j])]
                pred_selection_average = predictions_average[(predictions_average['STDeV'] == STDeV[i + j]) & 
                                                             (predictions_average['Windspeed'] >= W_min) & 
                                                             (predictions_average['Windspeed'] <= W_max)]
                # PREDICTIONS
                xs3 = pred_selection['Windspeed']
                ys3 = pred_selection.iloc[:,2]
                ax.scatter(xs3, ys3, marker='x',c='black', label='Prediction')

                # Average of predictions
                xs4 = pred_selection_average['Windspeed'].unique()
                ys4 = pred_selection_average.iloc[:,2]
                ax.scatter(xs4,ys4, marker='>',c='yellow', label='Predictions average')
            
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

# Mean of all errors
def plot_pred_error_2D_mean(ground_truth, predictions, title:str=None,W_min=5, W_max=25, STDeV:list|str|int=all, error_type='relative'):
    """
    Plots the prediction error in 2D scatter plots for different standard deviation (STDeV) values.
    Parameters:
    -----------
    ground_truth : DataFrame
        The ground truth data containing columns 'Windspeed', 'STDeV', and 'Leq'.
    predictions : DataFrame
        The predicted data containing columns 'Windspeed', 'STDeV', and 'Leq'.
    title : str, optional
        The title for the plot. Default is None.
    W_min : int, optional
        The minimum windspeed to consider for the plot. Default is 5.
    W_max : int, optional
        The maximum windspeed to consider for the plot. Default is 25.
    STDeV : list, str, or int, optional
        The standard deviation values to consider for the plot. Can be a list of values, a single value, or 'all' to consider all values. Default is 'all'.
    error_type : str, optional
        The type of error to plot. Can be 'absolute', 'relative', 'percentage', or 'pred_wrt_mean'. Default is 'relative'.
    Raises:
    -------
    ValueError
        If STDeV is not in the range [0.25, 2.5] with steps of 0.25.
        If STDeV is not a list, an integer, or 'all'.
        If error_type is not one of 'absolute', 'relative', 'percentage', or 'pred_wrt_mean'.
    Returns:
    --------
    None
    """
      
    if STDeV == all:
        STDeV = list(np.arange(0.25, 2.75, 0.25))
        # STDeV = ground_truth['STDeV'].unique()
    elif type(STDeV) == float:
        if STDeV not in set(np.arange(0.25, 2.75, 0.25)):
            raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
        STDeV = [STDeV]
    elif type(STDeV) == list:
        for STDeV_value in STDeV:
            if STDeV_value not in set(np.arange(0.25, 2.5, 0.25)):
                raise ValueError('STDeV must be in the range [0.25, 2.5] with steps of 0.25')
    else:
        raise ValueError('STDeV must be either a list, an integer or "all"')

    target = ground_truth.iloc[:, 2]
    prediction = predictions.iloc[:, 2]

    # Calculate average for each unique combination
    target_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean()

    # Store each average at the corresponding 'Windspeed'-'STDeV' combination. (sometimes multiple times)
    ground_truth['average_Leq'] = ground_truth.apply(lambda row: target_average.loc[(row['Windspeed'], row['STDeV'])], axis=1)
    # print(ground_truth)
    if predictions is not None:
        predictions_average = predictions.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()
    
    if error_type == 'absolute':
        error = abs((prediction - target))
    elif error_type == 'relative':
        error = abs((prediction - target))/ target
    elif error_type == 'percentage':
        error = abs((prediction - target))/ target * 100
    elif error_type == 'pred_wrt_mean':
        # This is the error of each prediction wrt the average of target values with similar inputs
        error = abs(prediction - ground_truth['average_Leq']) / ground_truth['average_Leq']
    else:
        raise "This error_type is not possible"
    
    predictions['error'] = error
    error_average = predictions.groupby(['Windspeed', 'STDeV'])['error'].mean().reset_index()
    
    for i in range(0, len(STDeV), 6):
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()
        for j in range(6):
            if i + j >= len(STDeV):
                break
            ax = axs[j]

            # Error
            error_selection = predictions[(predictions['STDeV'] == STDeV[i + j])]
            xs1 = error_selection['Windspeed']
            ys1 = error_selection['error']
            ax.scatter(xs1, ys1, marker = 'o', label=f"{error_type} error")

            if error_type != 'pred_wrt_mean':
                # Average of error
                error_average_selection = error_average[(error_average['STDeV'] == STDeV[i+j])]
                xs2= error_average_selection['Windspeed']
                ys2 = error_average_selection['error']
                ax.scatter(xs2, ys2, marker='2', label=f'Average of {error_type} Error')
           
            # Set labels and title
            ax.set_xlabel('Windspeed')
            ax.set_ylabel(f'{error_type} Error')
            ax.set_title(f'STDev={STDeV[i + j]}')
            ax.legend()
            ax.grid()

        [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
        fig.subplots_adjust(hspace=0.4)
        plt.suptitle(f'2D Scatter Plot, {error_type} error, W_speeds: [{W_min},{W_max}]\n{title}')






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