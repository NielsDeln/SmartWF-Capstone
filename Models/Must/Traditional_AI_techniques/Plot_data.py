import matplotlib as plt
import matplotlib.pyplot as plt
from EquivLoad import must_df

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

def plot_label_pred(ground_truth, predictions, title:str):
    # INPUT:
        # ground truth in fromat: [Wind_speed, STDeV, Leq]
        # predictions in format: [Windspeeds_pred, STDeV_pred, predictions]
        
    
    # Scatterplot with all predictions combined
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # First scatter plot
    xs1 = must_df['Windspeed']
    ys1 = must_df['STDeV']
    zs1 = ground_truth
    ax.scatter(xs1, ys1, zs1, marker='s', label='Data')

    # Second scatter plot
    xs2 = predictions[:,0]
    ys2 = predictions[:,1]
    zs2 = predictions[:,2]
    ax.scatter(xs2, ys2, zs2, marker='o', label='Predictions')

    # Set labels and title
    ax.set_xlabel('Windspeed')
    ax.set_ylabel('STDev')
    ax.set_zlabel('Leq')
    ax.set_title('3D Scatter Plots \n{title}')
    ax.legend()