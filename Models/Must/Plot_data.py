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

