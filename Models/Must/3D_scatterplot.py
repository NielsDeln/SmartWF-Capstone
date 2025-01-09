import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# making dataframe  
df = pd.read_csv("DEL_must_model.csv", sep='\t')  
   
# # output the dataframe 
# print(df['Windspeed'])

fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')

xs = df['Windspeed']
ys = df['STDeV']
zs = df['Leq_x']
ax1.scatter(xs, ys, zs)

ax1.set_xlabel('Windspeed')
ax1.set_ylabel('STDev')
ax1.set_zlabel('Leq')

plt.show()