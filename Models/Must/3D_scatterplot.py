import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixing random state for reproducibility
np.random.seed(19680801)


# making dataframe  
df = pd.read_csv("DEL_must_model.csv", sep='\t')  
   
# output the dataframe 
print(df['Windspeed'])

'''
def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 0, 25)
    ys = randrange(n, 0, 2.5)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)
'''

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