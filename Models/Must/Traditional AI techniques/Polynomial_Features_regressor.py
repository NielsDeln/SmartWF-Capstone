import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

must_df = pd.read_csv("..\DEL_must_model_2.csv", sep='\t')
y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

degrees = np.arange(2,3)
degree = 1
print(degrees)
for d in degrees:
    poly = PolynomialFeatures(degree=8, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)
    poly_reg_y_pred = poly_reg_model.predict(X_test)
    poly_reg_mae = mean_absolute_error(y_test,poly_reg_y_pred)
    print(f"LinearRegression degree {d} mae:", poly_reg_mae)
    # sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal')
    # sgd.fit(X_train, y_train)
    # predictions_sgd = sgd.predict(X_test)
    # mae_sgd = mean_absolute_error(y_test, predictions_sgd)
    # print(f"SGD Regression {d} mae:", mae_sgd)


'''
Optimal parameters for Linear Regression, degree=8, bias=False
'''
# Scatterplot with all predictions combined
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# First scatter plot
xs1 = must_df['Windspeed']
ys1 = must_df['STDeV']
zs1 = must_df['Leq_x']
ax.scatter(xs1, ys1, zs1, marker='s', label='Data')

# Second scatter plot
xs2 = X_test[:,0]
ys2 = X_test[:,1]
zs2 = poly_reg_y_pred
ax.scatter(xs2, ys2, zs2, marker='o', label='Linear Regression')

# # Third scatter plot
# zs3 = predictions_sgd
# ax.scatter(xs2, ys2, zs3, marker='^', label='SGD')

# Set labels and title
ax.set_xlabel('Windspeed')
ax.set_ylabel('STDev')
ax.set_zlabel('Leq')
ax.set_title('Combined 3D Scatter Plots')
ax.legend()
plt.show()

