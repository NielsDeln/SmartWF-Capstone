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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Models.Must.Traditional_AI_techniques.Plot_data import *
must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model.csv', sep='\t')
y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

# print(X.shape)
# print(y.shape)
# print(y, X)
degrees = np.arange(1,8)
for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test =  train_test_split(poly_features, y, test_size=0.3, random_state=42)

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


ground_truth = pd.DataFrame(np.column_stack((X_test[:,:2], y_test)), columns=['Windspeed', 'STDeV', 'Leq'])
predictions = pd.DataFrame(np.column_stack((X_test[:,:2], poly_reg_y_pred)), columns=['Windspeed', 'STDeV', 'Leq'])

plot_label_pred_3D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')
plot_err_3D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')

plot_label_pred_2D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')
plot_err_2D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', error_type='relative')

plot_mean_error(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', variant='Windspeed', error_type='relative')
plot_mean_error(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', variant='STDeV', error_type='relative')
plt.show()
