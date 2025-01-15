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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from Models.Must.Traditional_AI_techniques.Plot_data import *
must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model.csv', sep='\t')
y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)

# define the pipeline
pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('preprocessor', PolynomialFeatures(degree=7, include_bias=False)),
    ('estimator', SGDRegressor(penalty='l2'))
])

# fit the pipeline
pipe.fit(X_train, y_train)
# Get the predictions
y_pred_train_pipe = pipe.predict(X_test)
print(pipe.score(X_test, y_test))

# Define the parameter grid
param_grid = {
    'preprocessor__degree': [1, 2, 3, 4, 5, 6, 7, 20],
    'estimator__alpha': [1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'estimator__max_iter': [100, 500, 1000, 2000, 5000]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_absolute_error')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# Use the best estimator to make predictions
best_pipe = grid_search.best_estimator_
y_pred_train_pipe = best_pipe.predict(X_test)
print(best_pipe.score(X_test, y_test))

# Plot the results
ground_truth = pd.DataFrame(np.column_stack((X_test[:,:2], y_test)), columns=['Windspeed', 'STDeV', 'Leq'])
predictions = pd.DataFrame(np.column_stack((X_test[:,:2], y_pred_train_pipe)), columns=['Windspeed', 'STDeV', 'Leq'])

plot_label_pred_3D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')
plot_err_3D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')

plot_label_pred_2D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression')
plot_err_2D(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', error_type='relative')

plot_mean_error(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', variant='Windspeed', error_type='relative')
plot_mean_error(ground_truth, predictions, title='Polynomial feature extraction Linear Regression', variant='STDeV', error_type='relative')
plt.show()
