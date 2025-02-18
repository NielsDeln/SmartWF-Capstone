import os
import sys
import glob
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

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error


from Models.Must.Traditional_AI_techniques.Plot_data import *

path = r"C:\Users\HugoP\Desktop\SmartWF-Capstone\Models\Must\\" # use your path
all_files = glob.glob(os.path.join(path , "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep="\t")
    li.append(df)

must_df = pd.concat(li, axis=0, ignore_index=True)

y = must_df['Leq_y'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# TRAINING
knn = KNeighborsRegressor()
knn_param_grid = {
    'n_neighbors': [3, 5, 10, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
 }
# knn_param_grid = {'n_neighbors': [3], 'weights': ['distance'], 'p': [2]}

knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='neg_mean_absolute_error')
knn_grid_search.fit(X_train, y_train)

predictions_knn_opt = knn_grid_search.predict(X_val)
mae_knn_opt = mean_absolute_error(y_val, predictions_knn_opt)

print('KNN MAE score:', mae_knn_opt, '\n',
"Best KNN Parameters:", knn_grid_search.best_params_)

knn_best = KNeighborsRegressor(n_neighbors = knn_grid_search.best_params_['n_neighbors'], 
                               weights = knn_grid_search.best_params_['weights'], 
                               p = knn_grid_search.best_params_['p'])
knn_best.fit(X_train, y_train)

# Best parameters: {'n_neighbors': 12, 'p': 2, 'weights': 'distance'}
# TESTING
scores_knn_best = cross_val_score(knn_best, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
mean_knn_best = np.mean(-scores_knn_best)
print('mean knn_best', mean_knn_best)

predictions_knn_best = knn_best.predict(X_test)
mae_knn_best = mean_absolute_error(y_test, predictions_knn_best)
print(('mae knn_best', mae_knn_best))

# plt.scatter(y_test, predictions_knn_best, alpha=0.7, edgecolor='k', label='Predictions')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
# plt.title("Predictions vs Real Numbers)")
# plt.xlabel("Real Numbers (y_test)")
# plt.ylabel("Predictions")
# plt.legend()
# plt.grid(True)
# plt.show()

X_train = scaler.inverse_transform(X_train)
X_val = scaler.inverse_transform(X_val)
X_test = scaler.inverse_transform(X_test)

all_data = pd.DataFrame(np.column_stack((X[:,:2], y)), columns=['Windspeed', 'STDeV', 'Leq'])
ground_truth = pd.DataFrame(np.column_stack((X_test[:,:2], y_test)), columns=['Windspeed', 'STDeV', 'Leq'])
predictions = pd.DataFrame(np.column_stack((X_test[:,:2], predictions_knn_best)), columns=['Windspeed', 'STDeV', 'Leq'])

# ground_truth_average = ground_truth.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()
# predictions_average = predictions.groupby(['Windspeed', 'STDeV'])['Leq'].mean().reset_index()

# print(average_samples)
# plot_label_pred_3D(ground_truth, predictions, title='KNN Regressor\nLeq_y')
# plot_err_3D(ground_truth, predictions, title='KNN Regressor\nLeq_y')

# plot_label_pred_2D(ground_truth, predictions, title='KNN Regressor\nLeq_y',STDeV=all)
# plot_err_2D(ground_truth, predictions, title='KNN Regressor\nLeq_y',STDeV=all, error_type='relative')

# plot_mean_error(ground_truth, predictions, title='KNN Regressor\nLeq_y', variant='Windspeed', error_type='relative')
# plot_mean_error(ground_truth, predictions, title='KNN Regressor\nLeq_y', variant='STDeV', error_type='relative')

plot_label_pred_2D_mean(ground_truth, predictions, title='KNN Regressor\nLeq_y')
plot_pred_error_2D_mean(ground_truth, predictions, title= 'KNN Regressor \nLeq_y')
plt.show()