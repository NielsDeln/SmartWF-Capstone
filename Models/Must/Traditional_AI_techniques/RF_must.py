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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from Models.Must.Traditional_AI_techniques.Plot_data import *
must_df = pd.read_csv(filepath_or_buffer=r'Models\Must\DEL_must_model.csv', sep='\t')

y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, predictions_rf)
print("mean absolute error:", mae_rf)

parameters_RF = {'criterion' : ['absolute_error'],
                        'n_estimators': [30, 40],
                        'max_depth' : [8, 10, 12, 15],
                        'min_samples_leaf': [2, 5, 10]
}
# Best found paramters: {'criterion': 'absolute_error', 'max_depth': 15, 'min_samples_leaf': 5, 'n_estimators': 40
optimize_RF = GridSearchCV(rf, parameters_RF, scoring = 'neg_mean_absolute_error', n_jobs=-1, verbose=1)
optimize_RF.fit(X_train, y_train)

print("Best MAE before tuning:", mae_rf)
print("Best negative MAE of Random Forest model: ", optimize_RF.best_score_)
print("Best parameters:", optimize_RF.best_estimator_.get_params)

predictions_rf_best = optimize_RF.predict(X_test)

# Scaling back to original values
X_test = scaler.inverse_transform(X_test)
# scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
# mean_rf = np.mean(-scores_rf)
# print("mean absolute error cross validation",mean_rf)


# plt.scatter(y_test, predictions_rf, alpha=0.7, edgecolor='k', label='Predictions')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
# plt.title("Predictions vs Real Numbers")
# plt.xlabel("Real Numbers (y_test)")
# plt.ylabel("Predictions")
# plt.legend()
# plt.grid(True)


ground_truth = pd.DataFrame(np.column_stack((X_test[:,:2], y_test)), columns=['Windspeed', 'STDeV', 'Leq'])
predictions = pd.DataFrame(np.column_stack((X_test[:,:2], predictions_rf_best)), columns=['Windspeed', 'STDeV', 'Leq'])
# print(ground_truth)
# print(predictions)

plot_label_pred_3D(ground_truth, predictions, title='Random Forest Regressor')
plot_err_3D(ground_truth, predictions, title='Random Forest Regressor', error_type='absolute')

plot_label_pred_2D(ground_truth, predictions, title='Random Forest Regressor', STDeV=all)
plot_err_2D(ground_truth, predictions, title='Random Forest Regressor', STDeV=all, error_type='relative')

plot_mean_error(ground_truth, predictions, title='Random Forest Regressor', variant='Windspeed', error_type='relative')
plot_mean_error(ground_truth, predictions, title='Random Forest Regressor', variant='STDeV', error_type='relative')
plt.show()