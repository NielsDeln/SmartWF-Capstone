#from EquivLoad import must_df
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error

must_df = pd.read_csv('..\DEL_must_model_2.csv', sep='\t')
print(must_df)

y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

knn = KNeighborsRegressor()

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
 }

knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='neg_mean_absolute_error')
knn_grid_search.fit(X_train, y_train)

predictions_knn_opt = knn_grid_search.predict(X_val)
mae_knn_opt = mean_absolute_error(y_val, predictions_knn_opt)

print('KNN MAE score:', mae_knn_opt, '\n',
"Best KNN Parameters:", knn_grid_search.best_params_)

knn2 = KNeighborsRegressor(n_neighbors = knn_grid_search.best_params_['n_neighbors'], weights = knn_grid_search.best_params_['weights'], 
                           p = knn_grid_search.best_params_['p'])
knn2.fit(X_train, y_train)

scores_knn2 = cross_val_score(knn2, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
mean_knn2 = np.mean(-scores_knn2)
print('mean knn2', mean_knn2)

predictions_knn2 = knn2.predict(X_test)
mae_knn2 = mean_absolute_error(y_test, predictions_knn2)
print(('mae knn2', mae_knn2))

plt.scatter(y_test, predictions_knn2, alpha=0.7, edgecolor='k', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predictions vs Real Numbers)")
plt.xlabel("Real Numbers (y_test)")
plt.ylabel("Predictions")
plt.legend()
plt.grid(True)
plt.show()