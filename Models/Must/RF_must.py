#from EquivLoad import must_df
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

must_df = pd.read_csv('DEL_must_model.csv', sep='\t')
print(must_df)

y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, predictions_rf)
print(mae_rf)

scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
mean_rf = np.mean(-scores_rf)
print(mean_rf)

plt.scatter(y_test, predictions_rf, alpha=0.7, edgecolor='k', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predictions vs Real Numbers")
plt.xlabel("Real Numbers (y_test)")
plt.ylabel("Predictions")
plt.legend()
plt.grid(True)
plt.show()

