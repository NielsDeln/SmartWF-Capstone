#from EquivLoad import must_df

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

must_df = pd.read_csv('..\DEL_must_model_2.csv', sep='\t')
# print(must_df)

y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, predictions_rf)
print("mean absolute error:", mae_rf)

scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
mean_rf = np.mean(-scores_rf)
print("mean absolute error cross validation",mean_rf)

plt.scatter(y_test, predictions_rf, alpha=0.7, edgecolor='k', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predictions vs Real Numbers")
plt.xlabel("Real Numbers (y_test)")
plt.ylabel("Predictions")
plt.legend()
plt.grid(True)
plt.show()


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
zs2 = predictions_rf
ax.scatter(xs2, ys2, zs2, marker='o', label='Random Forest')

# Set labels and title
ax.set_xlabel('Windspeed')
ax.set_ylabel('STDev')
ax.set_zlabel('Leq')
ax.set_title('3D Scatter Plots Random Forest')
ax.legend()
plt.show()
