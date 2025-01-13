import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

must_df = pd.read_csv('..\DEL_must_model.csv', sep='\t')
print(must_df)

y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

sgd = SGDRegressor()
sgd.fit(X_train, y_train)
predictions_sgd = sgd.predict(X_test)
mae_sgd = mean_absolute_error(y_test, predictions_sgd)
print(mae_sgd)

scores_sgd = cross_val_score(sgd, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
mean_sgd = np.mean(-scores_sgd)
print(mean_sgd)