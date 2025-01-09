import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

must_df = pd.read_csv("..\DEL_must_model_2.csv", sep='\t')
y = must_df['Leq_x'].to_numpy()
X = must_df[['Windspeed', 'STDeV']].to_numpy()

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)
# print(X_test, y_test)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
poly_reg_y_predicted = poly_reg_model.predict(X_test)
poly_reg_mae = mean_absolute_error(y_test,poly_reg_y_predicted)
# print(poly_reg_mae)

sgd = SGDRegressor()
sgd.fit(X_train, y_train)
predictions_sgd = sgd.predict(X_test)
mae_sgd = mean_absolute_error(y_test, predictions_sgd)
# print(mae_sgd)

# scores_sgd = cross_val_score(sgd, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
# mean_sgd = np.mean(-scores_sgd)
# print(mean_sgd)


def plot_predictions(pred, y):
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    xs = must_df['Windspeed']
    ys = must_df['STDeV']
    zs = must_df['Leq_x']
    ax1.scatter(xs, ys, zs)

    ax2 = fig.add_subplot(projection='3d')
    ax2.scatter(pred[0],pred[1], y)

    ax2.set_xlabel('Windspeed')
    ax2.set_ylabel('STDev')
    ax2.set_zlabel('Leq')

    plt.show()


print(X_train)
print(predictions_sgd)
# plot_predictions(predictions_sgd, y)
