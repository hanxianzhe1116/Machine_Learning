import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from train_test_split import train_test_split
from SimpleLinearRegression import SimpleLinearRegression2
import metrics as me
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as skm
boston = datasets.load_boston()
# print(boston.DESCR)
x = boston.data[:, 5]
y = boston.target
x = x[y < 50.0]
y = y[y < 50.0]
x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)
# print(x_train.shape)
# print(x_test.shape)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)
# print(reg.a_)
# print(reg.b_)

y_predict = reg.predict(x_test)
# print(y_predict)

"""
# MSE误差
mse_test = np.sum((y_predict - y_test) ** 2) / len(y_test)
# mse_test = me.mean_squared_error(y_test, y_predict)
print(mse_test)

# RMSE误差
rmse_tese = np.sqrt(mse_test)
print(rmse_tese)

# MAE误差
mae_test = np.sum(np.absolute((y_predict - y_test))) / len(y_test)
print(mae_test)
"""

# R Square
R_Square = 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)
R_Square = skm.r2_score(y_test, y_predict)
print(R_Square)
# plt.scatter(x_train, y_train)
# plt.plot(x_train, reg.predict(x_train), color='r')
# plt.show()