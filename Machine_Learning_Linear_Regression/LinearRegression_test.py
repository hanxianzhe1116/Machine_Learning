import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from train_test_split import train_test_split
from LinearRegression import LinearRegression
boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
reg = LinearRegression()
reg.fit_normal(X_train, y_train)
print(reg.interception_)
print(reg.coef_)
print(reg.score(X_test, y_test))