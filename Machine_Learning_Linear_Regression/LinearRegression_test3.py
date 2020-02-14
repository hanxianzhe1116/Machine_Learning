import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

# 创建实例
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.fit(X, y))

print(lin_reg.coef_)
sorted_coef = np.argsort(lin_reg.coef_)
print(sorted_coef)

boston_feature_names = boston.feature_names
print(boston_feature_names[sorted_coef])
print(boston.DESCR)