import timeit
from timeit import Timer
import numpy as np
from SimpleLinearRegression import SimpleLinearRegression1
from SimpleLinearRegression import SimpleLinearRegression2
import matplotlib.pyplot as plt

'''
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_predict = 6
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
reg1.predict(np.array([x_predict]))
y_hat = reg1.predict(x)
plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()
'''
m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)
reg1 = SimpleLinearRegression1()
reg2 = SimpleLinearRegression2()

print(reg1.fit(big_x, big_y))
print(reg2.fit(big_x, big_y))
