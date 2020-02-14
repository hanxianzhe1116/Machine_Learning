import numpy as np
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# plt.scatter(x, y)
# plt.axis([0, 6, 0, 6])
# plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)

fenzi = 0.0
fenmu = 0.0
for x_i, y_i in zip(x, y):
    fenzi += (x_i-x_mean)*(y_i-y_mean)
    fenmu += (x_i-x_mean)**2

a = fenzi/fenmu
b = y_mean-a*x_mean

# print(a, b)
y_hat = a * x + b

x_predict = 6
y_predict = x_predict * a + b
print(y_predict)
plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()