import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x-2.5)**2 - 1
plt.plot(plot_x, plot_y)
plt.show()


def dJ(theta):
    return 2 * (theta - 2.5)


def J(theta):
    return (theta - 2.5)**2 - 1


def gradient_descent(init_theta, eta, epsilon=1e-8):
    theta = init_theta
    theta_history = [init_theta]

    # 为了避免死循环，可以加一个次数判断参数，确保不会陷入死循环
    while True:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
    return theta_history


def plot_theta_history(theta_history):
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()


plot_theta_history(gradient_descent(0., 0.1))