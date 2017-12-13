import numpy as np
from matplotlib import pyplot as plt

def weighted_less_squares(x, y, weights, degree, point):
    matrix = np.zeros((degree+1, degree+1))
    right_side = np.zeros((degree+1))
    xd = np.zeros((degree+1, len(x)))
    for i in range(degree+1):
        xd[i,:] = x ** i
    for i in range(degree+1):
        right_side[i] = np.sum(weights * y * x ** i)
        matrix[i,:] = np.sum(weights * x ** i * xd, axis=1)
    a = np.linalg.solve(matrix, right_side)
    return np.sum(a * point ** np.arange(degree+1))

def bisquare(x):
    result = 1 - x * x
    result[result < 0] = 0
    return result ** 2

def tricube(x):
    result = 1 - np.abs(x ** 3)
    result[result < 0] = 0
    return result ** 3

def loess(x, y, fraction, degree, weight_function):
    y = y.copy()
    domain = x.max() - x.min()
    halfwidth = domain * fraction / 2
    weights = np.zeros((len(x), len(x)))
    new_y = y.copy()
    for k, xk in enumerate(x):
        dx = (x - xk) / halfwidth
        weights[k,:] = weight_function(dx)
        new_y[k] = weighted_less_squares(x, y, weights[k,:], degree, xk)

    for i in range(10):
        eps = np.abs(y - new_y)
        delta = bisquare(eps / (6 * np.median(eps)))
        for k, xk in enumerate(x):
            weights[k,:] *= delta[k]
            new_y[k] = weighted_less_squares(x, y, weights[k,:], degree, xk)
    return new_y

x = np.linspace(0, 5*np.pi, 100)
signal = np.sin(x)
noise = np.random.rand(len(signal)) * 2 - 1
y = signal + noise
plt.title("LOESS (10 iterations)")
plt.plot(x, signal, "b-", lw=2, label="signal")
plt.plot(x, y, "b+", label="signal+noise")
smoothed = loess(x, y, 0.3, 0, bisquare)
plt.plot(x, smoothed, "m-", label="loess (d = 0, f = 0.3)")
smoothed = loess(x, y, 0.3, 1, bisquare)
plt.plot(x, smoothed, "g--", label="loess (d = 1, f = 0.3)")
smoothed = loess(x, y, 0.3, 2, bisquare)
plt.plot(x, smoothed, "r-", label="loess (d = 2, f = 0.3)")
smoothed = loess(x, y, 0.3, 3, bisquare)
plt.plot(x, smoothed, "k--", label="loess (d = 3, f = 0.3)")
plt.legend(loc=4)
plt.show()