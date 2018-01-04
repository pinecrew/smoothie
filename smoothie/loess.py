import numpy as np
from matplotlib import pyplot as plt

def weighted_less_squares(x, y, weights, degree, point):
    """
    Calculate weighted less squares approximation in given point
    """
    fit = np.polyfit(x, y, degree, w=weights)
    return np.polyval(fit, point)

def bisquare(x):
    """
    Bisquare weight function :math:`w(x)=(1-x^2)^2`
    """
    result = 1 - x * x
    result[result < 0] = 0
    return result ** 2

def tricube(x):
    """
    Tricube weight function :math:`w(x)=(1-|x|^3)^3`
    """
    result = 1 - np.abs(x ** 3)
    result[result < 0] = 0
    return result ** 3

def loess(x, y, fraction=2/3, degree=2, weight_function=bisquare, iterations=3):
    """
    LOESS -- LOcal regrESSion scatterplot smoothing, according to Cleveland (1979)
    """
    x, y = np.asarray(x), np.asarray(y)
    domain = x.max() - x.min()
    halfwidth = domain * fraction / 2
    new_y = y.copy()
    xi, xj = np.meshgrid(x, x)
    dx = (xi - xj) / halfwidth
    weights = weight_function(dx)
    for k, xk in enumerate(x):
        new_y[k] = weighted_less_squares(x, y, weights[k,:], degree, xk)
    for i in range(iterations):
        eps = np.abs(y - new_y)
        delta = bisquare(eps / (6 * np.median(eps)))
        weights *= delta[:np.newaxis]
        for k, xk in enumerate(x):
            new_y[k] = weighted_less_squares(x, y, weights[k,:], degree, xk)
    return new_y

if __name__ == "__main__":
    x = np.linspace(0, 5*np.pi, 100)
    signal = 10 * np.exp(-x/3)
    noise = np.random.rand(len(signal)) * 2 - 1
    y = signal + noise
    plt.title("LOESS (3 iterations)")
    plt.plot(x, signal, "b-", lw=2, label="signal")
    plt.plot(x, y, "b+", label="signal+noise")
    # smoothed = loess(x, y, 0.7, 0, bisquare)
    # plt.plot(x, smoothed, "m-", label="loess (d = 0, f = 0.7)")
    smoothed = loess(x, y, 0.7, 1, bisquare)
    plt.plot(x, smoothed, "g--", label="loess (d = 1, f = 0.7)")
    smoothed = loess(x, y, 0.7, 2, bisquare)
    plt.plot(x, smoothed, "r-", label="loess (d = 2, f = 0.7)")
    smoothed = loess(x, y, 0.7, 3, bisquare)
    plt.plot(x, smoothed, "k--", label="loess (d = 3, f = 0.7)")
    plt.legend(loc=1)
    plt.show()
