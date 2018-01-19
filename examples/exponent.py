import numpy as np
from matplotlib import pyplot as plt

from smoothie import loess

x = np.linspace(0, 5*np.pi, 100)
signal = 10 * np.exp(-x/3)
noise = np.random.rand(len(signal)) * 2 - 1
y = signal + noise
plt.title("LOESS (3 iterations)")
plt.plot(x, signal, "b-", lw=2, label="signal")
plt.plot(x, y, "b+", label="signal+noise")
smoothed = loess(x, y, 0.7, 1, bisquare)
plt.plot(x, smoothed, "g--", label="loess (d = 1, f = 0.7)")
smoothed = loess(x, y, 0.7, 2, bisquare)
plt.plot(x, smoothed, "r-", label="loess (d = 2, f = 0.7)")
smoothed = loess(x, y, 0.7, 3, bisquare)
plt.plot(x, smoothed, "k--", label="loess (d = 3, f = 0.7)")
plt.legend(loc=1)
plt.show()
