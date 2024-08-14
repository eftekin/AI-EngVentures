# Turning our function into an array
from math import sin

import matplotlib.pyplot as plt
import numpy as np

# our change in x value
dx = 0.01


def f(x):
    return sin(x)


sin_y = [f(x) for x in np.arange(0, 20, dx)]
sin_x = [x for x in np.arange(0, 20, dx)]

# Define your derivative here
sin_deriv = np.gradient(sin_y, dx)

plt.plot(sin_x, sin_y)
plt.plot(sin_x, sin_deriv)
plt.show()
