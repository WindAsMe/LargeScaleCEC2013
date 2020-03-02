import math
import numpy as np


# f=x[0]^2+3x[1]^2+2*x[1]*x[2]
def f(x):
    return x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[1] * x[2]


# f1=f+linear noise
def f1(x, a, b):
    return x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[1] * x[2] + a * x[3] ** 2 + b * x[4]


# f1=f+Gaussian noise
def f2(x, a, b):
    return x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[1] * x[2] + a * 20 * Gaussian(x[3], x[4]) + b * 10


def Gaussian(x, y):
    return (1 / 2 * math.pi * 3 ** 2) * np.exp(-(x ** 2 + y ** 2) / 2 * 3 ** 2)
