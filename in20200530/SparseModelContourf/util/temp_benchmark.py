import math
import numpy as np


def Griewank(x, y):
    part1 = x ** 2 / 4000 + y ** 2 / 4000
    part2 = np.cos(x / np.sqrt(x)) * np.cos(y / np.sqrt(y))

    return part1 - part2 + 1


def Schwefel(x, y):

    return 418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))


def Rosenbrock(x, y):

    return 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2


def Rastrigin(x, y):
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + 10 + y ** 2 - 10 * np.cos(2 * np.pi * y) + 10


def Ackley(x, y):
    part1 = x ** 2 + y ** 2
    part2 = np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)

    part1 = -0.2 * np.sqrt(1 / 2 * part1)
    part2 = 1 / 2 * part2
    return -20 * np.exp(part1) - np.exp(part2) + 20 + math.e
