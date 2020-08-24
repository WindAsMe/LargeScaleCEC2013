from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d


def sign(x):
    s = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if x[i][j] < 0:
                temp.append(-1)
            elif x[i][j] == 0:
                temp.append(0)
            else:
                temp.append(1)
        s.append(temp)
    return np.array(s)


def c1_f(x):
    c1 = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if x[i][j] > 0:
                temp.append(10)
            else:
                temp.append(5.5)
        c1.append(temp)
    return np.array(c1)


def c2_f(x):
    c2 = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if x[i][j] > 0:
                temp.append(7.9)
            else:
                temp.append(3.1)
        c2.append(temp)
    return np.array(c2)


def x_bar_f(x):
    x_bar = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if x[i][j] != 0:
                temp.append(np.log2(np.abs(x[i][j])))
            else:
                temp.append(0)
        x_bar.append(temp)
    return np.array(x_bar)


def ShiftedElliptic(x, y):
    x = sign(x) * np.exp(x_bar_f(x) + 0.049*(np.sin(c1_f(x)*x_bar_f(x))) + np.sin(c2_f(x)*x_bar_f(x)))
    y = sign(y) * np.exp(x_bar_f(y) + 0.049 * (np.sin(c1_f(y) * x_bar_f(y))) + np.sin(c2_f(y) * x_bar_f(y)))
    return 10**6*x**2 + 10**6*y**2


def draw(f):
    figure = plt.figure()
    ax = figure.gca(projection="3d")
    x1 = np.linspace(-10, 10, 2000)
    y1 = np.linspace(-10, 10, 2000)
    x, y = np.meshgrid(x1, y1)
    # print(x)
    z = f(x, y)
    ax.plot_surface(x, y, z, cmap="rainbow")
    plt.show()


draw(ShiftedElliptic)