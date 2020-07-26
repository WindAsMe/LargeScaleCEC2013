import copy
import random
import matplotlib.pyplot as plt
import numpy as np


def get_chrom(chrom, best_index):
    new_chrom = []
    for i in best_index:
        new_chrom.append(chrom[i])
    return np.array(new_chrom)


def create_points(best_index, number, scale_factor=0.7):
    best_indexes = [best_index]
    for i in range(number - 1):
        good_index = copy.deepcopy(best_index)
        for i in range(len(good_index)):
            good_index[i] += random.random() * scale_factor
        best_indexes.append(good_index)
    return best_indexes


def draw(points, function):
    step = 0.01
    x = np.arange(-10, 10, step)
    y = np.arange(-10, 10, step)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    plt.contour(X, Y, Z)
    for i in range(len(points)):
        plt.plot(points[i][0], points[i][1], '.', color='black')
    plt.show()


def draw_elite_1(elite, points, function):
    step = 0.01
    x = np.arange(-10, 10, step)
    y = np.arange(-10, 10, step)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    plt.contour(X, Y, Z)
    for i in range(len(points)):
        plt.plot(points[i][0], points[i][1], '.', color='black')
    plt.plot(elite[0], elite[1], '.', color='red')
    plt.show()


def draw_elite_10(elite, elites, points, function):
    step = 0.01
    x = np.arange(-10, 10, step)
    y = np.arange(-10, 10, step)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    plt.contour(X, Y, Z)
    for i in range(len(points)):
        if [points[i][0], points[i][1]] in elites:
            plt.plot(points[i][0], points[i][1], '.', color='green')
        else:
            plt.plot(points[i][0], points[i][1], '.', color='black')
    plt.plot(elite[0], elite[1], '.', color='red')
    plt.show()


def find_n_best(num_list, topk=50):
    tmp_list = copy.deepcopy(num_list)
    tmp_list.sort()
    min_num_index = [num_list.index(one) for one in tmp_list[:topk]]
    return min_num_index


def matrix2list(m):
    result = []
    for l in m:
        result.append(l[0])
    return result


def get_chrom_10(points_index, aim, reg, best_index):
    points_index = copy.deepcopy(points_index)
    matrix_combination = np.append(points_index, best_index, axis=0)
    first_evaluate = aim(matrix_combination, reg)
    first_evaluate_list = matrix2list(first_evaluate)
    points_index = find_n_best(first_evaluate_list, 10)
    chrom = get_chrom(matrix_combination, points_index)
    return chrom
