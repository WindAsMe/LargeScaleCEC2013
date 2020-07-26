import numpy as np


def fitness_evaluation(Phen, function, intercept=0):
    fitness_value_list = []
    for P in Phen:
        fitness_value_list.append([function(P) - intercept])
    return np.array(fitness_value_list)


def aim_original_function(Phen, function):
    return fitness_evaluation(Phen, function, 0)

