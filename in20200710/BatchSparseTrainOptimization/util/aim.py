import numpy as np


def fitness_evaluation(Phen, function, func_num):
    fitness_value_list = []
    for P in Phen:
        fitness_value_list.append([function(P, func_num)])
    return np.array(fitness_value_list)
