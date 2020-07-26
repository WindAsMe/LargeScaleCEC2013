import geatpy as ea
from before20200507.PreKnowledgeEvolution import benchmark
import numpy as np


class testProblem(ea.Problem):
    def __init__(self):
        M = 1
        Dim = 5
        name = 'Problem'
        maxormins = [1]
        varType = [1] * Dim
        lb = [-10] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varType, lb, ub, lbin, ubin)


    def aimFunc(self, pop):
        Vars = pop.Phen
        result = []
        for v in Vars:
            result.append([benchmark.Griewank(v)])
        pop.ObjV = np.array(result)
