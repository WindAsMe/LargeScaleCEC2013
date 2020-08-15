import geatpy as ea
import numpy as np


class MySimpleProblem(ea.Problem):
    def __init__(self, Dim, group, benchmark, scale_range, max_min):
        name = 'MyProblem'
        M = 1
        maxormins = [max_min]
        varTypes = [0] * Dim
        lb = [scale_range[0]] * Dim
        ub = [scale_range[1]] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.Dim = Dim
        self.benchmark = benchmark
        self.group = group
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        for i in range(len(pop.Phen)):
            for j in range(len(pop.Phen[i])):
                if j not in self.group:
                    pop.Phen[i][j] = 0
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)


class MyComplexProblem(ea.Problem):
    def __init__(self, Dim, function, benchmark, scale_range, max_min):
        name = 'MyProblem'
        M = 1
        maxormins = [max_min]
        varTypes = [0] * Dim
        lb = [scale_range[0]] * Dim
        ub = [scale_range[1]] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.Dim = Dim
        self.benchmark = benchmark
        self.function = function
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        pop.ObjV = self.function(pop.Phen, self.benchmark, self.Dim)
