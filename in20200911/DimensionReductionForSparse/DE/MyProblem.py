import geatpy as ea
import numpy as np


class MySimpleProblem(ea.Problem):
    def __init__(self, Dim, group, benchmark, scale_range, NIND, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        varTypes = [0] * Dim
        lb = [scale_range[0]] * Dim
        ub = [scale_range[1]] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.Dim = Dim
        self.NIND = NIND
        self.benchmark = benchmark
        self.group = group
        self.based_population = based_population
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        temp_Phen = []
        for i in range(self.NIND):
            temp_Phen.append(self.based_population)
        temp_Phen = np.array(temp_Phen)

        for element in self.group:
            temp_Phen[:, element] = pop.Phen[:, element]

        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p)])

        pop.Phen = temp_Phen
        pop.Chrom = temp_Phen
        pop.ObjV = np.array(result)


class MyComplexProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range, max_min):
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
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)
