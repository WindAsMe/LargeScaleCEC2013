import geatpy as ea
import numpy as np


class MySimpleProblem(ea.Problem):
    def __init__(self, group, benchmark, scale_range, NIND, maxormin, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [maxormin]
        self.Dim = len(group)
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.NIND = NIND
        self.benchmark = benchmark
        self.group = group
        self.based_population = based_population
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        temp_Phen = []
        for i in range(self.NIND):
            temp_Phen.append(self.based_population)
        temp_Phen = np.array(temp_Phen)

        flag = 0
        for element in self.group:
            temp_Phen[:, element] = pop.Phen[:, flag]
            flag += 1

        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p)])

        pop.ObjV = np.array(result)


class MyProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range, maxormin):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.benchmark = benchmark
        maxormins = [maxormin]
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)
