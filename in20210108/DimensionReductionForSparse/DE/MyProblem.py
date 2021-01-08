import geatpy as ea
import numpy as np


class CCDE_Problem(ea.Problem):
    def __init__(self, group, benchmark, scale_range, NIND, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
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


class Block_Problem(ea.Problem):
    def __init__(self, group, benchmark, up, down, NIND, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = len(group)
        varTypes = [0] * self.Dim
        sub_lb = []
        sub_ub = []
        for e in group:
            sub_lb.append(down[e])
            sub_ub.append(up[e])
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.NIND = NIND
        self.benchmark = benchmark
        self.group = group
        self.based_population = based_population

        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, sub_lb, sub_ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        temp_Phen = []
        for i in range(self.NIND):
            temp_Phen.append(self.based_population)
        temp_Phen = np.array(temp_Phen)

        for element in self.group:
            temp_Phen[:, element] = pop.Phen[:, self.group.index(element)]

        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)


