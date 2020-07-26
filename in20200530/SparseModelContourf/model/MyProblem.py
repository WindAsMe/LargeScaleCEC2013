import geatpy as ea


class MyProblem(ea.Problem):
    def __init__(self, Dim, function, reg, max_min):
        name = 'MyProblem'
        M = 1
        maxormins = [max_min]
        varTypes = [0] * Dim
        lb = [-10] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.function = function
        self.reg = reg
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        pop.ObjV = self.function(pop.Phen, self.reg)
