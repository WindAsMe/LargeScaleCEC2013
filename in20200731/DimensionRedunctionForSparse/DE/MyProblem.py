import geatpy as ea


class MySimpleProblem(ea.Problem):
    def __init__(self, Dim, function, group, benchmark, scale_range, max_min):
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
        self.group = group
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        pop.ObjV = self.function(pop.Phen, self.benchmark, self.Dim, self.group)


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
