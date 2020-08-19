import geatpy as ea


class MySimpleProblem(ea.Problem):
    def __init__(self, func_num, Dim, benchmark_function, evaluate_function, group, max_min):
        name = 'MyProblem'
        M = 1
        maxormins = [max_min]
        varTypes = [0] * Dim
        lb = [-10] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.func_num = func_num
        self.Dim = Dim
        self.benchmark_function = benchmark_function
        self.evaluate_function = evaluate_function
        self.group = group

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        # no relation variables set to 0
        for i in range(len(pop.Phen)):
            for j in range(len(pop.Phen[i])):
                if j not in self.group:
                    pop.Phen[i][j] = 0
        pop.ObjV = self.evaluate_function(pop.Phen, self.benchmark_function, self.func_num)

