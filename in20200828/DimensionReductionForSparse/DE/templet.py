import geatpy as ea
import numpy as np
from in20200828.DimensionReductionForSparse.util import help


class soea_SaNSDE_templet(ea.SoeaAlgorithm):

    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/SaNSDE'
        self.selFunc = 'rcs'  # 基向量的选择方式，采用随机补偿选择
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop):
        population = prophetPop
        NIND = population.sizes
        self.initialization()
        self.call_aimFunc(population)
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        while not self.terminated(population):
            # self.mutOper = help.F(0.5)
            if help.DE_choice(self.mutOper):
                """DE/rand/1/L"""
                r0 = ea.selecting(self.selFunc, population.FitnV, NIND)  # 得到基向量索引
                experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
                experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                          [r0])  # 变异
                experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
                self.call_aimFunc(experimentPop)  # 计算目标函数值
                tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
                tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
                population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群

            else:
                """DE/current-to-best/1/L"""
                r0 = np.arange(NIND)
                r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
                experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
                experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                  [r0, None, None, r_best, r0])  # 变异
                experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
                self.call_aimFunc(experimentPop)  # 计算目标函数值
                tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
                tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
                population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
