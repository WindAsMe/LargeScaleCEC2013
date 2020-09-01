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

    def run(self, prophetPop, Max_iteration):
        population = prophetPop
        NIND = population.sizes
        self.initialization()
        self.call_aimFunc(population)
        p = 0.5
        fp = 0.5
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        current_best_Fitness = population.ObjV[:, 0].min()

        flag = 0

        ns1 = 0
        ns2 = 0
        nf1 = 0
        nf2 = 0

        CRrec = []
        CR = 0.5
        while not self.terminated(population):
            self.mutOper = ea.Mutde(F=help.F(fp))
            temp_CR = help.CRm(CR)
            self.recOper = ea.Xovexp(XOVR=help.CRm(temp_CR), Half=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
            # update the p and CRm
            if flag + 1 == Max_iteration / 3 or flag + 1 == Max_iteration * 2 / 3:
                if ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2) > 0:
                    p = ns1 * (ns2 + nf2) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2))
                    fp = p
                if len(CRrec) > 0:
                    CR = sum(CRrec) / len(CRrec)
                    CRrec.clear()

            if help.DE_choice(p):
                """DE/rand/1/L"""
                r0 = ea.selecting(self.selFunc, population.FitnV, NIND)  # 得到基向量索引
                experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
                experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                          [r0])  # 变异
                experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
                self.call_aimFunc(experimentPop)  # 计算目标函数值
                tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
                tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
                temp_population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
                temp_ns1, temp_nf1 = help.same_elements_in_population(population, temp_population)
                ns1 += temp_ns1
                nf1 += temp_nf1
                population = temp_population
                if population.ObjV.min() < current_best_Fitness:
                    CRrec.append(temp_CR)
                    current_best_Fitness = population.ObjV.min()

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
                temp_population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
                temp_ns2, temp_nf2 = help.same_elements_in_population(population, temp_population)
                ns2 += temp_ns2
                nf2 += temp_nf2
                population = temp_population

                if population.ObjV.min() < current_best_Fitness:
                    CRrec.append(temp_CR)
                    current_best_Fitness = population.ObjV.min()

            flag += 1
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
