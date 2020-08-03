import geatpy as ea
from in20200731.DimensionReductionForSparse.parameters import global_optimum, threshold


class MySoea_DE_rand_1_L_templet(ea.soea_DE_rand_1_L_templet):

    def __init__(self, problem, population):
        ea.soea_DE_rand_1_L_templet.__init__(self, problem, population)  # 先调用父类构造方法

    def terminated(self, population):

        """
        描述:
            该函数用于判断是否应该终止进化，population为传入的种群
            重写终止进化方法，得到最优目标值
        """
        self.stat(population)  # 分析记录当代种群的数据
        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if self.currentGen + 1 >= self.MAXGEN or self.forgetCount >= self.maxForgetCount \
                or self.trappedCount >= self.maxTrappedCount or isArriveThreshold(self.preObjV, global_optimum, threshold):
            return True
        else:
            self.preObjV = self.obj_trace[self.currentGen, 1]  # 更新“前代最优目标函数值记录器”
            self.currentGen += 1  # 进化代数+1
            return False


def isArriveThreshold(preObjV, global_optimum, threshold):
    return global_optimum - threshold <= preObjV <= global_optimum + threshold
