import numpy as np
import geatpy as ea
from before20200507.PureCombinationRegression import pureCombination


def OptimizationForOriginal(aim, name, maxormin, dimension):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)

    range_item = []
    border_item = []
    varType_item = []
    code_item = []
    precision_item = []
    scale_item = []
    for i in range(0, dimension):
        range_item.append([-10, 10])
        border_item.append([1, 1])
        varType_item.append(0)
        code_item.append(1)
        precision_item.append(6)
        scale_item.append(0)
    ranges = np.vstack(range_item).T
    borders = np.vstack(border_item).T
    varTypes = np.array(varType_item)
    Encoding = 'BG'
    codes = code_item
    precisions = precision_item
    scales = scale_item
    FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

    NIND = 20
    MAXGEN = 1000
    maxormins = [maxormin]
    selectStyle = 'sus'
    recStyle = 'xovdp'
    mutStyle = 'mutbin'
    Lind = int(np.sum(FieldD[0, :]))
    pc = 0.9
    pm = 1 / Lind
    obj_trace = np.zeros((MAXGEN, 2))
    var_trace = np.zeros((MAXGEN, Lind))

    # start_time = time.time()
    Chrom = ea.crtpc(Encoding, NIND, FieldD)
    variable = ea.bs2real(Chrom, FieldD)
    ObjV = aim(variable, dimension)
    best_ind = np.argmin(ObjV)
    for gen in range(MAXGEN):
        FitnV = ea.ranking(maxormins * ObjV)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        Phen = ea.bs2real(Chrom, FieldD)
        ObjV = aim(Phen, dimension)
        best_ind = np.argmin(ObjV)
        obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]
        obj_trace[gen, 1] = ObjV[best_ind]
        var_trace[gen, :] = Chrom[best_ind, :]
    # end_time = time.time()
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])

    best_gen = np.argmin(obj_trace[:, [1]])
    print("The best Individual: ", obj_trace[best_gen, 1])
    print('--------------------------------------')
    # variable = ea.bs2real(var_trace[[best_gen], :], FieldD)
    # print('For the optimization factors:')
    index = []
    for i in range(variable.shape[1]):
        # print('x' + str(i) + '=', variable[0, i])
        index.append(variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1], index


def OptimizationForRegression(aim, name, reg, maxormin, dimension):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)

    range_item = []
    border_item = []
    varType_item = []
    code_item = []
    precision_item = []
    scale_item = []
    for i in range(0, dimension):
        range_item.append([-10, 10])
        border_item.append([1, 1])
        varType_item.append(0)
        code_item.append(1)
        precision_item.append(6)
        scale_item.append(0)
    ranges = np.vstack(range_item).T
    borders = np.vstack(border_item).T
    varTypes = np.array(varType_item)
    Encoding = 'BG'
    codes = code_item
    precisions = precision_item
    scales = scale_item
    FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

    NIND = 20
    MAXGEN = 1000
    maxormins = [maxormin]
    selectStyle = 'sus'
    recStyle = 'xovdp'
    mutStyle = 'mutbin'
    Lind = int(np.sum(FieldD[0, :]))
    pc = 0.9
    pm = 1 / Lind
    obj_trace = np.zeros((MAXGEN, 2))
    var_trace = np.zeros((MAXGEN, Lind))

    # start_time = time.time()
    Chrom = ea.crtpc(Encoding, NIND, FieldD)
    variable = ea.bs2real(Chrom, FieldD)
    # The 4 is the degree
    ObjV = aim(variable, reg, pureCombination.pure_combination, 4)
    best_ind = np.argmin(ObjV)
    for gen in range(MAXGEN):
        FitnV = ea.ranking(maxormins * ObjV)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND-1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        Phen = ea.bs2real(Chrom, FieldD)
        ObjV = aim(Phen, reg, pureCombination.pure_combination, 4)
        best_ind = np.argmin(ObjV)
        obj_trace[gen, 0] = np.sum(ObjV)/ObjV.shape[0]
        obj_trace[gen, 1] = ObjV[best_ind]
        var_trace[gen, :] = Chrom[best_ind, :]
    # end_time = time.time()
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])

    best_gen = np.argmin(obj_trace[:, [1]])
    print('The best Individual：', obj_trace[best_gen, 1])
    print('--------------------------------------')
    # variable = ea.bs2real(var_trace[[best_gen], :], FieldD)
    # print('For the optimization factors:')
    index = []
    for i in range(variable.shape[1]):
        # print('x' + str(i) + '=', variable[0, i])
        index.append(variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1], index
