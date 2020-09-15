import numpy as np
from in20200913.DimensionReductionForSparse.util import help
from scipy.stats import friedmanchisquare


def ave(array):
    result = []
    for i in range(len(array[0])):
        result.append(np.mean(array[:, i]))
    return result


def final(LASSO, Normal, One, Random):
    LASSO_final = LASSO[:, len(LASSO[0])-1]
    Normal_final = Normal[:, len(Normal[0]) - 1]
    One_final = One[:, len(One[0]) - 1]
    Random_final = Random[:, len(Random[0]) - 1]

    print('LASSO final: ', '%e' % np.mean(LASSO_final), '±', '%e' % np.std(LASSO_final, ddof=1))
    print('Normal final: ', '%e' % np.mean(Normal_final), '±', '%e' % np.std(Normal_final, ddof=1))
    print('One final: ', '%e' % np.mean(One_final), '±', '%e' % np.std(One_final, ddof=1))
    print('Random final: ', '%e' % np.mean(Random_final), '±', '%e' % np.std(Random_final, ddof=1))
    help.write_final(np.mean(LASSO_final), np.mean(Normal_final), np.mean(One_final), np.mean(Random_final))


if __name__ == '__main__':
    f = 'f5'
    LASSO = []

    Normal = []

    One = []

    Random = []

    LASSO = np.array(LASSO)
    Normal = np.array(Normal)
    One = np.array(One)
    Random = np.array(Random)
    final(LASSO, Normal, One, Random)

    LASSO_ave = ave(LASSO)
    Normal_ave = ave(Normal)
    One_ave = ave(One)
    Random_ave = ave(Random)

    x = np.linspace(0, 3000000, 100)
    help.draw_summary(x, LASSO_ave, Normal_ave, One_ave, Random_ave, f)
    print(x)
    generations = [99]
    for g in generations:
        print(x[g], ': ', friedmanchisquare(LASSO[:, g], Normal[:, g], One[:, g], Random[:, g]))

