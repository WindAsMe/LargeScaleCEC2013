import geatpy as ea
from before20200507.PreKnowledgeEvolution import TestProblem

if __name__ == '__main__':
    problem = TestProblem.testProblem()
    Encoding = 'BG'
    NIND = 50
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    algorithm = ea.moea_awGA_templet(problem, population)
    algorithm.MAXGEN = 50
    prophetPop = algorithm.run()
