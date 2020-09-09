from in20200828.DimensionReductionForSparse.main_interface.optimization_f.Lasso import Lasso_f
from in20200828.DimensionReductionForSparse.main_interface.optimization_f.Normal import Normal_f
from in20200828.DimensionReductionForSparse.main_interface.optimization_f.One import One_f
from in20200828.DimensionReductionForSparse.main_interface.optimization_f.Random import Random_f

# for i in range(10):
#     Lasso_f.f()

for i in range(7):
    Normal_f.f()

for i in range(8):
    One_f.f()

for i in range(9):
    Random_f.f()
