import numpy as np
from Root_Histo_maker import ROOT_Histo_Maker

s = 0
mc_data = []
weight = []
for i in range(4):
    mc_data.append(np.random.rand(100*int((i/2 + 1))).ravel())
    weight.append(np.ones(100*int((i/2 + 1))))
    s += int(100*(i/2 + 1))

data = np.random.rand( s)


ROOT_Histo_Maker(mc_data, weight, ["a", "b" , "c", "d"], data, bin_max=1, y_max=100)