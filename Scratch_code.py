from Analytic_res import *
source_rates = np.array([5, 5])
vol_dec = [1, 0.5, 1]
layer_dic = {0: [0, 1, 2], 1: [0, 1]}
data_type_dist = np.ones(2) / 2
res = bandwidth_efficiency(vol_dec, data_type_dist, layer_dic, source_rates)
print(res)