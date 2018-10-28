import numpy as np
import Analytic_res as ar
arrival_rates = np.array([1, 2])
service_rates = np.array([2, 3])
cur_layer_index = 1
layer_dic = {0: [0, 1, 2], 1: [0, 2]}
data_dist = np.array([0.4, 0.6])
vol_dec = np.array([1, 0.8, 0.8])


temp = ar.effective_rates(arrival_rates, service_rates, cur_layer_index, layer_dic, data_dist, vol_dec)
print(temp)