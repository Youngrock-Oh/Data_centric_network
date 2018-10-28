import numpy as np
import Analytic_res as ar


rates_0 = np.array([100 + 30*i for i in range(9)])
rates_1 = np.array([600, 800, 1000, 1200])
rates_2 = np.array([1000, 1300])
rates = [rates_0, rates_1, rates_2]
loc_0 = [[-10 + 10*i, -10 + 10*j] for i in range(3) for j in range(3)]
loc_1 = [[-5, -5], [-5, 5], [5, -5], [5, 5]]
loc_2 = [[0, 5], [0, -5]]
locations = [loc_0, loc_1, loc_2]

delta = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(len(rates) - 1)]
for i in range(len(rates) - 1):
    delta[i] = ar.delay_return(locations[i], locations[i + 1])
initial_a = [np.ones((len(locations[i]), len(locations[i + 1])))/len(locations[i + 1]) for i in range(len(rates) - 1)]
# data type distribution and layer_dic
data_type_dist = np.array([0.4, 0.6])
layer_dic = {0: [0, 1, 2], 1:[0, 2]}
vol_dec = np.array([1, 1, 1])
res = ar.grad_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec)