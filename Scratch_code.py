from Analytic_res import *
cur_layer_index = 2
layer_dic = {0: [0, 1, 2], 1: [0, 2, 3]}
vol_dec = np.zeros((2, 3))
vol_dec[0, :] = np.array([1, 1/2, 1/2])
vol_dec[1, :] = np.array([1, 1/2, 1/3])
arrival_rates = np.array([10, 5])
service_rates = np.array([10, 20])
data_dist = np.array([1/2, 1/2])
print(effective_rates(arrival_rates, service_rates, cur_layer_index, layer_dic, data_dist, vol_dec))