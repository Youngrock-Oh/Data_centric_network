import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar

start_time = time.time()

# location parameters
data_size = 500*10**3*8  # 500 KB
cycle_per_bit = 1000
cycle_per_slot_edge = 6 * 10 ** 9  # (cycles per slot)
cycle_per_slot_main_server = 5 * 10 ** 15  # (cycles per slot)
T_slot = 100*10**-3  # time slot 100ms

avg_rate_main_server = 1 / (data_size * cycle_per_bit / cycle_per_slot_main_server * T_slot)
rates_0 = np.array([23 + i for i in range(25)])
rates_1 = np.array([210 + 10 * i for i in range(9)])
rates_1[0] = 150
rates_1[1] = 150
rates_2 = np.array([300, 350, 350, 400])
rates_3 = np.array([1000])
rates = [rates_0, rates_1, rates_2, rates_3]
loc_0 = [[-12 + 6*i, -12 + 6*j] for i in range(5) for j in range(5)]
loc_1 = [[-8 + 8*i, -8 + 8*j] for i in range(3) for j in range(3)]
loc_2 = [[-6 + 12*i, -6 + 12*j] for i in range(2) for j in range(2)]
loc_3 = [[0, 0]]
locations = [loc_0, loc_1, loc_2, loc_3]
# data type distribution and layer_dic
data_type_dist = np.array([0.3, 0.3, 0.2, 0.2])
vol_dec = np.array([1, 0.8, 0.8, 1])
layer_dic = {0: [0, 1], 1: [0, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3]}
delta = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(len(rates) - 1)]
for i in range(len(rates) - 1):
    delta[i] = ar.delay_return(locations[i], locations[i + 1])
initial_a = [np.ones((len(locations[i]), len(locations[i + 1])))/len(locations[i + 1]) for i in range(len(rates) - 1)]
delta_2 = delta + [np.zeros((4, 1))]
simulation_time = 20  # sec
t = 0
res_A = ar.barrier_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec)
print("--- %s seconds ---" % (time.time() - start_time))

