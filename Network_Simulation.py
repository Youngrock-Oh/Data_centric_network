import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar

start_time = time.time()

# location parameters
data_size = 500*10**3*8  # 500 KB
cycle_per_bit_edge = 6*10**9  # (cycles per slot)
cycle_per_bit_main_server = 5*10**15  # (cycles per slot)
T_slot = 100*10**-3  # time slot 100ms

avg_rate_edge = 1 / (data_size / cycle_per_bit_edge * T_slot)
rates_0 = np.array([100 + 30*i for i in range(9)])
rates_1 = np.array([600, 800, 1000, 1200])
rates_2 = np.array([1000, 1300])
rates = [rates_0, rates_1, rates_2]
loc_0 = [[-10 + 10*i, -10 + 10*j] for i in range(3) for j in range(3)]
loc_1 = [[-5, -5], [-5, 5], [5, -5], [5, 5]]
loc_2 = [[0, 5], [0, -5]]
locations = [loc_0, loc_1, loc_2]
# data type distribution and layer_dic
data_type_dist = np.array([0.4, 0.6])
layer_dic = {0: [0, 1, 2], 1: [0, 2]}
vol_dec = np.array([1, 0.9, 0.8])
delta = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(len(rates) - 1)]
for i in range(len(rates) - 1):
    delta[i] = ar.delay_return(locations[i], locations[i + 1])
initial_a = [np.ones((len(locations[i]), len(locations[i + 1])))/len(locations[i + 1]) for i in range(len(rates) - 1)]
delta_2 = delta + [np.zeros((2, 1))]
simulation_time = 100  # sec
t = 0
simulation_cases = {0: "Uniform routing", 1: "Barrier method", 2: "Projected gradient method"}
simulation_service_time = np.zeros(3)
for case_num, case in simulation_cases.items():
    if case_num == 0:
        res_A = initial_a
    elif case_num == 1:
        res_A = ar.barrier_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec)
    else:
        res_A = ar.grad_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec)

    A_2 = res_A + [np.zeros((2, 1))]
    cur_network = NC.Network(rates, data_type_dist, layer_dic, delta_2, A_2, vol_dec)
    cur_time = 0
    while cur_time < simulation_time:
        close_event_info = cur_network.update_time()
        close_service_time = close_event_info[0]
        sending_index = close_event_info[1]
        cur_network.update(close_service_time, sending_index)
        cur_time += close_service_time
        if cur_time >= 100*t:
            print(cur_time)
            t += 1
    simulation_service_time[case_num] = cur_network.Net_completion_time / cur_network.Num_completed_data / 10
    # rescaling
    print("%s: Simulation for %s " % (case_num, case))
    print('Simulation result: %s sec' % simulation_service_time[case_num])
print("--- %s seconds ---" % (time.time() - start_time))

