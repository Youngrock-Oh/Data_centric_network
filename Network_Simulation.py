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
rates_0 = np.array([30 + 20*(i//5) for i in range(25)])
rates_1 = np.array([250+100* (i//3) for i in range(9)])
rates_2 = np.array([500, 600, 800, 900])
rates_3 = np.array([1500])
rates = [rates_0, rates_1, rates_2, rates_3]
loc_0 = [[-12 + 6*i, -12 + 6*j] for i in range(5) for j in range(5)]
loc_1 = [[-8 + 8*i, -8 + 8*j] for i in range(3) for j in range(3)]
loc_2 = [[-6 + 12*i, -6 + 12*j] for i in range(2) for j in range(2)]
loc_3 = [[0, 0]]
locations = [loc_0, loc_1, loc_2, loc_3]
# data type distribution and layer_dic
data_type_dist = 1/7*np.ones(7)
vol_dec = np.array([1, 0.8, 0.8, 1])
layer_dic = {0: [0, 1], 1: [0, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 3], 5: [0, 2, 3], 6: [0,1,3]}
delta = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(len(rates) - 1)]
for i in range(len(rates) - 1):
    delta[i] = ar.delay_return(locations[i], locations[i + 1])
initial_a = [np.ones((len(locations[i]), len(locations[i + 1])))/len(locations[i + 1]) for i in range(len(rates) - 1)]
delta_2 = delta + [np.zeros((4, 1))]
simulation_time = 10  # sec
t = 0
simulation_cases = {0: "Uniform routing", 1: "Barrier method", 2: "Projected gradient method", 3: "Legacy"}
simulation_service_time = np.zeros(4)
res_A = [[], [], [], []]
for case_num, case in simulation_cases.items():
    if case_num == 0:
        res_A[case_num] = initial_a
    elif case_num == 1:
        res_A[case_num] = ar.barrier_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec)
    elif case_num == 2:
        res_A[case_num] = ar.grad_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec)
    else:
        res_A[case_num] = initial_a
        data_type_dist = np.array([1])
        vol_dec = np.array([1, 1, 1, 1])
        layer_dic = {0: [0, 3]}

    A_2 = res_A[case_num] + [np.zeros((4, 1))]
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
    simulation_service_time[case_num] = cur_network.Net_completion_time / cur_network.Num_completed_data
    # rescaling
    print("%s: Simulation for %s " % (case_num, case))
    print('Simulation result: %s sec' % simulation_service_time[case_num])
    # print(res_A[case_num])
print("--- %s seconds ---" % (time.time() - start_time))

