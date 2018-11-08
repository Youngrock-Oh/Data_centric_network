import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar


def network_simulation(data_type_dist, layer_dic):
    start_time = time.time()
    result1 = np.array([])
    result2 = np.array([[], [], [], [], [], [], []])
    rates_0 = np.array([30 + 20 * (i // 5) for i in range(25)])
    rates_1 = np.array([250 + 100 * (i // 3) for i in range(9)])
    rates_2 = np.array([500, 600, 800, 900])
    rates_3 = np.array([2000])
    rates = [rates_0, rates_1, rates_2, rates_3]
    loc_0 = [[-12 + 6 * i, -12 + 6 * j] for i in range(5) for j in range(5)]
    loc_1 = [[-8 + 8 * i, -8 + 8 * j] for i in range(3) for j in range(3)]
    loc_2 = [[-6 + 12 * i, -6 + 12 * j] for i in range(2) for j in range(2)]
    loc_3 = [[0, 0]]
    locations = [loc_0, loc_1, loc_2, loc_3]
    # data type distribution and layer_dic
    # data_type_dist = 1 / 7 * np.ones(7)
    vol_dec = np.array([1, 0.8, 0.8, 1])
    # layer_dic = {0: [0, 1], 1: [0, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 3], 5: [0, 2, 3], 6: [0, 1, 3]}
    delta = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(len(rates) - 1)]
    for i in range(len(rates) - 1):
        delta[i] = ar.delay_return(locations[i], locations[i + 1])
    initial_a = [np.ones((len(locations[i]), len(locations[i + 1]))) / len(locations[i + 1]) for i in
                 range(len(rates) - 1)]
    delta_2 = delta + [np.zeros((4, 1))]
    simulation_time = 1000  # sec
    t = 0
    simulation_cases = {0: "Uniform routing", 1: "Barrier method", 2: "Projected gradient method", 3: "Legacy"}
    simulation_service_time = np.zeros(4)
    res_A = [[], [], [], []]
    print("---------Data type distribution: ", data_type_dist)
    print("---------Layer dic: ", layer_dic)
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
            if cur_time >= 100 * t:
                print(cur_time)
                t += 1
        simulation_service_time[case_num] = cur_network.avg_completion_time
        print("%s: Simulation for %s " % (case_num, case))
        print('Total avg processing completion time: %s sec' % simulation_service_time[case_num])
        print('Completion time for each data type: ', cur_network.avg_completion_time_types)
        result1 = np.append(result1, simulation_service_time[case_num])
        if case_num != 3:
            result2 = np.append(result2, cur_network.avg_completion_time_types.reshape((7, 1)), axis = 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    result1 = result1.reshape((4, 1))
    result2 = result2.reshape((7, 3, 1))
    res = [result1, result2]
    return res


f1 = open("C:/Users/oe/PycharmProjects/ETRI_Data_centric_network/data_info.txt", 'w')

data_total = np.zeros((4, 0))
data_types = np.zeros((7, 3, 0))
data_type_dist_set = np.zeros((8, 7))
data_type_dist_set[0, :] = 1 / 7 * np.ones(7)
data_type_dist_set[1, :] = np.array([1/3, 1/6, 1/6, 1/12, 1/12, 1/12, 1/12])
data_type_dist_set[2, :] = np.array([1/2, 1/8, 1/8, 1/16, 1/16, 1/16, 1/16])
data_type_dist_set[3, :] = np.array([1/4, 1/4, 1/4, 1/16, 1/16, 1/16, 1/16])
data_type_dist_set[4, :] = np.array([1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
data_type_dist_set[5, :] = np.array([2/3, 1/12, 1/12, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[6, :] = np.array([1/6, 1/3, 1/3, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[7, :] = np.array([1/6, 1/12, 1/12, 1/6, 1/6, 1/6, 1/6])

layer_dic_1 = {0: [0, 1], 1: [0, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 3], 5: [0, 2, 3], 6: [0, 1, 3]}
layer_dic_set = [layer_dic_1 ]
for data_type_dist in data_type_dist_set:
    for layer_dic in layer_dic_set:
        result = network_simulation(data_type_dist, layer_dic)
        temp_data_type_dist = data_type_dist.__str__() + "\n"
        temp_layer_dic = layer_dic.__str__() + "\n"
        temp_data_info = temp_data_type_dist + temp_layer_dic
        temp_type_service_time = result[1]
        data_total = np.append(data_total, result[0], axis = 1)
        data_types = np.append(data_types, result[1], axis = 2)
        f1.write(temp_data_info)

f1.close()
np.save('Total_service_time_YR.npy', data_total)
np.save('Type_service_time_YR.npy', data_types)
