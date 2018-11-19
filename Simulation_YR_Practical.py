import numpy as np
import Network_Classes as NC
import Analytic_res as ar
f1 = open("C:/Users/oe/PycharmProjects/ETRI_Data_centric_network/data_info_practical_3.txt", 'w')
data_total = np.zeros((4, 0))
# data and task configuration
data_task_c = {0: ["T1"], 1: ["T2"], 2: ["T3"], 3: ["T4"]}
task_vol_dec_c = {"T1": 1 / 2, "T2": 1 / 2, "T3": 1 / 2, "T4": 1 / 2}
data_type_dist = np.array([1/4, 1/4, 1/4, 1/4])
# Network configuration
rates_0 = np.array([30 + 10 * (i // 5) for i in range(25)])
rates_1 = np.array([250 + 50 * (i // 4) for i in range(16)])
rates_2 = np.array([350 + 100 * (i // 3) for i in range(9)])
rates_3 = np.array([700, 800, 900, 1000])
rates_4 = np.array([2000])
rates_input = [rates_0, rates_1, rates_2, rates_3, rates_4]

loc_0 = [[-12 + 6 * i, -12 + 6 * j] for i in range(5) for j in range(5)]
loc_1 = [[-9 + 6 * i, -9 + 6 * j] for i in range(4) for j in range(4)]
loc_2 = [[-8 + 8 * i, -8 + 8 * j] for i in range(3) for j in range(3)]
loc_3 = [[-6 + 12 * i, -6 + 12 * j] for i in range(2) for j in range(2)]
loc_4 = [[0, 0]]
locations_input = [loc_0, loc_1, loc_2, loc_3, loc_4]
simul_case_num = 4
data_bandwidth_efficiency = np.zeros((2, 0))
for case_index in range(simul_case_num):
    print("Case %d." % case_index)
    if case_index == 0:
        layer_task_c = {0: [], 1: [], 2: [], 3: [], 4: ["T1", "T2", "T3", "T4"]}
    elif case_index == 1:
        layer_task_c = {0: [], 1: [], 2: ["T1", "T2"], 3: [], 4: ["T3", "T4"]}
    elif case_index == 2:
        layer_task_c = {0: [], 1: ["T1"], 2: ["T2"], 3: [], 4: ["T3", "T4"]}
    else:
        layer_task_c = {0: [], 1: ["T1"], 2: ["T2"], 3: ["T3"], 4: ["T4"]}

    res = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    rate_factor_input = res[0]
    vol_dec_input = res[1]
    layer_dic_input = res[2]
    result = NC.network_simulation(rates_input, locations_input, data_type_dist, rate_factor_input, layer_dic_input)
    temp_data_info = layer_task_c.__str__() + "\n"
    temp_b_e = ar.bandwidth_efficiency(vol_dec_input, data_type_dist, layer_dic_input, rates_0)
    temp_metric = ar.avg_sum_required_layer(data_type_dist, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency = np.append(data_bandwidth_efficiency, temp_b_e_data, axis=1)
    data_total = np.append(data_total, result[0], axis=1)
    f1.write(temp_data_info)
f1.write(rates_input.__str__() + "\n")
f1.write(data_task_c.__str__())
f1.close()
np.save('Bandwidth_efficiency_practical_3.npy', data_bandwidth_efficiency)
np.save('Total_service_time_YR_practical_3.npy', data_total)
