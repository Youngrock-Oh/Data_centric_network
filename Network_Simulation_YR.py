import numpy as np
import Network_Classes as NC
import Analytic_res as ar
f1 = open("C:/Users/oe/PycharmProjects/ETRI_Data_centric_network/data_info_2.txt", 'w')
data_total = np.zeros((4, 0))
data_types = np.zeros((7, 3, 0))
# Network configuration
rates_0 = np.array([30 + 20 * (i // 5) for i in range(25)])
rates_1 = np.array([250 + 100 * (i // 3) for i in range(9)])
rates_2 = np.array([500, 600, 800, 900])
rates_3 = np.array([2900])
rates_c = [rates_0, rates_1, rates_2, rates_3]
loc_0 = [[-12 + 6 * i, -12 + 6 * j] for i in range(5) for j in range(5)]
loc_1 = [[-8 + 8 * i, -8 + 8 * j] for i in range(3) for j in range(3)]
loc_2 = [[-6 + 12 * i, -6 + 12 * j] for i in range(2) for j in range(2)]
loc_3 = [[0, 0]]
locations_c = [loc_0, loc_1, loc_2, loc_3]
# data and task configuration
data_task_c = {0: ["T1"], 1: ["T2"], 2: ["T1", "T2"], 3: ["T1", "T2", "T3"], 4: ["T3"], 5: ["T2", "T3"], 6: ["T1", "T3"]}
task_vol_dec_c = {"T1": 0.5, "T2": 0.5, "T3": 1}
data_type_dist_set = np.zeros((5, 7))
data_type_dist_set[0, :] = 1 / 7 * np.ones(7)
data_type_dist_set[1, :] = np.array([1/3, 1/6, 1/6, 1/12, 1/12, 1/12, 1/12])
data_type_dist_set[2, :] = np.array([2/3, 1/12, 1/12, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[3, :] = np.array([1/6, 1/3, 1/3, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[4, :] = np.array([1/6, 1/12, 1/12, 1/6, 1/6, 1/6, 1/6])

layer_task_c = {0: [], 1: ["T1"], 2: ["T2"], 3: ["T3"]}
res = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
vol_dec_input = res[1]
layer_dic_input = res[2]
data_bandwidth_efficiency = np.zeros((2, 0))
for case_index in range(len(data_type_dist_set)):
    print("Case %d" % case_index)
    data_type_dist_c = data_type_dist_set[case_index]
    result = NC.network_simulation(rates_c, locations_c, data_type_dist_c, data_task_c, task_vol_dec_c, layer_task_c)
    temp_data_info = data_type_dist_c.__str__() + "\n"
    temp_b_e = ar.bandwidth_efficiency_compare(data_type_dist_c, rates_0, layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(data_type_dist_c, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency = np.append(data_bandwidth_efficiency, temp_b_e_data, axis=1)
    data_total = np.append(data_total, result[0], axis=1)
    data_types = np.append(data_types, result[1], axis=2)
    f1.write(temp_data_info)

f1.close()
np.save('Bandwidth_efficiency_2.npy', data_bandwidth_efficiency)
np.save('Total_service_time_YR_2.npy', data_total)
np.save('Type_service_time_YR_2.npy', data_types)
