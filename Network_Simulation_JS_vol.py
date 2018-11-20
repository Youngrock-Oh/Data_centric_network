import numpy as np
import Network_Classes as NC
import Analytic_res as ar

# Rate
rates_0 = np.array([30 + 20 * (i // 5) for i in range(25)])
rates_1 = np.array([250 + 100 * (i // 3) for i in range(9)])
rates_2 = np.array([500, 600, 800, 900])
rates_3 = np.array([2900])
rates = [rates_0, rates_1, rates_2, rates_3]

# Volume decrement
task_vol_dec_c = {"T1": 0.5, "T2": 0.5, "T3": 1}
vol_dec_c_1 = {"T1": 0.25, "T2": 1, "T3": 1}
vol_dec_c_2 = {"T1": 1, "T2": 0.25, "T3": 1}
vol_dec_c_3 = {"T1": 0.1, "T2": 0.1, "T3": 1}
vol_dec_c_4 = {"T1": 0.01, "T2": 1, "T3": 1}
vol_dec_c_5 = {"T1": 1, "T2": 0.01, "T3": 1}
vol_dec_set = [task_vol_dec_c, vol_dec_c_1, vol_dec_c_2, vol_dec_c_3, vol_dec_c_4, vol_dec_c_5]

# data type distribution
cur_data_type_dist = 1/7*np.ones(7)
layer_task_c = {0: [], 1: ["T1"], 2: ["T2"], 3: ["T3"]}
data_task_c = {0: ["T1"], 1: ["T2"], 2: ["T1", "T2"], 3: ["T1", "T2", "T3"], 4: ["T3"], 5: ["T2", "T3"], 6: ["T1", "T3"]}

# Location
loc_0 = [[float(-12 + 6*i), float(-12 + 6*j)] for i in range(5) for j in range(5)]
loc_1 = [[float(-8 + 8*i), float(-8 + 8*j)] for i in range(3) for j in range(3)]
loc_2 = [[float(-6 + 12*i), float(-6 + 12*j)] for i in range(2) for j in range(2)]
loc_3 = [[float(0), float(0)]]
locations = [loc_0, loc_1, loc_2, loc_3]

# volume change
data_total_v = np.zeros((4, 0))
data_types_v = np.zeros((7, 3, 0))
f2 = open("Volume_change_new.txt", 'w')
data_bandwidth_efficiency_v = np.zeros((2, 0))
for vol in vol_dec_set:
    print("---------Volume decrement: ", vol)
    result = NC.network_simulation(rates, locations, cur_data_type_dist, data_task_c, vol, layer_task_c)
    temp_vol = vol.__str__() + "\n"
    temp_data_info = temp_vol
    res_temp = NC.task_convert_input(data_task_c, vol, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_v = np.append(data_bandwidth_efficiency_v, temp_b_e_data, axis=1)
    temp_type_service_time = result[1]
    data_total_v = np.append(data_total_v, result[0], axis=1)
    data_types_v = np.append(data_types_v, result[1], axis=2)
    f2.write(temp_data_info)
f2.close()
np.save('Bandwidth_efficiency_V_new.npy', data_bandwidth_efficiency_v)
np.save('Total_service_time_V_new.npy', data_total_v)
np.save('Type_service_time_V_new.npy', data_types_v)
