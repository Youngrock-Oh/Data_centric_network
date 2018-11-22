import numpy as np
import Network_Classes as NC
import Analytic_res as ar
import copy

# Default settings
# Volume decrement
task_vol_dec_c = {"T1": 0.5, "T2": 0.5, "T3": 1}
# data and task configuration
cur_data_type_c = 1 / 7 * np.ones(7)
data_task_c = {0: ["T1"], 1: ["T2"], 2: ["T1", "T2"], 3: ["T1", "T3"], 4: ["T3"], 5: ["T2", "T3"], 6: ["T1", "T2", "T3"]}
layer_task_c = {0: [], 1: ["T1"], 2: ["T2"], 3: ["T3"]}
# Rate
rates_0 = np.array([30 + 10 * (i // 5) for i in range(25)])
rates_1 = np.array([350 + 100 * (i // 3) for i in range(9)])
rates_2 = np.array([700, 800, 900, 1000])
rates_3 = np.array([2000])
rates_c = [rates_0, rates_1, rates_2, rates_3]
# Location
loc_0 = [[float(-12 + 6*i), float(-12 + 6*j)] for i in range(5) for j in range(5)]
loc_1 = [[float(-8 + 8*i), float(-8 + 8*j)] for i in range(3) for j in range(3)]
loc_2 = [[float(-6 + 12*i), float(-6 + 12*j)] for i in range(2) for j in range(2)]
loc_3 = [[float(0), float(0)]]
location_c = [loc_0, loc_1, loc_2, loc_3]

# Rate change
rates_01 = np.array([25 + 10 * (i // 5) for i in range(25)])
rates_02 = np.array([27.5 + 10 * (i // 5) for i in range(25)])
rates_03 = np.array([32.5 + 10 * (i // 5) for i in range(25)])
rates_04 = np.array([35 + 10 * (i // 5) for i in range(25)])

rates_11 = np.array([300 + 100 * (i // 3) for i in range(9)])
rates_12 = np.array([325 + 100 * (i // 3) for i in range(9)])
rates_13 = np.array([375 + 100 * (i // 3) for i in range(9)])
rates_14 = np.array([400 + 100 * (i // 3) for i in range(9)])

rates_21 = np.array([600, 700, 800, 900])
rates_22 = np.array([650, 750, 850, 950])
rates_23 = np.array([750, 850, 950, 1050])
rates_24 = np.array([800, 800, 1000, 1100])

rates0 = rates_c
rates1 = [rates_01, rates_1, rates_2, rates_3]
rates2 = [rates_02, rates_1, rates_2, rates_3]
rates3 = [rates_03, rates_1, rates_2, rates_3]
rates4 = [rates_04, rates_1, rates_2, rates_3]

rates5 = [rates_0, rates_11, rates_2, rates_3]
rates6 = [rates_0, rates_12, rates_2, rates_3]
rates7 = [rates_0, rates_13, rates_2, rates_3]
rates8 = [rates_0, rates_14, rates_2, rates_3]

rates9 = [rates_0, rates_1, rates_21, rates_3]
rates10 = [rates_0, rates_1, rates_22, rates_3]
rates11 = [rates_0, rates_1, rates_23, rates_3]
rates12 = [rates_0, rates_1, rates_24, rates_3]
rates13 = [rates_0, rates_1, rates_2, np.array([3000])]

rates_set = [rates_c, rates1, rates2, rates3, rates4, rates5, rates6,
             rates7, rates8, rates9, rates10, rates11, rates12, rates13]
# rates change
data_total_r = np.zeros((4, 0))
data_types_r = np.zeros((7, 3, 0))
f1 = open("Rate_change_new_R.txt", 'w')
data_bandwidth_efficiency_r = np.zeros((2, 0))
for rate in rates_set:
    print("---------Rates: ", rate)
    result = NC.network_simulation(rate, location_c, cur_data_type_c, data_task_c, task_vol_dec_c, layer_task_c)
    temp_ra = rate.__str__() + "\n"
    temp_data_info = temp_ra
    res_temp = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_c, rate[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_c, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_r = np.append(data_bandwidth_efficiency_r, temp_b_e_data, axis=1)
    temp_type_service_time = result[1]
    data_total_r = np.append(data_total_r, result[0], axis=1)
    data_types_r = np.append(data_types_r, result[1], axis=2)
    f1.write(temp_data_info)
f1.close()
np.save('Bandwidth_efficiency_R_new.npy', data_bandwidth_efficiency_r)
np.save('Total_service_time_R_new.npy', data_total_r)
np.save('Type_service_time_R_new.npy', data_types_r)

# Volume reduction change
vol_dec_c_1 = {"T1": 0.25, "T2": 1, "T3": 1}
vol_dec_c_2 = {"T1": 1, "T2": 0.25, "T3": 1}
vol_dec_c_3 = {"T1": 0.1, "T2": 0.1, "T3": 1}
vol_dec_c_4 = {"T1": 0.01, "T2": 1, "T3": 1}
vol_dec_c_5 = {"T1": 1, "T2": 0.01, "T3": 1}
vol_dec_set = [task_vol_dec_c, vol_dec_c_1, vol_dec_c_2, vol_dec_c_3, vol_dec_c_4, vol_dec_c_5]

# Data type distribution change
data_type_dist_set = np.zeros((5, 7))
data_type_dist_set[0, :] = cur_data_type_c
data_type_dist_set[1, :] = np.array([1/3, 1/6, 1/6, 1/12, 1/12, 1/12, 1/12])
data_type_dist_set[2, :] = np.array([2/3, 1/12, 1/12, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[3, :] = np.array([1/6, 1/3, 1/3, 1/24, 1/24, 1/24, 1/24])
data_type_dist_set[4, :] = np.array([1/6, 1/12, 1/12, 1/6, 1/6, 1/6, 1/6])

# Locations change
locations0 = copy.deepcopy(location_c)
locations1 = copy.deepcopy(location_c)
locations2 = copy.deepcopy(location_c)
locations3 = copy.deepcopy(location_c)
number = np.array([223.6/24, 316.6/24, 387.3/24, 469.8/24])
temp_loc_set = [locations0, locations1, locations2, locations3]
for i in range(len(temp_loc_set)):
    for j in range(len(temp_loc_set[i])):
        for k in range(len(temp_loc_set[i][j])):
            for s in range(len(temp_loc_set[i][j][k])):
                temp_loc_set[i][j][k][s] *= number[i]
loc_set = [temp_loc_set[0], temp_loc_set[1], temp_loc_set[2], temp_loc_set[3]]

# volume change
data_total_v = np.zeros((4, 0))
data_types_v = np.zeros((7, 3, 0))
f2 = open("Volume_change_new.txt", 'w')
data_bandwidth_efficiency_v = np.zeros((2, 0))
for vol in vol_dec_set:
    print("---------Volume decrement: ", vol)
    result = NC.network_simulation(rates_c, location_c, cur_data_type_c, data_task_c, vol, layer_task_c)
    temp_vol = vol.__str__() + "\n"
    temp_data_info = temp_vol
    res_temp = NC.task_convert_input(data_task_c, vol, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_c, rates_c[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_c, layer_dic_input)
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

# location change
data_total_l = np.zeros((4, 0))
data_types_l = np.zeros((7, 3, 0))
f3 = open("Location_change_new.txt", 'w')
data_bandwidth_efficiency_l = np.zeros((2, 0))
for location in loc_set:
    print("---------Locations: ", location)
    result = NC.network_simulation(rates_c, location, cur_data_type_c, data_task_c, task_vol_dec_c, layer_task_c)
    temp_loc = location.__str__() + "\n"
    temp_data_info = temp_loc
    res_temp = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_c, rates_c[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_c, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_l = np.append(data_bandwidth_efficiency_l, temp_b_e_data, axis=1)
    temp_type_service_time = result[1]
    data_total_l = np.append(data_total_l, result[0], axis = 1)
    data_types_l = np.append(data_types_l, result[1], axis = 2)
    f3.write(temp_data_info)
f3.close()
np.save('Bandwidth_efficiency_L_new.npy', data_bandwidth_efficiency_l)
np.save('Total_service_time_L_new.npy', data_total_l)
np.save('Type_service_time_L_new.npy', data_types_l)

# Type distribution change
data_total_dist = np.zeros((4, 0))
data_types_dist = np.zeros((7, 3, 0))
f4 = open("Type_dist_change_new.txt", 'w')
data_bandwidth_efficiency = np.zeros((2, 0))
for case_index in range(len(data_type_dist_set)):
    print("Case %d" % case_index)
    data_type_dist_c = data_type_dist_set[case_index]
    result = NC.network_simulation(rates_c, location_c, data_type_dist_c, data_task_c, task_vol_dec_c, layer_task_c)
    res = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    vol_dec_input = res[1]
    layer_dic_input = res[2]
    temp_data_info = data_type_dist_c.__str__() + "\n"
    temp_b_e = ar.bandwidth_efficiency_compare(data_type_dist_c, rates_0, layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_last_layer(data_type_dist_c, layer_dic_input)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency = np.append(data_bandwidth_efficiency, temp_b_e_data, axis=1)
    data_total_dist = np.append(data_total_dist, result[0], axis=1)
    data_types_dist = np.append(data_types_dist, result[1], axis=2)
    f1.write(temp_data_info)

f4.close()
np.save('Bandwidth_efficiency_dist_new.npy', data_bandwidth_efficiency)
print(data_bandwidth_efficiency)
np.save('Total_service_time_dist_new.npy', data_total_dist)
np.save('Type_service_time_dist_new.npy', data_types_dist)
