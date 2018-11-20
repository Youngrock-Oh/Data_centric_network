import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar
import copy


# Rate
rates_0 = np.array([30 + 20 * (i // 5) for i in range(25)])
rates_1 = np.array([250 + 100 * (i // 3) for i in range(9)])
rates_2 = np.array([500, 600, 800, 900])
rates_3 = np.array([2600])
rates = [rates_0, rates_1, rates_2, rates_3]
rates_01 = np.array([25 + 20 * (i // 5) for i in range(25)])
rates_02 = np.array([27.5 + 20 * (i // 5) for i in range(25)])
rates_03 = np.array([32.5 + 20 * (i // 5) for i in range(25)])
rates_04 = np.array([35 + 20 * (i // 5) for i in range(25)])

rates_11 = np.array([200 + 100 * (i // 3) for i in range(9)])
rates_12 = np.array([225 + 100 * (i // 3) for i in range(9)])
rates_13 = np.array([275 + 100 * (i // 3) for i in range(9)])
rates_14 = np.array([300 + 100 * (i // 3) for i in range(9)])

rates_21 = np.array([450, 550, 750, 850])
rates_22 = np.array([475, 575, 775, 875])
rates_23 = np.array([525, 625, 825, 925])
rates_24 = np.array([550, 650, 850, 950])

rates0 = rates
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

rates_set = [rates, rates1, rates2, rates3, rates4, rates5, rates6,
             rates7, rates8, rates9, rates10, rates11, rates12, rates13]

# Volume decrement
task_vol_dec_c = {"T1": 0.5, "T2": 0.5, "T3": 1}

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
locations0 = copy.deepcopy(locations)
locations1 = copy.deepcopy(locations)
locations2 = copy.deepcopy(locations)
locations3 = copy.deepcopy(locations)
number = np.array([223.6/24, 316.6/24, 387.3/24, 469.8/24])
temp_loc_set = [locations0, locations1, locations2, locations3]
for i in range(len(temp_loc_set)):
    for j in range(len(temp_loc_set[i])):
        for k in range(len(temp_loc_set[i][j])):
            for s in range(len(temp_loc_set[i][j][k])):
                temp_loc_set[i][j][k][s] *= number[i]
loc_set = [temp_loc_set[0], temp_loc_set[1], temp_loc_set[2], temp_loc_set[3]]

# rates change
data_total_r = np.zeros((4, 0))
data_types_r = np.zeros((7, 3, 0))
f1 = open("Rate_change_new_R.txt", 'w')
data_bandwidth_efficiency_r = np.zeros((2, 0))
for rate in rates_set:
    print("---------Rates: ", rate)
    result = NC.network_simulation(rate, locations, cur_data_type_dist, data_task_c, task_vol_dec_c, layer_task_c)
    temp_ra = rate.__str__() + "\n"
    temp_data_info = temp_ra
    res_temp = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rate[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist, layer_dic_input)
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

# location change
data_total_l = np.zeros((4, 0))
data_types_l = np.zeros((7, 3, 0))
f3 = open("Location_change_new.txt", 'w')
data_bandwidth_efficiency_l = np.zeros((2, 0))
for location in loc_set:
    print("---------Locations: ", location)
    result = NC.network_simulation(rates, location, cur_data_type_dist, data_task_c, task_vol_dec_c, layer_task_c)
    temp_loc = location.__str__() + "\n"
    temp_data_info = temp_loc
    res_temp = NC.task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
    vol_dec_input = res_temp[1]
    layer_dic_input = res_temp[2]
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0], layer_dic_input, vol_dec_input)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist, layer_dic_input)
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
