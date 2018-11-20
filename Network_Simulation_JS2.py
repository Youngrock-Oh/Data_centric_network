import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar
import copy


def network_simulation(rates, vol_dec, locations):
    start_time = time.time()
    result1 = np.array([])
    result2 = np.array([[], [], [], [], [], [], []])
    data_type_dist = 1/7*np.ones(7)
    layer_dic = {0: [0, 1], 1: [0, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 3], 5: [0, 2, 3], 6: [0, 1, 3]}
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
    for case_num, case in simulation_cases.items():
        if case_num == 0:
            res_A[case_num] = initial_a
        elif case_num == 1:
            res_A[case_num] = ar.barrier_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec)
        elif case_num == 2:
            res_A[case_num] = ar.grad_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec)
        else:
            res_A[case_num] = ar.legacy_optimal_routing(locations)
            data_type_dist = np.array([1])
            vol_dec = np.ones(len(vol_dec))
            layer_dic =  {0: [0, len(vol_dec) - 1]}

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


#Rate 
rates_0 = np.array([30 + 20 * (i // 5) for i in range(25)])
rates_1 = np.array([250 + 100 * (i // 3) for i in range(9)])
rates_2 = np.array([500, 600, 800, 900])
rates_3 = np.array([2000])
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
rates_set = [rates, rates1, rates2, rates3, rates4, rates5, rates6, rates7, rates8, rates9, rates10, rates11, rates12]

#Volume decrement
vol_dec = np.array([1, 0.8, 0.8, 1])
vol1 = np.array([1,32/45, 0.9,1])
vol2 = np.array([1,0.9, 32/45,1])
vol3 = np.array([1,0.01, 1,1])
vol4 = np.array([1, 0.1, 0.1,1])
vol5 = np.array([1, 1, 0.01 ,1])
vol_set = [vol1, vol2, vol3, vol4, vol5]

#Location 
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

#data type distribution
cur_data_type_dist = 1/7*np.ones(7)

#rates change
data_total_r = np.zeros((4, 0))
data_types_r = np.zeros((7, 3, 0))
f1 = open("Rate_change1.txt", 'w')
data_bandwidth_efficiency_r = np.zeros((2, 0))        
for rate in rates_set:
    print("---------Rates: ", rate)
    result = network_simulation(rate, vol_dec, locations)
    temp_ra = rate.__str__() + "\n"
    temp_data_info = temp_ra 
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rate[0])
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_r = np.append(data_bandwidth_efficiency_r, temp_b_e_data, axis=1)  
    temp_type_service_time = result[1]
    data_total_r = np.append(data_total_r, result[0], axis = 1)
    data_types_r = np.append(data_types_r, result[1], axis = 2)
    f1.write(temp_data_info)
f1.close()
np.save('Bandwidth_efficiency_R.npy', data_bandwidth_efficiency_r)
np.save('Total_service_time_R.npy', data_total_r)
np.save('Type_service_time_R.npy', data_types_r)



#volume change
data_total_v = np.zeros((4, 0))
data_types_v = np.zeros((7, 3, 0))
f2 = open("Volume_change.txt", 'w')
data_bandwidth_efficiency_v = np.zeros((2, 0))         
for vol in vol_set:
    print("---------Volume decrement: ", vol)
    result = network_simulation(rates, vol, locations)
    temp_vol = vol.__str__() + "\n"
    temp_data_info = temp_vol 
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0], vol)
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_v = np.append(data_bandwidth_efficiency_v, temp_b_e_data, axis=1)
    temp_type_service_time = result[1]
    data_total_v = np.append(data_total_v, result[0], axis = 1)
    data_types_v = np.append(data_types_v, result[1], axis = 2)
    f2.write(temp_data_info)
f2.close()
np.save('Bandwidth_efficiency_V.npy', data_bandwidth_efficiency_v)
np.save('Total_service_time_V.npy', data_total_v)
np.save('Type_service_time_V.npy', data_types_v)



#location change 
data_total_l = np.zeros((4, 0))
data_types_l = np.zeros((7, 3, 0))
f3 = open("Location_change.txt", 'w')        
data_bandwidth_efficiency_l = np.zeros((2, 0))
for location in loc_set:
    print("---------Locations: ", location)
    result = network_simulation(rates, vol_dec, location)
    temp_loc = location.__str__() + "\n"
    temp_data_info = temp_loc 
    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0])
    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist)
    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
    data_bandwidth_efficiency_l = np.append(data_bandwidth_efficiency_l, temp_b_e_data, axis=1)
    temp_type_service_time = result[1]
    data_total_l = np.append(data_total_l, result[0], axis = 1)
    data_types_l = np.append(data_types_l, result[1], axis = 2)
    f3.write(temp_data_info)
f3.close()
np.save('Bandwidth_efficiency_L.npy', data_bandwidth_efficiency_l)
np.save('Total_service_time_L.npy', data_total_l)
np.save('Type_service_time_L.npy', data_types_l)


#vol 마지막  
#data_total_v = np.zeros((4, 0))
#data_types_v = np.zeros((7, 3, 0))
#data_bandwidth_efficiency_v = np.zeros((2, 0))  
#for vol in [vol_set[4]]:
#    print("---------Volume decrement: ", vol)
#    result = network_simulation(rates, vol, locations)
#    temp_vol = vol.__str__() + "\n"
#    temp_data_info = temp_vol 
#    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0], vol)
#    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist)
#    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
#    data_bandwidth_efficiency_v = np.append(data_bandwidth_efficiency_v, temp_b_e_data, axis=1)
#    temp_type_service_time = result[1]
#    data_total_v = np.append(data_total_v, result[0], axis = 1)
#    data_types_v = np.append(data_types_v, result[1], axis = 2)

#loc 마지막
#data_total_l = np.zeros((4, 0))
#data_types_l = np.zeros((7, 3, 0))       
#data_bandwidth_efficiency_l = np.zeros((2, 0))
#for location in [loc_set[4]]:
#    print("---------Locations: ", location)
#    result = network_simulation(rates, vol_dec, location)
#    temp_loc = location.__str__() + "\n"
#    temp_data_info = temp_loc 
#    temp_b_e = ar.bandwidth_efficiency_compare(cur_data_type_dist, rates[0])
#    temp_metric = ar.avg_sum_required_layer(cur_data_type_dist)
#    temp_b_e_data = np.array([temp_b_e, temp_metric]).reshape((2, 1))
#    data_bandwidth_efficiency_l = np.append(data_bandwidth_efficiency_l, temp_b_e_data, axis=1)
#    temp_type_service_time = result[1]
#    data_total_l = np.append(data_total_l, result[0], axis = 1)
#    data_types_l = np.append(data_types_l, result[1], axis = 2)
#    


