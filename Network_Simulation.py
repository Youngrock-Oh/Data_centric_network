
import numpy as np
import Network_Classes as NC
import time
import Analytic_res as ar

start_time = time.time()

# location parameters
data_size = 500*10**3*8  # 500 KB
cycle_per_bit_edge = 6*10**9 # (cycles per slot)
cycle_per_bit_main_server = 5*10**15 # (cycles per slot)
T_slot = 100*10**-3 # time slot 100ms

#avg_rate_edge = 1 / (data_size / cycle_per_bit_edge * T_slot)
avg_rate_edge = 100
locations =[[[10, 0]] , [[0, 0]]]
rates = np.array([[0.1*avg_rate_edge], [avg_rate_edge]])
A = [[[1]], [[0]]]

# data type distribution and layer_dic
data_type_dist = [1]
layer_dic = {0: [0, 1]}

cur_network = NC.Network(locations, rates, data_type_dist, layer_dic, A)
cur_time = 0
simulation_time = 1000  # sec

t = 1
while cur_time < simulation_time:
    close_event_info = cur_network.update_time()
    close_service_time = close_event_info[0]
    sending_index = close_event_info[1]
    cur_network.update(close_service_time, sending_index)
    cur_time += close_service_time
    if cur_time > t:
        print(cur_time)
        t += 1

simulation_service_time = cur_network.Net_completion_time/cur_network.Num_completed_data
locations_source = np.array(locations[0])
locations_server = np.array(locations[1])
arrival_rates = np.array(rates[0])
service_rates = np.array(rates[1])
routing_p = np.array(A[0])
expected_service_time_1 = 1 / (rates[1] - rates[0])
expected_service_time_2 = ar.analytic_avg_delay(locations_source, locations_server, arrival_rates, service_rates, routing_p)
print(simulation_service_time, expected_service_time_1, expected_service_time_2)
print("--- %s seconds ---" % (time.time() - start_time))
