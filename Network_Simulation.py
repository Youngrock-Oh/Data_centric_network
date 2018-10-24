
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
service_rates = np.array([600, 800, 1000, 1200])
arrival_rates = np.array([100 + 30*i for i in range(9)])
rates = [[100 + 30*i for i in range(9)], [600, 800, 1000, 1200]]
locations = [[[-10 + 10*i, -10 + 10*j] for i in range(3) for j in range(3)], [[-5, -5], [-5, 5], [5, -5], [5, 5]]]
delta = ar.delay_return(locations[0], locations[1])
initial_a = np.ones((9,4))/4
res_grad = ar.grad_projected(arrival_rates, service_rates, delta, initial_a)
res_bar = ar.barrier_method(arrival_rates, service_rates, delta, initial_a)
A = res_bar['A']
A_2 = [A, np.zeros((4,1))]

# data type distribution and layer_dic
data_type_dist = [1]
layer_dic = {0: [0, 1]}

cur_network = NC.Network(locations, rates, data_type_dist, layer_dic, A_2)
cur_time = 0
simulation_time = 30  # sec

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
#locations_source = np.array(locations[0])
#locations_server = np.array(locations[1])
#routing_p = A
#expected_service_time = ar.analytic_avg_delay(locations_source, locations_server, arrival_rates, service_rates, routing_p)
print('Gradient method: ', res_grad['Mean_completion_time'], res_grad['A'])
print('Barrier method: ', res_bar['Mean_completion_time'], res_bar['A'])
print('Simulation time: ', simulation_service_time)
print("--- %s seconds ---" % (time.time() - start_time))

# Check KJ's condition

arrival_rates = np.array([1,2])
service_rates = np.array([3,4])
delta = np.array([[0.1, 0.14],[0.15, 0.2]])
initial_a = np.array([[1/2, 1/2], [1/2, 1/2]])
res_1 = ar.grad_projected(arrival_rates, service_rates, delta, initial_a)
res_2 = ar.barrier_method(arrival_rates, service_rates, delta, initial_a)
res_3 = ar.analytic_avg_delay(arrival_rates, service_rates, delta, initial_a )
test_1 = np.matmul(arrival_rates, res_1['A'] - service_rates)
test_2 = np.matmul(arrival_rates, res_2['A'] - service_rates)