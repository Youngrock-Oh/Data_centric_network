from math import sqrt
from numpy.random import exponential
from numpy.random import choice
from numpy.random import uniform
import numpy as np
import Network_Classes as NC

def uniform_random_network(net_region_width, net_region_height, layer_num, node_num, avg_rate_source_node):
    """
    Inputs:
    net_region_width, net_region_height (km),
    layer_num: # of layers (int)
    node_num: # of servers in each layer (list)
    avg_rate_source_node (float)

    Construct locations using PPP
    Construct rates using rate_margin and rate_sigma and uniform distribution
    Construct routing probability - uniform

    Outputs: locations, rates, routing probability A
    """
    locations = [[] for i in range(layer_num)]
    # rate parameters
    rates = [[] for i in range(layer_num)]
    avg_rate_layer = [[] for i in range(layer_num)]  # avg rate of a server in the layer
    avg_rate_layer[0] = avg_rate_source_node
    rate_margin = 0.8
    rate_sigma = 0.2
    for i in range(1, layer_num):
        avg_rate_layer[i] = node_num[i - 1] * avg_rate_layer[i - 1] / node_num[i] / rate_margin
    # Uniform routing probability A
    A = [[] for i in range(layer_num)]
    for i in range(layer_num - 1):
        A[i] = np.ones((node_num[i], node_num[i + 1])) / node_num[i + 1]
    A[layer_num - 1] = np.zeros((node_num[layer_num - 1], node_num[layer_num - 1]))

    # Construct locations, rates
    for i in range(layer_num):
        for j in range(node_num[i]):
            x = uniform(-net_region_width / 2, net_region_width / 2)
            y = uniform(-net_region_height / 2, net_region_height / 2)
            rate = uniform(avg_rate_layer[i] * (1 - rate_sigma), avg_rate_layer[i] * (1 + rate_sigma))
            locations[i].append([x, y])
            rates[i].append(rate)
    return [locations, rates, A]

# location parameters
data_size = 500*10**3*8  # 500 KB
cycle_per_bit_edge = 6*10**9 # (cycles per slot)
cycle_per_bit_main_server = 5*10**15 # (cycles per slot)
T_slot = 100*10**-3 # time slot 100ms

avg_rate_edge = 1 / (data_size / cycle_per_bit_edge * T_slot)
locations =[[[1, 0]] , [[0, 0]]]
rates = np.array([[0.8*avg_rate_edge], [avg_rate_edge]])
A = [[[0.1]], [[0]]]

# data type distribution and layer_dic
data_type_dist = [1]
layer_dic = {0: [0, 1]}

cur_network = NC.Network(locations, rates, data_type_dist, layer_dic, A)
time = 10
simulation_time = 10  # sec
while time < simulation_time:
    close_event_info = cur_network.update_time()
    close_service_time = close_event_info[0]
    sending_index = close_event_info[1]
    cur_network.update(close_service_time, sending_index)
    time += close_service_time

simulation_service_time = cur_network.Net_completion_time/cur_network.Num_completed_data
expected_service_time = 1 / (rates[1] - rates[0])
print(simulation_service_time, expected_service_time)