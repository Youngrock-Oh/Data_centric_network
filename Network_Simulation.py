from math import sqrt
from numpy.random import exponential
from numpy.random import choice
from numpy.random import uniform
import numpy as np
import Network_Classes as NC

# location parameters
net_region_width = 1.0  # km
net_region_height = 1.0  # km
layer_num = 4
node_num = [3, 5, 3, 1]
locations = [[] for i in range(layer_num)]

# rate parameters
rates = [[]]*layer_num
avg_rate_layer = [[]] * layer_num
avg_rate_layer[0] = 10.0
total_arrival_rate = node_num[0] * avg_rate_layer[0]
rate_margin = 0.8
rate_sigma = 0.2
for i in range(1, layer_num):
    avg_rate_layer[i] = node_num[i - 1] * avg_rate_layer[i - 1] / node_num[i] / rate_margin

# data type distribution and layer_dic
data_type_dist = (0.2, 0.35, 0.3, 0.1, 0.05)
layer_dic = {0: [0,1], 1: [0,1], 2: [0,1], 3: [0,1], 4: [0,1]}

# routing probability A
A = [[] for i in range(layer_num)]
for i in range(layer_num - 1):
    A[i] = np.ones((node_num[i], node_num[i + 1]))/node_num[i + 1]
A[layer_num - 1] = np.zeros((node_num[layer_num - 1], node_num[layer_num - 1]))

# Define locations, rates
for i in range(layer_num):
    for j in range(node_num[i]):
        x = uniform(-net_region_width/2, net_region_width/2)
        y = uniform(-net_region_height/2, net_region_height/2)
        rate = uniform(avg_rate_layer[i]*(1 - rate_sigma), avg_rate_layer[i]*(1 + rate_sigma))
        locations[i].append([x, y])
        rates[i].append(rate)

cur_network = NC.Network(locations, rates, data_type_dist, layer_dic, A)
time = 0
simulation_time = 1  # sec
while time < simulation_time:
    close_event_info = cur_network.update_time()
    close_event_time = close_event_info[0]
    sending_index = close_event_info[1]
    cur_network.update(close_event_time, sending_index)
    print(close_event_time)
    time += close_event_time
