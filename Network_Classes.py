from math import sqrt
from numpy.random import exponential
from numpy.random import choice
import numpy as np


# example for layer_dic:
layer_dic = {0: [0, 1, 2, 3], 1: [0, 1, 2], 2: [0, 1], 3: [0, 2, 3], 4: [0, 3]}
Num_completed_data = 0
c = 3e5 # km / sec

# Return path delay between two nodes
def delay(node_1, node_2):
    x = node_2.loc[0] - node_1.loc[0]
    y = node_2.loc[1] - node_1.loc[1]
    return sqrt(x * x + y * y) / c


def complete():
    global Num_completed_data
    Num_completed_data += 1


# Class "Data": type, need_layers, cur_node, next_node
class Data:
    def __init__(self, type):
        self.type = type
        self.need_layers = layer_dic.get(type)

class Node:
    def __init__(self, loc, rate, routing_P, cur_layer, source, data_type_dist):
        self.loc = loc
        self.rate = rate
        self.remaining_time = []
        self.routing_P = routing_P
        self.data_stack = []
        self.cur_layer = cur_layer
        self.source = source
        self.data_type_dist = data_type_dist

    def transfer_to(self, medium, sending_data, next_node):
        medium.data_stack.append([sending_data, next_node])  # Add the data to medium
        medium.delay_status.append(delay(self, next_node))  # Add the delay of the data to medium
        medium.remaining_time = min(medium.delay_status)  # Update medium remaining_time
        self.data_stack.remove(sending_data)
        if self.source:  # Source transfers the data and generates new data
            new_data_type = choice(len(self.data_type_dist), 1, True, self.data_type_dist)
            new_data = Data(new_data_type)
            self.add_data(new_data)
        else:  # Server just transfers the data
            if self.data_stack !=[]:
                self.remaining_time = exponential(1 / self.rate)
            else:
                self.remaining_time = []

    def add_data(self, data):
        if data.need_layers.count(self.cur_layer) == 0: # The data doesn't need to be processed in the node
            self.remaining_time = 0
            self.data_stack.insert(0, data)
        elif self.data_stack == []:
            self.data_stack.append(data)
            self.remaining_time = exponential(1 / self.rate)
        else: # Add the data to data_stack
            self.data_stack.append(data)


class Network:
    def __init__(self, locations, rates, data_type_dist, layer_dic, A):  # list, np.array, layer_dic, A_ij
        self.locations = locations
        self.rates = rates
        self.data_type_dist = data_type_dist
        self.layer_dic = layer_dic
        self.A = A
        # Construct nodes in the network
        self.network_nodes = []
        layer_num = len(locations)
        for l in range(layer_num):
            self.network_nodes.append([])
        for l in range(layer_num):
            for i in range(len(locations[l])):
                new_node = Node(locations[l][i], rates[l][i], A[l][i], l, l == 0, data_type_dist)
                if l == 0:
                    new_data_type = choice(len(data_type_dist), 1, True, data_type_dist)
                    new_data = Data(new_data_type)
                    new_node.add_data(new_data)
                self.network_nodes[l].append(new_node)
        self.medium = Medium()

    def update_time(self):
        layer_num = len(self.locations)
        close_service_time = self.network_nodes[0][0].remaining_time # initialization
        for l in range(layer_num):
            for i in range(len(self.locations[l])):
                if self.network_nodes[l][i].remaining_time != [] and \
                    self.network_nodes[l][i].remaining_time < close_service_time:
                    close_service_time = self.network_nodes[l][i].remaining_time
        if self.medium.remaining_time != [] and self.medium.remaining_time < close_service_time:
            close_service_time = self.medium.remaining_time
        return close_service_time

    def update(self, close_service_time):
        layer_num = len(self.locations)
        for l in range(layer_num):
            for i in range(len(self.locations[l])):
                sending_node = self.network_nodes[l][i]
                if sending_node.remaining_time == []:
                    pass
                elif sending_node.remaining_time == close_service_time:
                    sending_data = sending_node.data_stack[0]
                    if l < max(sending_data.need_layers): # the data must be transferred to the next layer
                        next_index = choice(len(sending_node.routing_P), 1, True, sending_node.routing_P)
                        next_node = self.network_nodes[l + 1][next_index]
                        sending_node.transfer_to(self.medium, sending_data, next_node)
                    else: # Processing the data is over
                        Complete()
                        del sending_node.data_stack[0]
                        if sending_node.data_stack != []:
                            sending_node.remaining_time = exponential(1 / sending_node.rate)
                        else:
                            sending_node.remaining_time = []
                else:
                    sending_node.remaining_time -= close_service_time
        self.medium.update(close_service_time)

# Class "Medium": corresponding to the transmission events
class Medium:
    def __init__(self):
        self.data_stack = []  # [Data, next_node], list
        self.delay_status = np.array([])  # [Delay time], array
        self.remaining_time = []

    def update(self, close_event_time):
        if self.remaining_time == close_event_time:  # Handover to next layer
            sending_data = self.data_stack[0][0]  # Find data to transfer
            next_node = self.data_stack[0][1]
            next_node.add_data(sending_data)  # Add the data to the next_node
            del self.data_stack[0]
            del self.delay_status[0]  # Remove the transferred data in the queue stack
            if self.data_stack != []:
                self.remaining_time = self.delay_status[0]
            else:
                self.remaining_time = []
        else:
            self.remaining_time -= close_event_time

