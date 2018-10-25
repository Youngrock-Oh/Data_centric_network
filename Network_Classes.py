from math import sqrt
from numpy.random import exponential
from numpy.random import choice
import numpy as np

# example for layer_dic: layer_dic = {0: [0, 1, 2, 3], 1: [0, 1, 2], 2: [0, 1], 3: [0, 2, 3], 4: [0, 3]}
c = 3e5  # km / sec


# Return path delay between two nodes
def delay(node_1, node_2):
    x = node_2.loc[0] - node_1.loc[0]
    y = node_2.loc[1] - node_1.loc[1]
    return sqrt(x * x + y * y) / c


# Class "Data": type, need_layers, cur_node, next_node
class Data:
    def __init__(self, data_type, layer_dic):
        """
        data_type: (int) data type
        layer_dic: (dict) required layer index
        """
        self.type = data_type
        self.need_layers = layer_dic.get(data_type)
        self.spending_time = 0


class Node:
    def __init__(self, rate, routing_p, data_type_dist, layer_dic, delta, layer_index, node_index):
        self.rate = rate
        self.routing_P = routing_p
        self.data_type_dist = data_type_dist
        self.layer_dic = layer_dic
        self.delta = delta
        self.layer_index = layer_index
        self.node_index = node_index
        self.data_stack = []
        self.remaining_time = []

    def spent_time(self, close_service_time):
        if self.data_stack:
            num_data = len(self.data_stack)
            for i in range(num_data):
                self.data_stack[i].spending_time += close_service_time

    def transfer_to(self, medium, sending_data, next_node):
        medium.data_stack.append([sending_data, next_node])  # Add the data to medium
        medium.delay_time.append(self.delta[next_node.node_index])  # Add delay in the delay queue
        medium.remaining_time = min(medium.delay_time)
        self.data_stack.remove(sending_data)
        if self.layer_index == 0:  # Source transfers the data and generates new data
            new_data_type = choice(len(self.data_type_dist), 1, True, self.data_type_dist)
            new_data_type = int(new_data_type)
            new_data = Data(new_data_type, self.layer_dic)
            self.add_data(new_data)
        else:  # Server just transfers the data
            if self.data_stack:
                self.remaining_time = exponential(1 / self.rate)
            else:
                self.remaining_time = []

    def add_data(self, data):
        if data.need_layers.count(self.layer_index) == 0:  # The data doesn't need to be processed in the node
            self.remaining_time = 0
            self.data_stack.insert(0, data)
        elif not self.data_stack:
            self.data_stack.append(data)
            self.remaining_time = exponential(1 / self.rate)
        else:  # Add the data to data_stack
            self.data_stack.append(data)


class Network:
    def __init__(self, rates, data_type_dist, layer_dic, delta, A):  # A, delta: [array, array, ...]
        self.rates = rates
        self.data_type_dist = data_type_dist
        self.layer_dic = layer_dic
        self.delta = delta
        self.A = A
        self.Num_completed_data = 0
        self.Net_completion_time = 0
        # Construct nodes in the network
        layer_num = len(rates)
        self.network_nodes = [[] for i in range(layer_num)]
        for l in range(layer_num):
            for i in range(len(rates[l])):
                new_node = Node(rates[l][i], A[l][i], data_type_dist, layer_dic, delta[l][i], l, i)
                if l == 0:
                    new_data_type = choice(len(data_type_dist), 1, True, data_type_dist)
                    new_data_type = int(new_data_type)
                    new_data = Data(new_data_type, layer_dic)
                    new_node.add_data(new_data)
                self.network_nodes[l].append(new_node)
        self.medium = Medium()

    def update_time(self):
        layer_num = len(self.rates)
        close_service_time = self.network_nodes[0][0].remaining_time  # initialization
        sending_index = [0, 0]
        for l in range(layer_num):
            for i in range(len(self.rates[l])):
                if self.network_nodes[l][i].remaining_time and self.network_nodes[l][i].remaining_time < 0:
                    print('Error!')
                if self.network_nodes[l][i].data_stack and \
                        self.network_nodes[l][i].remaining_time < close_service_time:
                    close_service_time = self.network_nodes[l][i].remaining_time
                    sending_index = [l, i]
        if self.medium.data_stack and self.medium.remaining_time < close_service_time:
            close_service_time = self.medium.remaining_time
            sending_index = 'medium'
        return [close_service_time, sending_index]

    def update(self, close_service_time, sending_index):
        layer_num = len(self.rates)
        if sending_index == 'medium':
            for l in range(layer_num):
                for i in range(len(self.rates[l])):
                    sending_node = self.network_nodes[l][i]
                    if sending_node.data_stack:
                        sending_node.remaining_time -= close_service_time
                    if l != 0:
                        sending_node.spent_time(close_service_time)
            self.medium.update(close_service_time)
        else:
            self.medium.update(close_service_time)
            for l in range(layer_num):
                for i in range(len(self.rates[l])):
                    sending_node = self.network_nodes[l][i]
                    if l != 0:
                        sending_node.spent_time(close_service_time)
                    if l == sending_index[0] and i == sending_index[1]:
                        sending_data = sending_node.data_stack[0]
                        if l < max(sending_data.need_layers):  # the data must be transferred to the next layer
                            next_index = choice(len(sending_node.routing_P), 1, True, sending_node.routing_P)
                            next_index = int(next_index)  # convert numpy.ndarray to int
                            next_node = self.network_nodes[l + 1][next_index]
                            sending_node.transfer_to(self.medium, sending_data, next_node)
                        else:  # Processing the data is over
                            self.Num_completed_data += 1
                            self.Net_completion_time += sending_data.spending_time
                            del sending_node.data_stack[0]
                            if not sending_node.data_stack:
                                sending_node.remaining_time = []
                            elif sending_node.data_stack[0].need_layers.count(l) > 0:
                                sending_node.remaining_time = exponential(1 / sending_node.rate)
                            else:
                                sending_node.remaining_time = 0
                    else:
                        if sending_node.data_stack:
                            sending_node.remaining_time -= close_service_time


# Class "Medium": corresponding to the transmission events
class Medium:
    def __init__(self):
        self.data_stack = []  # [Data, next_node], list
        self.delay_time = []  # delay time
        self.remaining_time = []
        # Assume that a node transmits multiple packets simultaneously -> well matched to our mathematical model

    def update(self, close_event_time):
        if self.data_stack:
            self.spent_time(close_event_time)
            if self.remaining_time == close_event_time:  # Handover to next layer
                sending_index = int(np.argmin(self.delay_time))
                sending_data = self.data_stack[sending_index][0]  # Find data to transfer
                next_node = self.data_stack[sending_index][1]
                next_node.add_data(sending_data)  # Add the data to the next_node
                del self.data_stack[sending_index]
                del self.delay_time[sending_index]

            if self.data_stack:
                temp = np.array(self.delay_time)
                temp -= close_event_time
                self.delay_time = list(temp)
                self.remaining_time = min(self.delay_time)
            else:
                self.remaining_time = []

    def spent_time(self, close_service_time):
        if self.data_stack:
            num_data = len(self.data_stack)
            for i in range(num_data):
                self.data_stack[i][0].spending_time += close_service_time
