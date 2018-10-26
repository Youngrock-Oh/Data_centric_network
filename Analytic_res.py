from math import sqrt
import numpy as np
from numpy.random import uniform
import math


c = 3e5  # km / sec
# Return path delay between two nodes
def delay(loc_1, loc_2):
    x = loc_1[0] - loc_2[0]
    y = loc_1[1] - loc_2[1]
    return sqrt(x * x + y * y) / c


def delay_return(locations_source, locations_server):
    '''
	Inputs: two layers location coordinates (array)
	Outputs: two layers delay matrix (array)
	'''

    m = len(locations_source)
    n = len(locations_server)
    delay_matrix = np.zeros((m, n))
    for i in range(m):
        for k in range(n):
            delay_matrix[i, k] = delay(locations_source[i], locations_server[k])
    return delay_matrix


def analytic_avg_delay_two_layers(arrival_rates, service_rates, delta, A):
    """
    :param arrival_rates: arrival_rates (array, 1 by m size)
    :param service_rates: service_rates (array, 1 by n size)
    :param delta:
    :param A: routing probabilities (array  m by n size)
    :return: expected service time including propagation delay considering just two layers
    """

    m = len(arrival_rates)
    n = len(service_rates)
    lambda_hat = np.matmul(arrival_rates, A)
    res_sum = 0
    for i in range(m):
        res_sum += np.dot(A[i, :], 1 / (service_rates - lambda_hat) + delta[i, :]) * arrival_rates[i]
    return res_sum / sum(arrival_rates)


def analytic_avg_delay(rates, delta, routing_p, vol_dec):
    """
    :param rates: [array (rates in layer 0), array (rates in layer 0), ...]
    :param delta:
    :param routing_p: routing probabilities [array (routing probabilites in layer 0), array (routing probabilites in layer 0), ...]
    :param vol_dec:
    :return: expected service time including propagation delay
    """
    layer_num = len(rates)
    lambda_hat = [np.zeros((1, len(rates[i]))) for i in range(layer_num)]
    lambda_hat[0] = rates[0]
    for i in range(1, layer_num):
        lambda_hat[i] = np.matmul(lambda_hat[i - 1], routing_p[i - 1])
        test = rates[i] - lambda_hat[i]
        if test[test <= 0]:
            print("Initial A is wrong!")
            return 1
    res_sum = 0
    for i in range(layer_num - 1):
        res_sum += analytic_avg_delay_two_layers(lambda_hat[i], rates[i + 1] / vol_dec[:i+1].prod(), delta[i], routing_p[i])
    return res_sum




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
            rate = uniform(avg_rate_layer[i] * (1. - rate_sigma), avg_rate_layer[i] * (1. + rate_sigma))
            locations[i].append([x, y])
            rates[i].append(rate)
    return [locations, rates, A]


def no_delay_optimal(arrival_rates, service_rates):
    '''
    	Find the optimal completion time using Lagrange multiplier for a network without propagation delays
    	considering only two layers
    '''
    n = len(service_rates)
    num = 0
    for j in range(n):
        num += sqrt(service_rates[j])
    denom = sum(service_rates) - sum(arrival_rates)
    K = pow(num / denom, 2)
    lambda_hat = np.zeros((n, 1))
    for j in range(n):
        lambda_hat[j] = service_rates[j] - sqrt(service_rates[j]/K)
    service_time = 0
    for j in range(n):
        service_time += lambda_hat[j] / (service_rates[j] - lambda_hat[j])
    service_time = service_time / sum(arrival_rates)
    result = {'lambda_hat': lambda_hat, 'Mean_completion_time': service_time}
    return result
