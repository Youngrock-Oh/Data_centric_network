from math import sqrt
import numpy as np
from numpy.random import uniform
import math
from KJ import grad_projected
from JS import barrier_method


c = 3e5  # km / sec
# Return path delay between two nodes


def delay(loc_1, loc_2):
    x = loc_1[0] - loc_2[0]
    y = loc_1[1] - loc_2[1]
    return sqrt(x * x + y * y) / c * 10  # multiplied by 10 for scaling


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
    :param rates: [array (rates in layer 0), array (rates in layer 1), ...]
    :param delta:
    :param routing_p: routing probabilities [array (routing probabilites in layer 0), array (routing probabilites in layer 1), ...]
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


def cur_vol(cur_layer_index, layer_dic, vol_dec):
    data_type_num = len(layer_dic.keys())
    res = np.ones(data_type_num)
    for i in range(data_type_num):
        for j in range(cur_layer_index + 1):
            if layer_dic[i].count(j) > 0:
                res[i] *= vol_dec[j]
    return 1 / res


def effective_rates(arrival_rates, service_rates, cur_layer_index, layer_dic, data_dist, vol_dec):
    data_type_num = len(data_dist)
    effective_dist = np.zeros(data_type_num)
    data_vol = cur_vol(cur_layer_index, layer_dic, vol_dec)
    for i in range(data_type_num):
        if layer_dic[i].count(cur_layer_index + 1) > 0:
            effective_dist[i] = data_dist[i]
    eff_arrival_rates = arrival_rates * sum(effective_dist)
    eff_service_rates = service_rates * (np.dot(data_vol, effective_dist) / sum(effective_dist))
    return [eff_arrival_rates, eff_service_rates]


def grad_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec):
    layer_num = len(rates)
    optimal_A = []
    source_rates = rates[0]
    for l in range(layer_num - 2):
        temp_arr_rates = source_rates
        temp_ser_rates = rates[l + 1]
        eff_rates = effective_rates(temp_arr_rates, temp_ser_rates, l, layer_dic, data_type_dist, vol_dec)
        eff_arr_rates = eff_rates[0]
        eff_ser_rates = eff_rates[1]
        temp_res = grad_projected(eff_arr_rates, eff_ser_rates, delta[l], initial_a[l])
        temp_A = temp_res['A']
        optimal_A.append(temp_A)
        source_rates = np.matmul(source_rates, temp_A)
    last_layer_num = len(rates[layer_num - 2])
    optimal_A.append(np.ones((last_layer_num, 1)))
    return optimal_A


def barrier_multi_layers(rates, delta, initial_a, layer_dic, data_type_dist, vol_dec):
    layer_num = len(rates)
    optimal_A = []
    source_rates = rates[0]
    for l in range(layer_num - 2):
        temp_arr_rates = source_rates
        temp_ser_rates = rates[l + 1]
        eff_rates = effective_rates(temp_arr_rates, temp_ser_rates, l, layer_dic, data_type_dist, vol_dec)
        eff_arr_rates = eff_rates[0]
        eff_ser_rates = eff_rates[1]
        temp_res = barrier_method(eff_arr_rates, eff_ser_rates, delta[l], initial_a[l])
        temp_A = temp_res['A']
        optimal_A.append(temp_A)
        source_rates = np.matmul(source_rates, temp_A)
    last_layer_num = len(rates[layer_num - 2])
    optimal_A.append(np.ones((last_layer_num, 1)))
    return optimal_A
