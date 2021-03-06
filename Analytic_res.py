from math import sqrt
import numpy as np
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
            res[i] *= vol_dec[i, j]
    return res


def effective_rates(arrival_rates, service_rates, cur_layer_index, layer_dic, data_dist, vol_dec):
    data_type_num = len(data_dist)
    effective_dist = np.zeros(data_type_num)
    data_vol = cur_vol(cur_layer_index, layer_dic, vol_dec)
    for i in range(data_type_num):
        if layer_dic[i].count(cur_layer_index + 1) > 0:
            effective_dist[i] = data_dist[i]
    eff_arrival_rates = arrival_rates * sum(effective_dist)
    eff_service_rates = service_rates / (np.dot(data_vol, effective_dist) / sum(effective_dist))
    return [eff_arrival_rates, eff_service_rates]


def grad_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec):
    layer_num = len(rates)
    optimal_a = []
    source_rates = rates[0]
    for l in range(layer_num - 2):
        temp_arr_rates = source_rates
        temp_ser_rates = rates[l + 1]
        eff_rates = effective_rates(temp_arr_rates, temp_ser_rates, l, layer_dic, data_type_dist, vol_dec)
        eff_arr_rates = eff_rates[0]
        eff_ser_rates = eff_rates[1]
        if sum(eff_arr_rates) == 0:  # just passing through the layer
            temp_a = np.ones((len(eff_arr_rates), len(eff_ser_rates))) / len(eff_ser_rates)
        else:
            initial_a = valid_initial_rates(eff_arr_rates, eff_ser_rates, 0.9)
            temp_res = grad_projected(eff_arr_rates, eff_ser_rates, delta[l], initial_a)
            temp_a = temp_res['A']
        optimal_a.append(temp_a)
        source_rates = np.matmul(source_rates, temp_a)
    last_layer_num = len(rates[layer_num - 2])
    optimal_a.append(np.ones((last_layer_num, 1)))
    return optimal_a


def barrier_multi_layers(rates, delta, layer_dic, data_type_dist, vol_dec):
    layer_num = len(rates)
    optimal_a = []
    source_rates = rates[0]
    for l in range(layer_num - 2):
        temp_arr_rates = source_rates
        temp_ser_rates = rates[l + 1]
        eff_rates = effective_rates(temp_arr_rates, temp_ser_rates, l, layer_dic, data_type_dist, vol_dec)
        eff_arr_rates = eff_rates[0]
        eff_ser_rates = eff_rates[1]
        if sum(eff_arr_rates) == 0:  # just passing through the layer
            temp_a = np.ones((len(eff_arr_rates), len(eff_ser_rates))) / len(eff_ser_rates)
        else:
            initial_a = valid_initial_rates(eff_arr_rates, eff_ser_rates, 0.9)
            temp_res = barrier_method(eff_arr_rates, eff_ser_rates, delta[l], initial_a)
            temp_a = temp_res['A']
        optimal_a.append(temp_a)
        source_rates = np.matmul(source_rates, temp_a)
    last_layer_num = len(rates[layer_num - 2])
    optimal_a.append(np.ones((last_layer_num, 1)))
    return optimal_a


def valid_initial_rates(source_rates, server_rates, para):
    """
    :param source_rates: source rates (array)
    :param server_rates: server rates (array)
    :param para: parameter for finding initial rates
    :return: valid initial routing probabilities that guarantees queue stability
    """
    eps = 0.001
    sources_num = len(source_rates)
    servers_num = len(server_rates)
    initial_a = eps * np.ones((sources_num, servers_num))
    for i in range(servers_num):
        temp = np.ones(sources_num) * para * server_rates[i] / np.sum(source_rates)
        initial_a[:, i] = np.minimum(temp, 1 - np.sum(initial_a, 1) + initial_a[:, i])
    source_rates = np.matmul(source_rates.reshape((1, sources_num)), initial_a).flatten()
    # print(server_rates - source_rates) # to check validity
    return initial_a


def legacy_optimal_routing(locations):
    """
    :param locations: coordinates info for spatial distribution of nodes in the network
    :return: a, (list that consists of arrays) the optimal routing probability in the legacy network
    """
    layer_num = len(locations)
    a = [np.zeros((len(locations[i]), len(locations[i + 1]))) for i in range(layer_num - 1)]
    for i in range(layer_num - 1):
        for j in range(len(locations[i])):
            delay_info = [delay(locations[i][j], locations[i + 1][k]) for k in range(len(locations[i + 1]))]
            min_delay_index = np.argmin(delay_info)
            a[i][j][min_delay_index] = 1
    return a


def bandwidth_efficiency(vol_dec, data_type_dist, layer_dic, source_rates):
    """
    :param vol_dec: (array), volume decrease ratio after processing in each layer for each data type
    :param data_type_dist: (array), data type distribution
    :param layer_dic: (dictionary), required layer info for each data type
    :param source_rates: (array), source rates in the network
    :return: res, bandwidth efficiency which is proportion to the product of rate and data volume
    """
    layer_num = np.size(vol_dec, axis=1)
    res = 0
    data_type_num = len(data_type_dist)
    departure_process_rate = sum(source_rates)
    for l in range(layer_num - 1):
        temp_dist = np.zeros(data_type_num)
        for i in range(data_type_num):
            temp_max_layer = max(layer_dic[i])
            if temp_max_layer > l:
                temp_dist[i] = data_type_dist[i]
        cur_vol_temp = cur_vol(l, layer_dic, vol_dec)
        avg_data_vol = np.dot(cur_vol_temp, temp_dist)
        res += avg_data_vol
    return departure_process_rate * res


def bandwidth_efficiency_compare(data_type_dist, source_rates, layer_dic, vol_dec):
    """
    :param data_type_dist: (array), data type distribution
    :param source_rates: (array), source rates in the network
    :param layer_dic: (dictionary), required layer info for each data type
    :param vol_dec: (array), volume decrease ratio after processing in each layer for each data type
    :return: res, ratio of bandwidth usages between in-network processing and legacy networks
    """
    data_type_num = len(data_type_dist)
    layer_num = np.size(vol_dec, axis=1)
    legacy_vol_dec = np.ones((data_type_num, layer_num))
    legacy_data_type_dist = np.array([1])
    legacy_layer_dic = {0: [0, layer_num - 1]}
    b_e_legacy = bandwidth_efficiency(legacy_vol_dec, legacy_data_type_dist, legacy_layer_dic, source_rates)
    b_e_in_network_processing = bandwidth_efficiency(vol_dec, data_type_dist, layer_dic, source_rates)
    res = b_e_in_network_processing / b_e_legacy
    return res


def avg_last_layer(data_type_dist, layer_dic):
    """
    :param data_type_dist: (array), data type distribution
    :param layer_dic: (dictionary), required layer info for each data type
    :return: res, average last layer
    """
    temp = list(layer_dic.values())
    temp_max = np.array([max(temp[i]) for i in range(len(data_type_dist))])
    res = np.dot(data_type_dist, temp_max)
    return res


def avg_sum_required_layer(data_type_dist, layer_dic):
    """
    :param data_type_dist: (array), data type distribution
    :param layer_dic: (dictionary), required layer info for each data type
    :return: res, expected sum of the required layers
    """
    temp = list(layer_dic.values())
    temp_sum = np.array([sum(temp[i]) for i in range(len(data_type_dist))])
    res = np.dot(data_type_dist, temp_sum)
    return res

