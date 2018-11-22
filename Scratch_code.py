from Analytic_res import *
from Network_Classes import *
arrival_rates = 2
service_rates = 10
cur_layer_index = 2
layer_task_c = {0: [], 1: ["T1"], 2: ["T2", "T3"], 3: ["T4"]}
data_task_c = {0: ["T1", "T2"], 1: ["T2", "T3"], 2: ["T3", "T4"], 3: ["T1", "T2", "T3", "T4"]}
task_vol_dec_c = {"T1": 0.4, "T2": 0.5, "T3": 0.6, "T4": 0.7}
temp = task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
vol_dec = temp[0]
layer_dic = temp[2]
data_dist = np.array([0.3, 0.3, 0.2, 0.2])
x = effective_rates(arrival_rates, service_rates, cur_layer_index, layer_dic, data_dist, vol_dec)
print(x)