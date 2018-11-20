from Network_Classes import *

data_task_c = {0: ["T1"], 1: ["T1", "T2"], 2: ["T1", "T2", "T3"], 3: ["T1", "T2", "T3", "T4"]}
task_vol_dec_c = {"T1": 1 / 2, "T2": 1 / 10, "T3": 1 / 2, "T4": 1 / 2}
layer_task_c = {0: [], 1: [], 2: ["T1", "T2"], 3: [], 4: ["T3", "T4"]}
res = task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
print("--- rate factor \n ", res[0])
print("--- vol dec \n ", res[1])
print("--- layer_dic \n ", res[2])

data_task_c = {0: ["T1", "T2"], 1: ["T2", "T3"], 2: ["T3", "T4"]}
task_vol_dec_c = {"T1": 1 / 2, "T2": 1 / 10, "T3": 1 / 2, "T4": 1 / 2}
layer_task_c = {0: [], 1: [], 2: ["T1", "T2"], 3: [], 4: ["T3", "T4"]}
res = task_convert_input(data_task_c, task_vol_dec_c, layer_task_c)
print("--- rate factor \n ", res[0])
print("--- vol dec \n ", res[1])
print("--- layer_dic \n ", res[2])