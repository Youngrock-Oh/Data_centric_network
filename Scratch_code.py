from Network_Classes import *
vol_dec = np.zeros((2, 3))
vol_dec[0, :] = np.array([1, 1/2, 1/2])
vol_dec[1, :] = np.array([1, 1/2, 1/3])
arrival_rates = np.array([10, 5])
service_rates = np.array([10, 20])
data_dist = np.array([1/2, 1/2])

data_task = {0: ["T1"], 1: ["T1", "T2"], 2: ["T1", "T2", "T3"], 3: ["T1", "T2", "T3", "T4"]}
task_vol_dec = {"T1": 1/2, "T2": 1/2, "T3": 1/2, "T4": 1/2}
layer_task = {0: [], 1: [], 2: ["T1", "T2"], 3: [], 4: ["T3", "T4"]}
res = task_convert_input(data_task, task_vol_dec, layer_task)
print(res[0])
print(res[1])
print(res[2])

