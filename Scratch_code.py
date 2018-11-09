from Analytic_res import *
source_rates = np.array([5, 5])
data_type_dist = np.zeros(7)
data_type_dist[0] = 1/2
data_type_dist[1] = 1/2
print(avg_last_layer(data_type_dist))