import numpy as np
import Analytic_res as ar

source_rates = np.array([200+100 * (i//3) for i in range(9)])
server_rates = np.array([5000, 600, 800, 900])

para = 0.99
initial_a = ar.valid_initial_rates(source_rates, server_rates, para)
print(initial_a)
print(np.sum(initial_a, axis = 1))