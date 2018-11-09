from Analytic_res import *

loc_0 = [[-12 + 6 * i, -12 + 6 * j] for i in range(5) for j in range(5)]
loc_1 = [[-8 + 8 * i, -8 + 8 * j] for i in range(3) for j in range(3)]
loc_2 = [[-6 + 12 * i, -6 + 12 * j] for i in range(2) for j in range(2)]
loc_3 = [[0, 0]]
locations = [loc_0, loc_1, loc_2, loc_3]

a = legacy_optimal_routing(locations)
print(a)