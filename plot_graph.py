import matplotlib.pyplot as plt

#rate change
source_sum = [1625, 1687.5, 1750, 1812.5, 1875]
uniform = [0.00480761, 0.00487737, 0.00497003, 0.00504967, 0.00511887]
barrier = [0.00375296, 0.00381381, 0.00388169, 0.00394348, 0.00400407]
proj = [0.00376578, 0.00383409, 0.00390587, 0.00396181, 0.00402157]
legacy = [0.00320459, 0.00377052, 0.00457402, 0.00585761, 0.00868518]

plt.figure(1)
plt.plot(source_sum, uniform, '-o', label='Uniform')
plt.plot(source_sum, barrier, ':v', label='Barrier')
plt.plot(source_sum, proj, '-.^', label='Projected')
plt.plot(source_sum, legacy, '--s' ,label='Legacy')
plt.xlabel('Sum of Source Rates')
plt.ylabel('Total Service Time')
plt.legend()


layer1_sum = [2700, 2925, 3150, 3375, 3600]
uniform1 = [0.00605465, 0.00541926, 0.00497003, 0.00463654, 0.00439489]
barrier1 = [0.00445962, 0.00414364, 0.00388168, 0.00366541, 0.0034974]
proj1 = [0.00448198, 0.00415399, 0.00390587, 0.00368297, 0.00350338]
legacy1 = [0.00455244, 0.00451871, 0.00457402, 0.00450228, 0.00456761]

plt.figure(2)
plt.plot(layer1_sum, uniform1, '-o', label='Uniform')
plt.plot(layer1_sum, barrier1, ':v', label='Barrier')
plt.plot(layer1_sum, proj1, '-.^', label='Projected')
plt.plot(layer1_sum, legacy1, '--s' ,label='Legacy')
plt.xlabel('Sum of Layer1 Rates')
plt.ylabel('Total Service Time')
plt.legend()


layer2_sum=[2600, 2700, 2800, 2900, 3000]
uniform2 = [0.00511736, 0.00503588, 0.00497003, 0.00490825, 0.00486236]
barrier2 = [0.00396383, 0.00392512, 0.00388169, 0.00383407, 0.00380493]
proj2 = [0.00399754, 0.00394268, 0.00390587, 0.00385688, 0.00380873]
legacy2 = [0.00451143, 0.00456977, 0.00457402, 0.0046013, 0.00446863]

plt.figure(3)
plt.plot(layer2_sum, uniform2, '-o', label='Uniform')
plt.plot(layer2_sum, barrier2, ':v', label='Barrier')
plt.plot(layer2_sum, proj2, '-.^', label='Projected')
plt.plot(layer2_sum, legacy2, '--s' ,label='Legacy')
plt.xlabel('Sum of Layer2 Rates')
plt.ylabel('Total Service Time')
plt.legend()
plt.xticks([2600, 2700, 2800, 2900, 3000])
plt.yticks([0.0050, 0.0045, 0.0040])

#vol change
#0.64
vol = [0.7111111, 0.8, 0.9]
uniform_v1 = [0.00481217, 0.00497909, 0.00516913]
barrier_v1 = [0.00375076, 0.00387724, 0.00404659]
proj_v1 = [0.00376706, 0.00389519, 0.00406601]
legacy_v1 = [0.00458648, 0.00454547, 0.0045871]
efficiency_v1 = [0.732593, 0.744762, 0.759577]

plt.figure(4)
plt.plot(vol, uniform_v1, '-o', label='Uniform')
plt.plot(vol, barrier_v1, ':v', label='Barrier')
plt.plot(vol, proj_v1, '-.^', label='Projected')
plt.plot(vol, legacy_v1, '--s', label='Legacy')
plt.xlabel('Decrement of Layer1')
plt.ylabel('Total Service Time')
plt.legend()
plt.xticks([0.7, 0.8, 0.9])
plt.yticks([0.0050, 0.0045, 0.0040])
plt.figure(5)
plt.plot(vol, efficiency_v1, '-o')
plt.xlabel('Decrement of Layer1')
plt.ylabel('Bandwidth Efficiency')
plt.xticks([0.7, 0.8, 0.9])

plt.figure(6)
plt.plot(vol, uniform_v1[::-1], '-o', label='Uniform')
plt.plot(vol, barrier_v1[::-1], ':v', label='Barrier')
plt.plot(vol, proj_v1[::-1], '-.^', label='Projected')
plt.plot(vol, legacy_v1[::-1], '--s', label='Legacy')
plt.xlabel('Decrement of Layer2')
plt.ylabel('Total Service Time')
plt.legend()
plt.xticks([0.7, 0.8, 0.9])
plt.yticks([0.0040, 0.0045, 0.0050])
plt.figure(7)
plt.plot(vol, efficiency_v1[::-1], '-o')
plt.xlabel('Decrement of Layer2')
plt.ylabel('Bandwidth Efficiency')
plt.xticks([0.7, 0.8, 0.9])

#0.01
vol2 = [0.01, 0.1,  1]
uniform_v2 =[0.0036941, 0.00376924, 0.0051406]
barrier_v2 =[0.00275275, 0.00281177, 0.00396074]
proj_v2 = [0.00276193, 0.00283321, 0.00397077]
legacy_v2 = [0.00459566, 0.00461283, 0.00440168]
efficiency_v2 = [0.57381, 0.548095, 0.715238]

plt.figure(8)
plt.plot(vol2, uniform_v2, '-o', label='Uniform')
plt.plot(vol2, barrier_v2, ':v', label='Barrier')
plt.plot(vol2, proj_v2, '-.^', label='Projected')
plt.plot(vol2, legacy_v2, '--s', label='Legacy')
plt.xlabel('Decrement of Layer1')
plt.ylabel('Total Service Time')
plt.legend()
plt.figure(9)
plt.plot(vol2, efficiency_v2, '-o')
plt.xlabel('Decrement of Layer1')
plt.ylabel('Bandwidth Efficiency')

uniform_v21 = [0.0051406, 0.00376924, 0.0036941]
barrier_v21 = [0.00396074, 0.00281177, 0.00275275]
proj_v21 = [0.00397077, 0.00283321, 0.00276193]
legacy_v21 = [0.00440168, 0.00461283, 0.00459566]
efficiency_v21 = [0.715238, 0.548095, 0.57381]

plt.figure(10)
plt.plot(vol2, uniform_v21, '-o', label='Uniform')
plt.plot(vol2, barrier_v21, ':v', label='Barrier')
plt.plot(vol2, proj_v21, '-.^', label='Projected')
plt.plot(vol2, legacy_v21, '--s', label='Legacy')
plt.xlabel('Decrement of Layer2')
plt.ylabel('Total Service Time')
plt.legend()
plt.figure(11)
plt.plot(vol2, efficiency_v21, '-o')
plt.xlabel('Decrement of Layer2')
plt.ylabel('Bandwidth Efficiency')


#loc change
area = [13.3*13.3, 24*24, 223.6*223.6, 316.6*316.6, 387.3*387.3, 469.8*469.8]
uniform_l = [0.00453855, 0.00496639, 0.0128247, 0.0164602, 0.0192825, 0.0225108]
barrier_l = [0.00368161, 0.00387546, 0.00756083, 0.009280803, 0.0104368, 0.0118633]
proj_l = [0.00369384, 0.00389243, 0.00755169, 0.00920971, 0.0104517, 0.0118551]
legacy_l = [0.004297, 0.00459959, 0.0091127, 0.0112205, 0.0128381, 0.0147556]

plt.figure(12)
plt.plot(area, uniform_l, '-o', label='Uniform')
plt.plot(area, barrier_l, ':v', label='Barrier')
plt.plot(area, proj_l, '-.^', label='Projected')
plt.plot(area, legacy_l, '--s', label='Legacy')
plt.xlabel('Area(km^2)')
plt.ylabel('Total Service Time')
plt.legend()
plt.yticks([0.0050, 0.0100, 0.0150, 0.0200])




