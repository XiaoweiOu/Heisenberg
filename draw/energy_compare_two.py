##### NOTE #####
## Latex legend r'$\mu = 0.3$'

import matplotlib.pyplot as plt
import csv
import numpy as np
plt.rcParams['font.size'] = 13
# plt.rcParams['figure.figsize'] = [8, 6] # for square canvas
# plt.rcParams['figure.subplot.left'] = 0.2
# plt.rcParams['figure.subplot.bottom'] = 0.1
# plt.rcParams['figure.subplot.right'] = 0.8
# plt.rcParams['figure.subplot.top'] = 0.9

number_particle = 6 ** 2
reference_energy = -24.44 # -24.44 # -18.135

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_energy.csv'
label1= 'mass=4'
energy1 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy1.append(float(row[1]))

# energy filter: use a window of size 30
energy1_filtered = energy1[:-30]
for i in range(len(energy1)-30):
    energy1_filtered[i] = np.mean(energy1[i:i+30])

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_pureSR_symmetrized_energy.csv'
label2= 'pure SR'
energy2 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy2.append(float(row[1]))

energy2_filtered = energy2[:-30]
for i in range(len(energy2)-30):
    energy2_filtered[i] = np.mean(energy2[i:i+30])

energy2_filtered = energy2

fig, ax = plt.subplots()
ax.plot((np.array(energy1_filtered[:2000])-reference_energy)/number_particle,label=label1)
ax.plot((np.array(energy2_filtered[:2000])-reference_energy)/number_particle,label=label2)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
plt.yscale('log')
plt.legend()
plt.savefig('./draw/energy_compare_curve.jpg')

# Scatter plot to compare the final converging values
'''final_energy1 = (np.mean(energy1[2000-30:2000])-reference_energy)/number_particle
final_energy2 = (np.mean(energy2[2000-30:2000])-reference_energy)/number_particle

plt.clf()
plt.cla()
fig, ax = plt.subplots()
plt.scatter([1,2],[final_energy1,final_energy2])
plt.xticks([1,2],[label1,label2])
plt.yscale('log')
ax.set_ylabel(r'$\Delta$E/N')
plt.savefig('./draw/energy_compare_final.jpg')'''