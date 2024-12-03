##### NOTE #####
## Latex legend r'$\mu = 0.3$'

import matplotlib.pyplot as plt
import csv
import numpy as np
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({'axes.labelweight': 'normal'})
plt.rc('axes', linewidth = 1.5)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)

color1 = 'cornflowerblue'
color2 = 'forestgreen'

number_particle = 6 ** 2
reference_energy = -24.44 # -24.44 # -18.135
reference_energy1 = -24.44
reference_energy2 = -18.135

result_file = 'result/AmpCNNPhiLinearLarger_SUN6_minsr_mass2_j205_energy.csv'
label1= r'$J_2/J_1=0$'
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

# result_file = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_j205_energy.csv'
# label2= r'$J_2/J_1=0.5$'
# energy2 = []
# with open(result_file,'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     fields = next(csvreader)
#     for row in csvreader:
#         energy2.append(float(row[1]))

# energy2_filtered = energy2[:-30]
# for i in range(len(energy2)-30):
#     energy2_filtered[i] = np.mean(energy2[i:i+30])

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot((np.array(energy1_filtered[:2000])-reference_energy1)/number_particle,label=label1,color=color1)
# ax.plot((np.array(energy2_filtered[:2000])-reference_energy2)/number_particle,label=label2,color=color2)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
ax.set_yscale('log')
ax.grid(which='major', axis='both')
ax.legend()
plt.savefig('./draw/energy_compare_curve.pdf',bbox_inches = 'tight', pad_inches=0.1)

'''# Scatter plot to compare the final converging values
final_energy1 = (np.mean(energy1[2000-30:2000])-reference_energy)/number_particle
final_energy2 = (np.mean(energy2[2000-30:2000])-reference_energy)/number_particle

final_energy1 = 0.0034
final_energy2 = 0.0049
label1 = 'data augmentation'
label2 = 'representative'

plt.clf()
plt.cla()
fig, ax = plt.subplots()
plt.scatter([1,2],[final_energy1,final_energy2])
plt.xticks([1,2],[label1,label2])
plt.yscale('log')
ax.set_ylabel(r'$\Delta$E/N')
plt.savefig('./draw/energy_compare_final.jpg')'''
