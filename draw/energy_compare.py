##### NOTE #####
## Latex legend r'$\mu = 0.3$'

import matplotlib.pyplot as plt
import csv
import numpy as np
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({'axes.labelweight': 'normal'})
# plt.rcParams['figure.subplot.left'] = 0.1
# plt.rcParams['figure.subplot.bottom'] = 0.1
# plt.rcParams['figure.subplot.right'] = 0.9
# plt.rcParams['figure.subplot.top'] = 0.9
plt.rc('axes', linewidth = 1.5)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)

color0 = 'blueviolet'
color1 = 'cornflowerblue'
color2 = 'forestgreen'

number_particle = 6 ** 2
reference_energy = -24.44 # -24.44 # -18.135

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_tritildemass11_linearsolve_epsilon_symmetrized_LR0.006_energy.csv'
label1= r'$\tilde{\epsilon}$ Method (SR)'
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

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_minsrtritildemass11_linearsolve_epsilon_symmetrized_LR0.006_energy.csv'
label2= r'$\tilde{\epsilon}$ Method (MinSR)'
energy2 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy2.append(float(row[1]))

# energy filter: use a window of size 30
energy2_filtered = energy2[:-30]
for i in range(len(energy2)-30):
    energy2_filtered[i] = np.mean(energy2[i:i+30])

print(f'epsilon tilde LR=0.006:\nSR: {(np.mean(energy1_filtered[-10])-reference_energy)/36}\nMinSR: {(np.mean(energy2_filtered[-10])-reference_energy)/36}')

#result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_energy.csv
result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_energy.csv'
label5= r'$\tilde{O}$ Method (SR)'
energy5 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy5.append(float(row[1]))

energy5_filtered = energy5[:-30]
for i in range(len(energy5)-30):
    energy5_filtered[i] = np.mean(energy5[i:i+30])

print(f'two Otilde SR: {(np.mean(energy5_filtered[-10])-reference_energy)/36}')

#result/AmpCNNPhiMultiLinearTest_SUN6_minsrmass4_symmetrized_adaptivestep_beta0.05_energy.csv 
result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_minsrmass4_symmetrized_adaptivestep_beta0.05_energy.csv'
label9= r'$\tilde{O}$ Method (MinSR)'
energy9 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy9.append(float(row[1]))

energy9_filtered = energy9[:-30]
for i in range(len(energy9)-30):
    energy9_filtered[i] = np.mean(energy9[i:i+30])

fig, ax = plt.subplots(figsize=[8,6], dpi=120, nrows=1, ncols=1)
ax.plot((np.array(energy1_filtered[:3000])-reference_energy)/number_particle,label=label1,markersize=2)
ax.plot((np.array(energy2_filtered[:3000])-reference_energy)/number_particle,label=label2,markersize=2)
# ax.plot((np.array(energy3_filtered[:])-reference_energy)/number_particle,label=label3,markersize=2)
# ax.plot((np.array(energy4_filtered[:])-reference_energy)/number_particle,label=label4,markersize=2)
ax.plot((np.array(energy5_filtered[:3000])-reference_energy)/number_particle,label=label5,markersize=2)
# ax.plot((np.array(energy6_filtered[:])-reference_energy)/number_particle,label=label6,markersize=2)
# ax.plot((np.array(energy7_filtered[:])-reference_energy)/number_particle,label=label7,markersize=2)
# ax.plot((np.array(energy8_filtered[:])-reference_energy)/number_particle,label=label8,markersize=2)
ax.plot((np.array(energy9_filtered[:3000])-reference_energy)/number_particle,label=label9,markersize=2)
# ax.plot((np.array(energy10_filtered[:])-reference_energy)/number_particle,label=label10,markersize=2)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
# ax.grid(which='major', axis='both')
ax.set_yscale('log')
ax.legend()
plt.savefig('./draw_v3/energy_compare.pdf', pad_inches=0.1)