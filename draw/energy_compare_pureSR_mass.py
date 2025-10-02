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

result_file = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_pureSR_symmetrized_energy.csv'
label1= r'$m=1$'
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

energy1_filtered = np.append(energy1_filtered, [energy1_filtered[-1]]*(2000 - len(energy1_filtered)))

result_file = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_energy.csv'
label2= r'$m=4$ SR'
energy2 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy2.append(float(row[1]))

energy2_filtered = energy2[:-30]
for i in range(len(energy2)-30):
    energy2_filtered[i] = np.mean(energy2[i:i+30])

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_minsrmass4_symmetrized_adaptivestep_smallerbeta_energy.csv'
label3= r'$m=4$ MinSR'
energy3 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy3.append(float(row[1]))

energy3_filtered = energy3[:-30]
for i in range(len(energy3)-30):
    energy3_filtered[i] = np.mean(energy3[i:i+30])

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot((np.array(energy1_filtered[:2000])-reference_energy)/number_particle,label=label1)
ax.plot((np.array(energy2_filtered[:2000])-reference_energy)/number_particle,label=label2)
ax.plot((np.array(energy3_filtered[:2000])-reference_energy)/number_particle,label=label3)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
ax.set_yscale('log')
ax.grid(which='major', axis='both')
ax.legend()
plt.savefig('./draw/energy_compare_curve_pureSR_mass.pdf',bbox_inches = 'tight', pad_inches=0.1)