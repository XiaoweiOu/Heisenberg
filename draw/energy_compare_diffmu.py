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

number_particle = 6 ** 2
reference_energy = -24.44 # -24.44 # -18.135

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_mu01_energy.csv'
label1= r'$\mu = 0.1$'
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

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_mu03_energy.csv'
label2= r'$\mu = 0.3$'
energy2 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy2.append(float(row[1]))

energy2_filtered = energy2[:-30]
for i in range(len(energy2)-30):
    energy2_filtered[i] = np.mean(energy2[i:i+30])

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_mu05_energy.csv'
label3= r'$\mu = 0.5$'
energy3 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy3.append(float(row[1]))

energy3_filtered = energy3[:-30]
for i in range(len(energy3)-30):
    energy3_filtered[i] = np.mean(energy3[i:i+30])

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_mu07_energy.csv'
label4= r'$\mu = 0.7$'
energy4 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy4.append(float(row[1]))

energy4_filtered = energy4[:-30]
for i in range(len(energy4)-30):
    energy4_filtered[i] = np.mean(energy4[i:i+30])

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_nomu_energy.csv'
label5= 'no momemtum\nmethod'
energy5 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy5.append(float(row[1]))

energy5_filtered = energy5[:-30]
for i in range(len(energy5)-30):
    energy5_filtered[i] = np.mean(energy5[i:i+30])

fig, ax = plt.subplots(figsize=[8,6], dpi=120, nrows=1, ncols=1)
ax.plot((np.array(energy1_filtered[:10000])-reference_energy)/number_particle,label=label1)
ax.plot((np.array(energy2_filtered[:10000])-reference_energy)/number_particle,label=label2)
ax.plot((np.array(energy3_filtered[:10000])-reference_energy)/number_particle,label=label3)
ax.plot((np.array(energy4_filtered[:10000])-reference_energy)/number_particle,label=label4)
ax.plot((np.array(energy5_filtered[:10000])-reference_energy)/number_particle,label=label5)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
ax.grid(which='major', axis='both')
ax.set_yscale('log')
ax.legend()
# plt.savefig('./draw/energy_compare_diffmu_curve.pdf',bbox_inches = 'tight', pad_inches=0.1)
plt.savefig('./draw/energy_compare_diffmu_curve.pdf', pad_inches=0.1)

# Scatter plot to compare the final converging values
final_energy1 = (np.mean(energy1[10000-30:10000])-reference_energy)/number_particle
final_energy2 = (np.mean(energy2[10000-30:10000])-reference_energy)/number_particle
final_energy3 = (np.mean(energy3[10000-30:10000])-reference_energy)/number_particle
final_energy4 = (np.mean(energy4[10000-30:10000])-reference_energy)/number_particle
final_energy5 = (np.mean(energy5[10000-30:10000])-reference_energy)/number_particle

fig, ax = plt.subplots(figsize=[8,6], dpi=120, nrows=1, ncols=1)
ax.scatter([1,2,3,4,5],[final_energy1,final_energy2,final_energy3,final_energy4,final_energy5],s=100)
ax.set_xticks([1,2,3,4,5],[label1,label2,label3,label4,label5])
ax.set_ylabel(r'$\Delta$E/N')
ax.ticklabel_format(axis='y',style='sci',scilimits=[0,0])
ax.grid(which='both', axis='both')
# plt.savefig('./draw/energy_compare_diffmu_final.pdf',bbox_inches = 'tight', pad_inches=0.1)
plt.savefig('./draw/energy_compare_diffmu_final.pdf', pad_inches=0.1)