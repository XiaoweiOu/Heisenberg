import matplotlib.pyplot as plt
import matplotlib
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

result_file = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_energy.csv'
number_particle = 6 ** 2
reference_energy = -24.44 # -24.44 # -18.135

epoch = []
energy = []
energy_std = []
grad_amp_before = []
grad_phi_before = []
grad_amp_after = []
grad_phi_after = []
learning_rate = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        epoch.append(int(row[0]))
        energy.append(float(row[1]))
        energy_std.append(float(row[2]))
        grad_amp_before.append(float(row[3]))
        grad_phi_before.append(float(row[4]))
        grad_amp_after.append(float(row[5]))
        grad_phi_after.append(float(row[6]))
        learning_rate.append(float(row[7]))

# energy filter: use a window of size 30
energy_filtered = energy[:-30]
for i in range(len(energy)-30):
    energy_filtered[i] = np.mean(energy[i:i+30])

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot(energy_filtered[:2000],color=color1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy')
ax.grid(which='major', axis='both')
plt.savefig('./draw/energy.pdf',bbox_inches = 'tight', pad_inches=0.1)
# print("<E> in the last 30 epochs: %.3f" % (np.mean(energy_filtered[2000-30:2000])))

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot((np.array(energy_filtered[:2000])-reference_energy)/number_particle,color=color1)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
ax.grid(which='major', axis='both')
ax.set_yscale('log')
plt.savefig('./draw/energy_precision.pdf',bbox_inches = 'tight', pad_inches=0.1)

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot(grad_amp_before[:500],label='Gradient of log-amplitude',color=color1)
ax.plot(grad_phi_before[:500],label='Gradient of phase',color=color2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Norm of gradient')
ax.legend()
ax.grid(which='major', axis='both')
plt.savefig('./draw/gradnorm_before.pdf',bbox_inches = 'tight', pad_inches=0.1)

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot(grad_amp_after[:500],label='Gradient of log-amplitude',color=color1)
ax.plot(grad_phi_after[:500],label='Gradient of phase',color=color2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Norm of gradient')
ax.legend()
ax.grid(which='major', axis='both')
plt.savefig('./draw/gradnorm_after.pdf',bbox_inches = 'tight', pad_inches=0.1)

fig, ax = plt.subplots(figsize=[7,6], dpi=120, nrows=1, ncols=1)
ax.plot(learning_rate[:2000],'o',markersize=3,color=color1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Learning rate')
ax.grid(which='major', axis='both')
plt.savefig('./draw/learning_rate.pdf',bbox_inches = 'tight', pad_inches=0.1)