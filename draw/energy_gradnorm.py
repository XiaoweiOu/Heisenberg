import matplotlib.pyplot as plt
import csv
import numpy as np
plt.rcParams.update({'font.size': 13})

result_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_pureSR_symmetrized_energy.csv'
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

plt.plot(energy_filtered[:2000])
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.savefig('./draw/energy.jpg')
print("<E> in the last 30 epochs: %.3f" % (np.mean(energy_filtered[2000-30:2000])))

plt.clf()
plt.cla()
fig, ax = plt.subplots()
ax.plot((np.array(energy_filtered[:2000])-reference_energy)/number_particle)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\Delta$E/N')
plt.yscale('log')
plt.savefig('./draw/energy_precision.jpg')

plt.clf()
plt.cla()
plt.plot(grad_amp_before[:2000],label='Gradient of log-amplitude')
plt.plot(grad_phi_before[:2000],label='Gradient of phase')
plt.xlabel('Iteration')
plt.ylabel('Norm of gradient')
plt.legend()
plt.savefig('./draw/gradnorm_before.jpg')

plt.clf()
plt.cla()
plt.plot(grad_amp_after[:2000],label='Gradient of log-amplitude')
plt.plot(grad_phi_after[:2000],label='Gradient of phase')
plt.xlabel('Iteration')
plt.ylabel('Norm of gradient')
plt.legend()
plt.savefig('./draw/gradnorm_after.jpg')

plt.clf()
plt.cla()
plt.plot(learning_rate[:2000],'o',markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Learning rate')
plt.savefig('./draw/learning_rate.jpg')