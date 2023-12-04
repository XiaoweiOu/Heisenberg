import matplotlib.pyplot as plt
import csv
import numpy as np
plt.rcParams.update({'font.size': 13})

result_file = 'result/AmpDensePhiDenseHubbard_hubbard2_energy.csv'
number_particle = 6 ** 2
reference_energy = -4.205

epoch = []
energy = []

with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        epoch.append(int(row[0]))
        energy.append(float(row[1]))

plt.plot(energy[:2000])
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.savefig('./draw/energy.jpg')
print("<E> in the last 30 epochs: %.3f" % (np.mean(energy[-30:])))