import matplotlib.pyplot as plt
import csv
import numpy as np

result_file = 'result/AmpCNNPhiLinear_SUN6_energy.csv'
label1='sr'
energy1 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy1.append(float(row[1]))

result_file = 'result/AmpCNNPhiLinear_SUN6_mass4_energy.csv'
label2='mass=4'
energy2 = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        energy2.append(float(row[1]))

plt.plot(energy1[:1000],label=label1)
plt.plot(energy2[:1000],label=label2)
plt.xlabel('iteration')
plt.ylabel('energy')
plt.legend()
plt.savefig('./draw/energy_compare.jpg')