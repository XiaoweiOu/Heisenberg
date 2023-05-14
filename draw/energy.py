import matplotlib.pyplot as plt
import csv
import numpy as np

result_file = './result/CNN_MSR_SUN6_2000sample_test3_energy.csv'
epoch = []
energy = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        epoch.append(int(row[0]))
        energy.append(float(row[1]))

print("Average energy in the last 20 iterations:",np.mean(energy[-20:]))

plt.plot(energy)
plt.xlabel('epoch')
plt.ylabel('energy')
plt.savefig('energy.png')