import matplotlib.pyplot as plt
import csv
import numpy as np

result_file = 'result/CNN_MSR_SUN6_j205_energy.csv'
epoch = []
energy = []
with open(result_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        epoch.append(int(row[0]))
        energy.append(float(row[1]))

plt.plot(energy)
plt.xlabel('epoch')
plt.ylabel('energy')
plt.text(len(energy)*0.5,energy[0],"<E> in the last 10 epochs: %.3f" % (np.mean(energy[-10:])))
plt.savefig('./draw/energy.jpg')