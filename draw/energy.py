import matplotlib.pyplot as plt
import csv

result_file = 'result.csv'
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
plt.savefig('energy.png')