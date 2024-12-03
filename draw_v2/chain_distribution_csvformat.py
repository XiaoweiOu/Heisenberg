import numpy as np
from collections import Counter
import csv

sample_file = './result/AmpCNNPhiPhasorHubbard_hubbard2_samples.csv'

with open(sample_file,'r') as csvfile:
    csvreader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    samples = []
    for line in csvreader:
        newline = []
        for ele in line:
            newline.append(np.fromstring(ele[1:-1],dtype=np.float32,sep=', '))
        samples.append(newline)
samples = np.array(samples)
print('INFO: total number of configurations',len(samples))

hash_samples = []
for sample in samples:
    hash_samples.append(sample.tobytes())


dist = Counter(hash_samples)
print('INFO: number of the distinct configurations',len(dist))

state = samples[0]
keys = [np.reshape(np.frombuffer(list(dist.keys())[i],state.dtype).astype(int),state.shape).tolist() for i in range(len(dist))]
zipped = zip(keys,dist.values())
results = sorted(zipped,key = lambda x:x[1])
for ele in results[::-1]:
    print(ele[0],'count:',ele[1],'frequency:',ele[1]/len(samples))

# for i in range(len(dist)):
#     print(np.reshape(np.frombuffer(list(dist.keys())[i],state.dtype),state.shape).tolist())
    
#     print('count:',list(dist.values())[i],'frequency:',list(dist.values())[i]/len(samples))
#     print()