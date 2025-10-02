import numpy as np
from collections import Counter

sample_file = 'result/AmpCNNPhiMultiLinearTest_SUN6_pureSR_symmetrized_j205_samples.npy'
samples = np.load(sample_file)
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