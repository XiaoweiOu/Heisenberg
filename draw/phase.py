import os
import csv
import numpy as np
import tensorflow as tf
import statistics as stt
import matplotlib.pyplot as plt
from tensorflow import keras

import sys
sys.path.append('..')
sys.path.append('.')

from graph import Hypercube
from sampler import MetropolisExchange

def marshall_sign(binary,lattice_size):
    sum_sign=0
    #sum a checkboard sublattice
    for i in range(lattice_size):
        for j in range(lattice_size):
            if i%2==0 and j%2==0:
                sum_sign+=(binary[lattice_size*i+j]+1)/2
            if i%2==1 and j%2==1:
                sum_sign+=(binary[lattice_size*i+j]+1)/2
            
    return (sum_sign%2)*np.pi

# configurations used in the training
mcmc_data = np.load('result/AmpCNNPhiMultiLinear_SUN6_mass6_j205_nesterov_samples.npy')

# some random configurations
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=False)
sampler = MetropolisExchange(num_samples=2000, graph=square2d)
random_data = sampler.get_initial_random_samples(square2d.num_points)

# Network prediction
model = tf.keras.models.load_model('result/AmpCNNPhiMultiLinear_SUN6_mass6_j205_nesterov')
pred = model(tf.constant(mcmc_data))
phase = tf.math.imag(pred).numpy()
phase = phase % (2*np.pi)

# Marshall sign rule
marshall_sign_config=[]
for config in mcmc_data:
    sign = marshall_sign(config,square2d.length)
    marshall_sign_config.append(sign)

# Calculate difference
diff = phase - marshall_sign_config
diff = diff % (2*np.pi)
for ele in diff:
    if ele > np.pi:
        ele = 2*np.pi - ele

# Draw plots
plt.hist(phase,bins=30)
plt.xlabel('phase')
plt.ylabel('Number of configurations')
plt.savefig('./draw/phase.jpg')

plt.clf()
plt.cla()
plt.hist(diff-stt.mean(diff),bins=30)
plt.xlabel('Included angle')
plt.ylabel('Number of configurations')
plt.text(0+2*stt.stdev(diff),800,'std={:.3f}'.format(stt.stdev(diff)))
plt.savefig('./draw/marshall_sign_diff.jpg')

plt.clf()
plt.cla()
plt.hist(marshall_sign_config,bins=30)
plt.xlabel('Marshall sign')
plt.ylabel('Number of configurations')
plt.savefig('./draw/marshall_sign.jpg')

print("### INFO ### plot saved")