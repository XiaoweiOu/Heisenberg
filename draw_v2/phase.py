import os
import csv
import numpy as np
import tensorflow as tf
import statistics as stt
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

import sys
sys.path.append('..')
sys.path.append('.')

from graph import Hypercube
from sampler import MetropolisExchange
from symmetry import C4v
from model import AmpCNNPhiMultiLinear,AmpCNNPhiSiteBond
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({'axes.labelweight': 'normal'})
plt.rc('axes', linewidth = 1.5)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)

color1 = 'cornflowerblue'
color2 = 'forestgreen'

square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=False)
symmetry = C4v(length = square2d.length)

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

##### FIRST ONE: J2/J1=0

# configurations used in the training
label1 = r'$J_2/J_1 = 0$'
mcmc_data1 = np.load('result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_samples.npy')

# Network prediction
load_model_path1 = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep'
mlp1 = AmpCNNPhiMultiLinear(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path1, symmetry = symmetry)
pred1 = mlp1.log_val(tf.constant(mcmc_data1))
phase1 = tf.math.imag(pred1).numpy()
phase1 = phase1 % (2*np.pi)
phase1 = tf.squeeze(phase1)
lnA1 = tf.math.real(pred1).numpy()
lnA1 = lnA1 - tf.reduce_max(lnA1)
lnA1 = tf.squeeze(lnA1)

# Marshall sign rule
marshall_sign_config1=[]
for config1 in mcmc_data1:
    sign1 = marshall_sign(config1,square2d.length)
    marshall_sign_config1.append(sign1)

# Calculate included angle: [0,pi]
diff1 = phase1.numpy() - marshall_sign_config1
diff1 = diff1 % (2*np.pi)
diff1 = diff1 - stt.mean(diff1)

##### SECOND ONE: J2/J1=0.5

# configurations used in the training
label2 = r'$J_2/J_1 = 0.5$'
mcmc_data2 = np.load('result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_j205_samples.npy')

# Network prediction
load_model_path2 = 'result_paper_v2/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_j205'
mlp2 = AmpCNNPhiMultiLinear(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path2, symmetry = symmetry)
pred2 = mlp2.log_val(tf.constant(mcmc_data2))
phase2 = tf.math.imag(pred2).numpy()
phase2 = phase2 % (2*np.pi)
phase2 = tf.squeeze(phase2)
lnA2 = tf.math.real(pred2).numpy()
lnA2 = lnA2 - tf.reduce_max(lnA2)
lnA2 = tf.squeeze(lnA2)

# Marshall sign rule
marshall_sign_config2=[]
for config2 in mcmc_data2:
    sign2 = marshall_sign(config2,square2d.length)
    marshall_sign_config2.append(sign2)

# Calculate included angle: [0,pi]
diff2 = phase2.numpy() - marshall_sign_config2
diff2 = diff2 % (2*np.pi)
diff2 = diff2 - stt.mean(diff2)

# Draw plots
plt.hist(phase1,bins=30)
plt.xlabel('phase')
plt.ylabel('Number of configurations')
plt.savefig('./draw/phase.jpg')

plt.clf()
plt.cla()
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(phase1,lnA1,marker='.',label=label1)
ax.scatter(phase2,lnA2,marker='.',label=label2)
plt.legend(loc='upper center')
plt.savefig('./draw/phase_polar.jpg')

plt.clf()
plt.cla()
plt.hist(marshall_sign_config1,bins=30)
plt.xlabel('Marshall sign')
plt.ylabel('Number of configurations')
plt.savefig('./draw/marshall_sign.jpg')

# How to draw the difference?
fig, ax = plt.subplots(figsize=[7,6], dpi=300, nrows=1, ncols=1)
ax.ecdf(diff1,label = label1,color=color1)
ax.ecdf(diff2,label = label2,color=color2)
print('std_diff1',stt.stdev(diff1))
print('std_diff2',stt.stdev(diff2))
ax.set_xlabel(r'Phase relative to MSR, $\Phi - \Phi_{MSR}$')
ax.set_xlim(left=-0.05,right=0.05)
ax.set_ylabel('Cumulative distribution function')
ax.legend()
ax.grid(which='major',axis='both')
plt.savefig('./draw/marshall_sign_diff.pdf',bbox_inches = 'tight', pad_inches=0.1)

plt.clf()
plt.cla()
plt.scatter(x = marshall_sign_config1,y = phase1,marker='*')
plt.xlabel('Marshall sign')
plt.ylabel('Network prediction')
plt.savefig('./draw/marshall_sign_diff_2D.jpg')


print("### INFO ### plot saved")