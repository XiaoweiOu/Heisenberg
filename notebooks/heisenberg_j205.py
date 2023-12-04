import sys
import os
sys.path.append('..')
sys.path.append('.')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Set the same seed (old:11)
np.random.seed(22)
tf.random.set_seed(22)

from graph import Hypercube
from hamiltonian import HeisenbergJ1J2,HeisenbergJ1J2Transform
from sampler import MetropolisExchange
from functools import partial
from model import AmpCNNPhiMSR,AmpCNNPhiPhasor,AmpCNNPhiLinear,AmpCNNPhiMultiLinear,AmpCNNPhiMultiLinearTriangle,AmpCNNPhiSiteBond,AmpCNNPhiMultiLinearTest
from learner import Learner,KerasLearner
from symmetry import C4v,C4vZ2

## Reference energy: j1=1, j2=0
reference_energy = {'4':None,'6':-18.135,'8':None,'10':-49.56} #Question: '6' -18.137133

## Define the physical system
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=True)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.5, total_sz=0.0)

## Define the sampler
sampler = MetropolisExchange(num_samples=2000, graph=square2d)

## Define the symmetry
symmetry = C4v(length = square2d.length)

## If load trained network and initial samples
learning_rate = 0.04
load_model_path = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_j205'
initial_sample_path = 'result/AmpCNNPhiMultiLinearTest_SUN6_mass4_symmetrized_adaptivestep_j205_samples.npy'
if load_model_path is not None:
    learning_rate = 0.2

## Define the neural networks model
mlp = AmpCNNPhiMultiLinearTest(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path, symmetry = symmetry)

## (!! Need to set every time !!) Define the information for this run
# current run name: could be the same as load run name
run_name = mlp.__class__.__name__+'_SUN'+str(square2d.length)+'_mass4_symmetrized_adaptivestep_j205'
is_save = True

## Define hyperparameters for learner
optimizer = tf.keras.optimizers.SGD(learning_rate, momentum = 0.5, nesterov = True)
stopping_threshold = 0.01
num_epochs = 2000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold, observables = [],
                  reference_energy = reference_energy[str(square2d.length)], use_gradient = 'amp_penalty',
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name,
                  use_adaptive_step_size = True)
learner.learn()