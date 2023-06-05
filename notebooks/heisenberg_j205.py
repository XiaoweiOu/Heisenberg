import sys
import os
import psutil
sys.path.append('..')
sys.path.append('.')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Set the same seed (old:11)
np.random.seed(22)
tf.random.set_seed(22)

from graph import Hypercube
from hamiltonian import HeisenbergJ1J2
from sampler import MetropolisExchange
from functools import partial
from model import AmpCNNPhiMSR,AmpCNNPhiPhasor,AmpCNNPhiLinear
from learner import Learner,KerasLearner

## Reference energy: j1=1, j2=0
reference_energy = {'4':None,'6':-18.0,'8':None,'10':-49.56}

## Define the physical system
square2d = Hypercube(length=10, dimension=2, pbc=True, next_nearest=True)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.5, total_sz=0.0)

## (!! Need to set every time !!) Define the information for this run
run_name = 'CNN_MSR_SUN'+str(square2d.length)+'_j205' # current run name: could be the same as load run name
load_model_path = None 
initial_sample_path = None 
is_save = True

## Define the sampler
sampler = MetropolisExchange(num_samples=2000, graph=square2d)

## Define the neural networks model
mlp = AmpCNNPhiMSR(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path)

## Define hyperparameters for learner
learning_rate = 0.04
optimizer = tf.keras.optimizers.SGD(learning_rate)
stopping_threshold = 0.01
num_epochs = 1000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold, observables = [],
                  reference_energy = reference_energy[str(square2d.length)], use_gradient = 'sr',
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name)
learner.learn()