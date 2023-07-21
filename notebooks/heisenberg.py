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
reference_energy = {'4':-11.23,'6':-24.44,'8':-43.10,'10':-67.15}

## Define the physical system
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=False)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.0, total_sz=0.0)

## Define the sampler
sampler = MetropolisExchange(num_samples=2000, graph=square2d)

## If load trained network and initial samples
load_model_path = None
initial_sample_path = None

## Define the neural networks model
mlp = AmpCNNPhiPhasor(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path)

## (!! Need to set every time !!) Define the information for this run
# current run name: could be the same as load run name
run_name = mlp.__class__.__name__+'_SUN'+str(square2d.length)
is_save = True

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