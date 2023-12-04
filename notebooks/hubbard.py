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
from hamiltonian import Hubbard
from sampler import MetropolisHubbard
from model import AmpDensePhiDenseHubbard
from learner import KerasLearner,TestKerasLearner

reference_energy = {'2':-4.205,'4':-6.457}

## Define the physical system
square2d = Hypercube(length=2, dimension=2, pbc=True, next_nearest=False)
hamil = Hubbard(square2d, t_coeff=1, U_coeff=8)

## Define the sampler
sampler = MetropolisHubbard(num_samples=500, num_sites=square2d.num_points)

## If load trained network and initial samples
load_model_path = None #'result/AmpDensePhiDenseHubbard_hubbard2'
initial_sample_path = None #'result/AmpDensePhiDenseHubbard_hubbard2_samples.npy'

## Define the neural networks model
mlp = AmpDensePhiDenseHubbard(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path)

## (!! Need to set every time !!) Define the information for this run
# current run name: could be the same as load run name
run_name = mlp.__class__.__name__+'_hubbard'+str(square2d.length)
is_save = True

## Define hyperparameters for learner
initial_learning_rate = 0.001

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=50,
#     decay_rate=0.95,
#     staircase=True)

### TEST ### Nesterov accelerated gradient descent
optimizer = tf.keras.optimizers.SGD(learning_rate = initial_learning_rate, momentum = 0.3, nesterov=True)

stopping_threshold = 0.01
num_epochs = 200

## Training Process
learner = TestKerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold, observables = [],
                  reference_energy = reference_energy[str(square2d.length)], use_gradient = 'sr',
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name)
learner.learn()
