import sys
import os
sys.path.append('..')
sys.path.append('.')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Set the same seed (old:22)
np.random.seed(8) #Previously 8 works for MinSR
tf.random.set_seed(8)

from graph import Hypercube
from hamiltonian import HeisenbergJ1J2
from sampler import MetropolisExchange
from model import *
from learner import Learner,KerasLearner
from symmetry import C4v

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
# tf.keras.backend.set_floatx('float64') # Improve accuracy
# tf.config.experimental.enable_tensor_float_32_execution(False)

## Reference energy: j1=1, j2=0
reference_energy = {'4':-11.23,'6':-24.44,'8':-43.10,'10':-67.15}

## Define the physical system
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=False)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.0, total_sz=0.0)

## Define the sampler
sampler = MetropolisExchange(num_samples=2000, graph=square2d)

## Define the symmetry
symmetry = C4v(length = square2d.length)

## If load trained network and initial samples
learning_rate = 0.04 #default=0.04
load_model_path = 'result/AmpCNNPhiMultiLinearTest_SUN6_minsrmass4_symmetrized_adaptivestep_beta0.05' #'result/AmpCNNPhiMultiLinearTest_SUN6_minsr_recover'
initial_sample_path = 'result/AmpCNNPhiMultiLinearTest_SUN6_minsrmass4_symmetrized_adaptivestep_beta0.05_samples.npy' #'result/AmpCNNPhiMultiLinearTest_SUN6_minsr_recover_samples.npy'
if load_model_path is not None:
    learning_rate = 0.2 #last step size

## Define the neural networks model
mlp = AmpCNNPhiMultiLinearTest(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path, symmetry = symmetry)

## (!! Need to set every time !!) Define the information for this run
# current run name: could be the same as load run name
run_name = mlp.__class__.__name__+'_SUN'+str(square2d.length)+'_minsrmass4_symmetrized_adaptivestep_beta0.05'
is_save = True

## Define hyperparameters for learner
optimizer = tf.keras.optimizers.SGD(learning_rate, momentum = 0.5, nesterov = True)
stopping_threshold = 0.01
num_epochs = 10000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold, observables = [],
                  reference_energy = reference_energy[str(square2d.length)], use_gradient = 'minsr',
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name,
                  use_adaptive_step_size = True)
learner.learn()