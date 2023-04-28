import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Set the same seed
np.random.seed(42)
tf.random.set_seed(42)

from graph import Hypercube
from hamiltonian import HeisenbergJ1J2
from sampler import MetropolisExchange
from functools import partial
from model import MLPComplexTanh,CNN
from learner import Learner,KerasLearner
from logger import Logger

## Define the physical system
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=True)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.0, total_sz=0.0)

## Define this run information
reference_energy = {'4':-11.23,'6':-24.44}  # 4*4 lattice = -11.23; 6*6 lattice = -24.44
run_name = 'CNN_MSR_SUN'+str(square2d.length) # current run name: could be the same as load run name
load_model_path = None #'./result/CNN_MSR_SUN6'
initial_sample_path = None #'./result/CNN_MSR_SUN6_samples.csv'
is_save = False

## Define the sampler
sampler = MetropolisExchange(num_samples=1000)

## Define the neural networks model
mlp = CNN(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path)

## Define hyperparameters for learner
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
stopping_threshold = 0.01
num_epochs = 1000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold,
                  observables = [], reference_energy = reference_energy[str(square2d.length)], 
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name)
learner.learn()