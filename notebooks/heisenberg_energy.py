import sys
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
from model import MLPComplexTanh,CNN
from learner import Learner,KerasLearner
from logger import Logger

## Reference energy: j1=1, j2=0
reference_energy = {'4':-11.23,'6':-24.44,'8':-43.10,'10':-67.15}

## Define the physical system
square2d = Hypercube(length=8, dimension=2, pbc=True, next_nearest=False)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.0, total_sz=0.0)

## Define the information for this run
run_name = 'CNN_MSR_SUN'+str(square2d.length)+'_test2' # current run name: could be the same as load run name
load_model_path = './result/CNN_MSR_SUN8' #relative path to the folder of loaded model
initial_sample_path = './result/CNN_MSR_SUN8_samples.csv' #relative path to the file of initial sample
is_save = True

## Define the sampler
sampler = MetropolisExchange(num_samples=2000, graph=square2d)

## Define the neural networks model
mlp = CNN(length = square2d.length, dimension = square2d.dimension, loadpath = load_model_path)

## Define hyperparameters for learner
learning_rate = 0.04
optimizer = tf.keras.optimizers.SGD(learning_rate)
stopping_threshold = 0.01
num_epochs = 1000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold,
                  observables = [], reference_energy = reference_energy[str(square2d.length)],
                  initial_sample_path = initial_sample_path, is_save = is_save, run_name = run_name)
learner.learn()