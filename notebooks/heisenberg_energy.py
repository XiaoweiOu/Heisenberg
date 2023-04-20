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

## Define the model
square2d = Hypercube(length=6, dimension=2, pbc=True, next_nearest=True)
hamil = HeisenbergJ1J2(square2d, j1=1.0, j2=0.0, total_sz=0.0)
reference_energy = -97.766

## Define the sampler
sampler = MetropolisExchange(num_samples=1000)

## Define the neural networks model
initializer = partial(np.random.normal, loc=0.0, scale=0.01)
mlp = CNN(length = square2d.length, num_points = square2d.num_points)

## Define hyperparameters for learner
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
stopping_threshold = 0.01
num_epochs = 1000

## Training Process
learner = KerasLearner(hamiltonian = hamil, model = mlp, sampler = sampler, optimizer = optimizer,
                  num_epochs = num_epochs, stopping_threshold = stopping_threshold,
                  observables = [], reference_energy=reference_energy)
learner.learn()