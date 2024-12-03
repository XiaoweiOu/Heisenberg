import os
import csv
import numpy as np
import tensorflow as tf
import statistics as stt
import matplotlib.pyplot as plt
from tensorflow import keras

model = tf.keras.models.load_model('result/AmpCNNPhiPhasorHubbard_hubbard2')

#configurations used in the training
mcmc_data = [[[1, 1], [1, 0], [0, 0], [0, 1]],
             [[1, 1], [1, 0], [0, 1], [0, 0]],
             [[1, 0], [1, 1], [0, 1], [0, 0]]]

#six dominant configurations in the exact diagonalization
exact_data = [[[1, 1], [1, 1], [0, 0], [0, 0]],
[[1, 1], [1, 0], [0, 0], [0, 1]],
[[1, 1], [0, 1], [1, 0], [0, 0]],
[[1, 1], [0, 0], [1, 1], [0, 0]],
[[1, 1], [1, 0], [0, 1], [0, 0]],
[[1, 0], [1, 1], [0, 1], [0, 0]]]

#prediction
pred = model(tf.constant(mcmc_data))
wfs = tf.math.exp(pred)
print('### INFO ### log_wfs',pred)
print('### INFO ### wave function from model prediction for the MCMC configurations',wfs)
print('### INFO ### probability from model prediction for the MCMC configurations',wfs*tf.math.conj(wfs))

print('\n')

pred = model(tf.constant(exact_data))
wfs = tf.math.exp(pred)
print('### INFO ### log_wfs',pred)
print('### INFO ### wave function from model prediction for the exact dominant configurations',wfs)
print('### INFO ### probability from model prediction for the exact dominant configurations',wfs*tf.math.conj(wfs))