import os
import csv
import numpy as np
import tensorflow as tf
import statistics as stt
import matplotlib.pyplot as plt
from tensorflow import keras

model = tf.keras.models.load_model('result/CNN_MSR_SUN6_amp_cnn_phi_linear_amp_penalty_GlorotNormal')

#configurations used in the training
with open('result/CNN_MSR_SUN6_amp_cnn_phi_linear_amp_penalty_GlorotNormal_samples.csv','r') as file:
    mcmc_reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
    mcmc_data = [line for line in mcmc_reader]

#prediction
pred = model.predict(tf.constant(mcmc_data),batch_size=50)
phase = tf.math.imag(pred).numpy()
phase = phase % (2*np.pi)

plt.hist(phase,bins=20)
plt.xlabel('phase')
plt.ylabel('Number of conf')
plt.savefig('./draw/phase.jpg')