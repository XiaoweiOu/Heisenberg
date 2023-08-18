import numpy as np
import tensorflow as tf
import os
import psutil
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNPhiPhasorHubbard(Model):
    """
    Amplitude network: one CNN layer(16;5*5;GlorotNormal)
    Phase network: one dense layer(RandomNormal initializer)
    """
    def __str__(self):
        return "Amplitude network: CNN Phase: linear layer"

    def create_model(self):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(self.length ** self.dimension,2),dtype=tf.float64)
        inputs_flt = layers.Flatten()(inputs)
        
        ## amplitude
        dense1_amp = layers.Dense(units=8, activation='relu', name = 'dense1_amp')(inputs_flt)
        dense2_amp = layers.Dense(units=8, activation='relu', name = 'dense2_amp')(dense1_amp)
        amp = tf.reduce_sum(dense2_amp,axis=-1)

        ## phase
        dense1_phi = layers.Dense(units=8, activation='relu', name = 'dense1_phi')(inputs_flt)
        dense2_phi = layers.Dense(units=8, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = tf.reduce_sum(dense2_phi,axis=-1)

        ## output
        # stack the values for output
        outputs = tf.complex(amp*tf.math.cos(phi),amp*tf.math.sin(phi))

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def derlog(self, x):
        """
        Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$ where W can be the weights or the biases.
        """
        ## Note that the network output here is wfs instead of log_psi!!!
        with tf.GradientTape(persistent=True) as g:
            wfs = self.net(x)
            log_psi = tf.math.log(wfs)
            real_psi = tf.math.real(log_psi)
            imag_psi = tf.math.imag(log_psi)

        #Not using vectorized calculation (pfor = Flase) can prevent the excessiv memory consumption.
        real_jacobians = g.jacobian(real_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)
        imag_jacobians = g.jacobian(imag_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)

        return [tf.complex(real_jacobians[i], imag_jacobians[i]) for i in range(len(real_jacobians))]