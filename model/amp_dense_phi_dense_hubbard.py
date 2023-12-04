import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpDensePhiDenseHubbard(Model):
    """
    Amplitude network: two dense layers
    Phase network: two dense layers
    """
    def __str__(self):
        return "Amplitude network: dense Phase: dense"

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,2),dtype=tf.float64)
        inputs_flt = layers.Flatten()(inputs)
        
        ## amplitude
        dense1_lnA = layers.Dense(units=8, activation='relu', name = 'dense1_lnA', kernel_initializer = tf.keras.initializers.Constant(0.1))(inputs_flt)
        dense2_lnA = layers.Dense(units=8, activation='relu', name = 'dense2_lnA', kernel_initializer = tf.keras.initializers.Constant(0.1))(dense1_lnA)
        lnA = tf.reduce_sum(dense2_lnA,axis=-1)

        # This number is wrong right now!!!
        self.num_param_amp = (length ** dimension * 2 + 1) * 8 + (8+1)*8

        ## phase
        dense1_phi = layers.Dense(units=8, activation='relu', name = 'dense1_phi')(inputs_flt)
        dense2_phi = layers.Dense(units=8, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = tf.reduce_sum(dense2_phi,axis=-1)

        ## output
        # stack the values for output
        # outputs = tf.complex(amp*tf.math.cos(phi),amp*tf.math.sin(phi))
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def log_val(self, x):
        ## How to exclude small wfs???
        log_wfs = self.net(x)
        threshold_real = tf.constant(-20.,shape=tf.shape(log_wfs))

        big_enough = tf.greater(tf.math.real(log_wfs),threshold_real)
        new_log_wfs = tf.complex(tf.where(big_enough,tf.math.real(log_wfs),threshold_real),tf.math.imag(log_wfs))

        return tf.expand_dims(new_log_wfs,axis=-1)