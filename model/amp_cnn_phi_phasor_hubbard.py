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

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,2),dtype=tf.float64)
        #inputs_re = layers.Reshape((length,length,2),input_shape=(length ** dimension,2))(inputs)
        
        ## amplitude
        #inputs_pbc = self.PBCs(inputs_re,padding=int((5-1)/2))#kernel_size=5
        #conv_A = layers.Conv2D(filters=16, kernel_size=3, padding='VALID', name='conv_amp', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='relu')(inputs_pbc)
        #conv_A = tf.math.abs(conv_A)
        #pool_A,A_indices = tf.math.top_k(conv_A, k=1)
        #lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])

        inputs_flt = layers.Flatten()(inputs)
        dense1 = layers.Dense(units=100, kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.005))(inputs_flt)
        dense2 = layers.Dense(units=100, kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.005))(dense1)
        lnA = layers.Dense(units=1, kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.005))(dense2)
        lnA = tf.squeeze(lnA)

        ## phase
        # phasor: a regular cnn layer for phi
        #input_phi = self.PBCs(inputs_re,padding=int((7-1)/2))#kernel_size=7
        #convPhi = layers.Conv2D(filters=24, kernel_size=3, padding='VALID', name='convPhi_1',kernel_initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=1.0),activation=None)(input_phi)
        #out1 = tf.math.exp(tf.complex(0.,convPhi))
        #out2 = tf.math.reduce_sum(out1,axis=[-1,-2,-3])
        #phi = tf.math.angle(out2)

        batch_size = tf.shape(inputs)[0]
        phi = tf.zeros(shape=(batch_size,))

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
