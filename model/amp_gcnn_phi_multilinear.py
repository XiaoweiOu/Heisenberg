import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import Model
from keras_gcnn.layers import GConv2D

class AmpGCNNPhiMultiLinear(Model):
    """
    Amplitude network: two layers of GCNN
    Phase network: two layers of GCNN
    """
    def create_model(self,length,dimension):
        """
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,),dtype=tf.float64)
        inputs_re = layers.Reshape((length,length),input_shape=(length ** dimension,))(inputs)

        ## expand a dimension for feature at the end
        inputs_re = tf.expand_dims(inputs_re,-1)

        ## amplitude
        inputs_pbc = self.PBCs(inputs_re,padding=4)#kernel_size=5
        gcnn_A = GConv2D(8, h_input='Z2', h_output='D4', kernel_size=5, padding='VALID')(inputs_pbc)
        gcnn_A = layers.Activation('relu')(gcnn_A)

        gcnn_A = self.PBCs(gcnn_A,padding=4)#kernel_size=5
        gcnn_A = GConv2D(8, h_input='D4', h_output='D4', kernel_size=5, padding='VALID')(gcnn_A)
        gcnn_A = layers.Activation('relu')(gcnn_A)

        gcnn_A = tf.math.abs(gcnn_A)
        pool_A,A_indices = tf.math.top_k(gcnn_A, k=1)
        lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])

        ## phase
        phi = layers.Dense(units=1, kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5))(inputs)
        phi = tf.squeeze(phi)

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)