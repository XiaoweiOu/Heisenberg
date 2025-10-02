import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNComplexPhiMultiLinear(Model):
    """
    Amplitude network: one CNN layer(16;5*5;GlorotNormal)
    Phase network: two dense layers(RandomNormal initializer)
    """
    def __str__(self):
        return "Amplitude network: CNN Phase: linear layer"

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,),dtype=tf.float64)
        inputs_re = layers.Reshape((length,length),input_shape=(length ** dimension,))(inputs)

        ## expand a dimension for feature at the end
        inputs_re = tf.expand_dims(inputs_re,-1)
        
        ## amplitude
        #  kernel_initializer=tf.keras.initializers.GlorotNormal()
        inputs_pbc = self.PBCs(inputs_re,padding=5)#kernel_size=6
        conv_A = layers.Conv2D(filters=128, kernel_size=6, padding='VALID', name='conv_amp', kernel_initializer=tf.keras.initializers.GlorotNormal())(inputs_pbc)
        self.num_param_amp = (6*6+1)*128
        conv_A = tf.math.abs(conv_A)
        pool_A = tf.math.reduce_max(conv_A, axis=-1)
        # lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])
        
        pool_A_reshape = layers.Reshape((length ** dimension,),input_shape=(length,length))(pool_A)
        real = layers.Dense(units=1)(pool_A_reshape)
        imag = layers.Dense(units=1)(pool_A_reshape)
        lnA = tf.complex(real, imag)
        lnA = tf.squeeze(lnA,axis=-1)

        # splits = tf.split(conv_A, num_or_size_splits = 2, axis = -1)
        # first_max = tf.math.reduce_max(splits[0], axis = -1)
        # second_max = tf.math.reduce_max(splits[1], axis = -1)
        # combine_max = tf.complex(first_max, second_max)

        # # logsumexp
        # exp_combine_max = tf.math.exp(combine_max)
        # sum_exp_combine_max = tf.math.reduce_sum(exp_combine_max, axis=[-1,-2])
        # lnA = tf.math.log(sum_exp_combine_max)

        ## phase
        dense1_phi = layers.Dense(units=16, activation='relu', name = 'dense1_phi')(inputs)
        dense2_phi = layers.Dense(units=16, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = layers.Dense(units=1, name = 'dense3_phi')(dense2_phi)
        phi = tf.squeeze(phi,axis=-1)

        ## output
        # stack the values for output
        outputs = lnA + tf.complex(0.,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)