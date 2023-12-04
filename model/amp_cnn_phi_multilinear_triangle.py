import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNPhiMultiLinearTriangle(Model):
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
        inputs_pbc = self.PBCs(inputs_re,padding=5)#kernel_size=6
        conv_A = layers.Conv2D(filters=64, kernel_size=6, padding='VALID', name='conv_amp', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='relu')(inputs_pbc)
        self.num_param_amp = (6*6+1)*64
        pool_A,A_indices = tf.math.top_k(conv_A, k=1)
        lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])

        ## phase
        # a simple linear function
        dense1_phi = layers.Dense(units=64, activation='relu', name = 'dense1_phi')(inputs)
        dense2_phi = layers.Dense(units=64, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = tf.reduce_sum(dense2_phi,axis=-1)

        # triangle wave
        period = 2*np.pi
        phi_tri = tf.complex(4*tf.math.abs(phi/period - tf.math.floor(phi/period+0.5))-1, 4*tf.math.abs(phi/period - 0.5 - tf.math.floor(phi/period))-1)
        phi_tri = tf.math.log(phi_tri)

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,0.) + phi_tri

        return tf.keras.Model(inputs=inputs, outputs=outputs)