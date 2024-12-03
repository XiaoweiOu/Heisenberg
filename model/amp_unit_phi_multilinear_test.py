import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpUnitPhiMultiLinearTest(Model):
    """
    Amplitude network: unit
    Phase network: two dense layers(RandomNormal initializer)

    lnA=0 to make all configurations equally probable.
    Check if this setting can train the sign with small |psi(\sigma)|^2 correctly.
    """
    def __str__(self):
        return "Amplitude network: unit Phase: linear layer"

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,),dtype=tf.float64)

        ## phase
        dense1_phi = layers.Dense(units=8, activation='relu', name = 'dense1_phi')(inputs)
        dense2_phi = layers.Dense(units=8, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = layers.Dense(units=1, name = 'dense3_phi')(dense2_phi)
        phi = tf.squeeze(phi,axis=-1)

        ## amplitude
        lnA = 0.
        self.num_param_amp = 0

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)