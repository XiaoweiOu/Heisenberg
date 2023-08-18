import numpy as np
import tensorflow as tf
import os
import psutil
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNPhiPhasor(Model):
    """
    Amplitude network: one inception layer
    Phase network: conv->exp(i*)->sum->angle
    """
    def __str__(self):
        return "Amplitude network: CNN Phase: phasor(see code for details)"

    def create_model(self):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(self.length ** self.dimension,),dtype=tf.float64)
        inputs_re = layers.Reshape((self.length,self.length),input_shape=(self.length ** self.dimension,))(inputs)

        ## expand a dimension for feature at the end
        inputs_re = tf.expand_dims(inputs_re,-1)
        
        ## amplitude
        def inception(inputs,name,num_channels=2048):
            """
            Inception layer: consist of a 3x3 conv layer, a 5x5 conv layer, and a 3x3 maxpooling layer
            """

            initializer = tf.keras.initializers.HeNormal()
            inputs_3 = self.PBCs(inputs,padding=2)#kernel_size=3
            inputs_5 = self.PBCs(inputs,padding=4)#kernel_size=5
        
            # dropout rate is suggested to between 0.2 and 0.5
            conv3 = layers.Conv2D(num_channels, kernel_size=3, padding='VALID', name=name+'_incep_3',kernel_initializer=initializer, activation='relu')(inputs_3)
            conv3 = layers.Dropout(rate=0.2)(conv3)
            conv5 = layers.Conv2D(num_channels, kernel_size=5, padding='VALID', name=name+'_incep_5',kernel_initializer=initializer, activation='relu')(inputs_5)
            conv5 = layers.Dropout(rate=0.2)(conv5)
            max3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='VALID',name=name+'max_3')(inputs_3)
       
            # concatenate final outputs
            outputs = layers.Concatenate(axis=3)([conv3, conv5, max3])
            return outputs
        
        incep_A = inception(inputs_re,name='amp_inception',num_channels=10)
        pool_A,A_indices = tf.math.top_k(incep_A, k=1)
        pool_A = tf.math.reduce_mean(pool_A,axis=-1)
        pool_A = layers.Flatten()(pool_A)
        lnA = tf.math.reduce_sum(pool_A,axis=1)

        ## phase
        # phasor: a regular cnn layer for phi
        input_phi = self.PBCs(inputs_re,padding=6)#kernel_size=7
        convPhi = layers.Conv2D(filters=24, kernel_size=7, padding='VALID', name='convPhi_1',kernel_initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=1.0),activation=None)(input_phi)
        out1 = tf.math.exp(tf.complex(0.,convPhi))
        out2 = tf.math.reduce_sum(out1,axis=[-1,-2,-3])
        phi = tf.math.angle(out2)

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)