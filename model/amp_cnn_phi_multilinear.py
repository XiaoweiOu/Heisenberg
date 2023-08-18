import numpy as np
import tensorflow as tf
import os
import psutil
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNPhiMultiLinear(Model):
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
        
        inputs_pbc = self.PBCs(inputs_re,padding=5)#kernel_size=6
        conv_A = layers.Conv2D(filters=64, kernel_size=6, padding='VALID', name='conv_amp', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='relu')(inputs_pbc)
        conv_A = tf.math.abs(conv_A)
        pool_A,A_indices = tf.math.top_k(conv_A, k=1)
        lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])

        ## phase
        # a simple linear function
        def msr_init(shape, dtype = None):
            indices = tf.constant([[i*self.length+j] for i in range(self.length) for j in range(self.length) if (i%2==0 and j%2==0) or (i%2==1 and j%2==1)])
            updates = tf.constant([np.pi]*(shape[0]//2))
            updates = tf.expand_dims(updates,axis=-1)
            msr_weights = tf.scatter_nd(indices, updates, shape)
            return msr_weights

        dense1_phi = layers.Dense(units=64, activation='relu', name = 'dense1_phi')(inputs)
        dense2_phi = layers.Dense(units=64, activation='relu', name = 'dense2_phi')(dense1_phi)
        phi = tf.reduce_sum(dense2_phi,axis=-1)

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)