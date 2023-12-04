import numpy as np
import tensorflow as tf
import os
import psutil
from tensorflow import keras
from tensorflow.keras import layers
from model import Model

class AmpCNNPhiMSR(Model):
    """
    Amplitude network: one inception layer
    Phase network: Marshall sign rule
    """
    def __str__(self):
        return "Amplitude network: CNN Phase: sign rule"

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

            self.num_param_amp = (3*3+1)*num_channels + (5*5+1)*num_channels

            # concatenate final outputs
            outputs = layers.Concatenate(axis=3)([conv3, conv5, max3])
            return outputs

        incep_A = inception(inputs_re,name='amp_inception',num_channels=10)
        pool_A,A_indices = tf.math.top_k(incep_A, k=1)
        pool_A = tf.math.reduce_mean(pool_A,axis=-1)
        pool_A = layers.Flatten()(pool_A)
        lnA = tf.math.reduce_sum(pool_A,axis=1)

        ## phase
        # Marshall sign rule
        def marshall(x):
            batch_size=tf.shape(x)[0]
            indices=[]
            for i in range(length):
                for j in range(length):
                    if i%2==0 and j%2==0:
                        indices.append([i,j])
                    if i%2==1 and j%2==1:
                        indices.append([i,j])
            indices=tf.expand_dims(indices,axis=0)
            indices=tf.tile(indices,[batch_size,1,1])
            gather=tf.gather_nd(params=x[:,:,:,0],indices=indices,batch_dims=1)
            gather = (-gather+1)/2
            return tf.math.reduce_sum(gather,axis=1)*np.pi
        
        phi = layers.Lambda(function=marshall)(inputs_re)

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)