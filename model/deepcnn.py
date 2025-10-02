import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers
from model import Model

class DeepCNN(Model):
    def residual_block(self, x, num_channels, kernel_size):
        residual = x

        # First convolutional layer
        x = self.PBCs(x,padding=kernel_size-1)
        x = layers.Conv2D(num_channels, kernel_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second convolutional layer
        x = self.PBCs(x,padding=kernel_size-1)
        x = layers.Conv2D(filters=1, kernel_size=kernel_size)(x)
        x = layers.BatchNormalization()(x)

        # Add shortcut connection
        x = layers.Add()([residual, x])
        x = layers.ReLU()(x)

        return x

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## In this case, we don't separate amplitude and phase,
        ## so this number is meaningless.
        self.num_param_amp = 1

        ## reshape input to be 2D data
        inputs = tf.keras.Input(shape=(length ** dimension,),dtype=tf.float64)
        inputs_re = layers.Reshape((length,length),input_shape=(length ** dimension,))(inputs)

        ## expand a dimension for feature at the end
        inputs_re = tf.expand_dims(inputs_re,-1)
        
        ## Use a single path for both amplitude and phase
        x1 = self.residual_block(inputs_re, num_channels = 64, kernel_size = 6)
        x2 = self.residual_block(x1, num_channels = 64, kernel_size = 6)

        ## Map to real and imag parts in the final layer
        x_reshape = layers.Reshape((length ** dimension,),input_shape=(length,length))(x2)
        real = layers.Dense(units=1)(x_reshape)
        imag = layers.Dense(units=1)(x_reshape)
        
        ## output
        # stack the values for output
        outputs = tf.complex(real,imag)
        outputs = tf.squeeze(outputs, axis=-1)

        return tf.keras.Model(inputs=inputs, outputs=outputs)