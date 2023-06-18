import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tool import time_it

class Model:
    def __init__(self, length, dimension, loadpath = None):
        """
        Base class to construct a convolutional neural network using keras
        
        Args:
        length: length of lattice
        dimension: dimension of lattice
        loadpath: path to load model

        class variables:
        batch_size
        num_points: shape of input data, or say number of particles
        net: keras model object
        """
        self.batch_size = 2000
        self.num_points = length ** dimension
        
        if loadpath == None:
            self.net = self.create_model(length,dimension)
            self.net.compile()
        else:
            self.net = tf.keras.models.load_model(loadpath)

    def __str__(self):
        pass

    def create_model(self,length,dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        raise NotImplementedError

    def PBCs(self,inputs,padding):
        """
        apply periodic boundary condition to the input lattice
        padding = int((kernel_size-1)/2)
        """
        inputs_pc = layers.Concatenate(axis=1)([inputs[:,-padding:,:,:],inputs,inputs[:,:padding,:,:]])
        inputs_pc = layers.Concatenate(axis=2)([inputs_pc[:,:,-padding:,:],inputs_pc,inputs_pc[:,:,:padding,:]])
        return inputs_pc

    def log_val(self, x):
        """
            Calculate log(\Psi(x)) 
            Args:
                x: the configuration needed to be calculated
        """
        # log_psi = self.net.predict(x, batch_size = self.batch_size, verbose = 0)
        log_psi = self.net(x)
        return tf.expand_dims(log_psi,axis=-1)

    def log_val_diff(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        log_val_xprime = self.log_val(xprime)
        log_val_x = self.log_val(x)
        return log_val_xprime-log_val_x

    def derlog(self, x):
        """
        Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$ where W can be the weights or the biases.
        """
        with tf.GradientTape(persistent=True) as g:
            log_psi = self.net(x)
            real_psi = tf.math.real(log_psi)
            imag_psi = tf.math.imag(log_psi)

        #Not using vectorized calculation (pfor = Flase) can prevent the excessiv memory consumption.
        real_jacobians = g.jacobian(real_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)
        imag_jacobians = g.jacobian(imag_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)

        return [tf.complex(real_jacobians[i], imag_jacobians[i]) for i in range(len(real_jacobians))]