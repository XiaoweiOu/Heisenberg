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

    # def PBCs(self,inputs,padding):
    #     """
    #     apply periodic boundary condition to the input lattice
    #     padding = int((kernel_size-1)/2)
    #     """
    #     inputs_pc = layers.Concatenate(axis=1)([inputs[:,-padding:,:,:],inputs,inputs[:,:padding,:,:]])
    #     inputs_pc = layers.Concatenate(axis=2)([inputs_pc[:,:,-padding:,:],inputs_pc,inputs_pc[:,:,:padding,:]])
    #     return inputs_pc

    def PBCs(self,inputs,padding):
        """
        apply periodic boundary condition to one side of the input lattice
        padding = kernel_size-1
        """
        inputs_pc = layers.Concatenate(axis=1)([inputs,inputs[:,:padding,:,:]])
        inputs_pc = layers.Concatenate(axis=2)([inputs_pc,inputs_pc[:,:,:padding,:]])
        return inputs_pc

    def log_val(self, x):
        """
            Calculate log(\Psi(x)) 
            Args:
                x: the configuration needed to be calculated
        """
        log_psi = self.net(x)
        return tf.expand_dims(log_psi,axis=-1)
    
    def log_val_one_sample(self, x):
        """
            Calculate log(\Psi(x)) 
            Args:
                x: one configuration needed to be calculated
        """
        x_expand = tf.expand_dims(x,axis=0)
        log_psi = self.net(x_expand)
        return log_psi

    def log_val_symmetrized(self, x):
        """
            Calculate log(\Psi(x)) with c4v symmetry
            log(\Psi(x)) = log(1/|S| \sum_{x' \in S} exp(net(x')))
            |S|=8
        """
        order_c4v = 8

        ## identity operation
        sum = tf.math.exp(self.net(x))
        
        ## other operations
        for i in range(order_c4v-1):
            x_new = transform(x,i)
            sum += tf.math.exp(self.net(x_new))

        ## output
        log_psi = tf.math.log(1/order_c4v * sum)
        return tf.expand_dim(log_psi,axis=-1)

    def transform(self, x, idx):
        """
            Transform a few configurations under c4v group
            Args:
                idx: index of the group operation (1-7 for c4v group)
        """
        
    
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

    def log_val_diff_one_sample(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        log_val_xprime = self.log_val_one_sample(xprime)
        log_val_x = self.log_val_one_sample(x)
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
