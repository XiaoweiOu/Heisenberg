import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tool import time_it
import sys

class Model:
    def __init__(self, length, dimension, loadpath = None, symmetry = None):
        """
        Base class to construct a convolutional neural network using keras
        
        Args:
        length: length of lattice
        dimension: dimension of lattice
        loadpath: path to load model

        class variables:
        num_points: shape of input data, or say number of particles
        net: keras model object
        """
        self.batch_size = 2000
        self.num_points = length ** dimension
        self.symmetry = symmetry
        self.length = length
        self.dimension = dimension
        self.num_param_amp = 2368 # It should be changed to the specific network architecture.

        if loadpath == None:
            self.net = self.create_model(length,dimension)
            self.net.compile()
        else:
            self.net = tf.keras.models.load_model(loadpath)

        self.net.summary()

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
        apply periodic boundary condition to one side of the input lattice
        padding = kernel_size-1
        """
        inputs_pc = layers.Concatenate(axis=1)([inputs,inputs[:,:padding,:,:]])
        inputs_pc = layers.Concatenate(axis=2)([inputs_pc,inputs_pc[:,:,:padding,:]])
        return inputs_pc

    def log_val(self, x):
        """
            Calculate log(\Psi(x))

            With c4v symmetry: |S|=8
            log(\Psi(x)) = log(1/|S| \sum_{x' \in S} exp(net(x')))

            Args:
                x: the configuration needed to be calculated
        """
        if self.symmetry is None:
            log_psi = self.net(x)
            
        else:
            x_symmetrized = self.symmetry.symmetrize_config(x)
            log_psi_symmetrized = self.net(x_symmetrized)

            num_samples = x.shape[0] # Number of original configurations
            batch_indices_list = [[i+j*num_samples for j in range(self.symmetry.order)] for i in range(num_samples)]
            batch_indices_tensor = tf.constant(batch_indices_list, dtype=tf.int32)
            selected_elements = tf.gather(log_psi_symmetrized, batch_indices_tensor, axis=0)

            ## v1: average the logarithmic wave function
            log_psi = tf.reduce_mean(selected_elements, axis=1)

            ## v1.5: for amplitude, consider the average; for phase, keep the original value
            # log_psi = tf.complex(tf.math.real(tf.reduce_mean(selected_elements, axis=1)),tf.math.imag(log_psi_symmetrized[:num_samples]))

            ## v2: average the wave function
            # selected_elements = selected_elements - 15 #tf.cast(tf.math.reduce_mean(tf.math.real(selected_elements)),tf.complex64)
            # selected_elements = tf.math.exp(selected_elements)
            # log_psi = tf.reduce_mean(selected_elements, axis=1)
            # log_psi = tf.math.log(log_psi)

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
    
    # def derlog_old(self, x):
    #     """
    #     Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$ where W can be the weights or the biases.
    #     There are some numerical differences from the new method below.
    #     """
    #     with tf.GradientTape(persistent=True) as g:
    #         log_psi = self.net(x)
    #         real_psi = tf.math.real(log_psi)
    #         imag_psi = tf.math.imag(log_psi)

    #     #Not using vectorized calculation (pfor = Flase) can prevent the excessiv memory consumption.
    #     real_jacobians = g.jacobian(real_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)
    #     imag_jacobians = g.jacobian(imag_psi, self.net.trainable_variables, parallel_iterations = 10000, experimental_use_pfor = False)

    #     return [tf.complex(real_jacobians[i], imag_jacobians[i]) for i in range(len(real_jacobians))]
    
    def derlog(self, x, batch_size=100):
        """
        Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$ where W can be the weights or the biases.
        batch_size = 500 (symmetrized version: 100)
        """
        for i in range(int(x.shape[0]/batch_size)):
            x_sliced = x[i*batch_size:(i+1)*batch_size,:]
        
            with tf.GradientTape(persistent=True) as g:
                log_psi = self.log_val(x_sliced)
                log_psi = tf.squeeze(log_psi,-1)

                real_psi = tf.math.real(log_psi)
                imag_psi = tf.math.imag(log_psi)

            if i==0:
                real_jacobians = g.jacobian(real_psi, self.net.trainable_variables)
                imag_jacobians = g.jacobian(imag_psi, self.net.trainable_variables)
            else:
                real_jac = g.jacobian(real_psi, self.net.trainable_variables)
                real_jacobians = [tf.concat([real_jacobians[i],real_jac[i]],0) for i in range(len(real_jacobians))]
                imag_jac = g.jacobian(imag_psi, self.net.trainable_variables)
                imag_jacobians = [tf.concat([imag_jacobians[i],imag_jac[i]],0)for i in range(len(imag_jacobians))]

        return [tf.complex(real_jacobians[i], imag_jacobians[i]) for i in range(len(real_jacobians))]