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

        ### TEST ### fix log_psi to be the true values
        '''
        if (x.numpy()==np.array( [[1, 1], [1, 1], [0, 0], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 0.6518541365441228 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [1, 0], [0, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 6.767733036407472e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [1, 0], [0, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -1.1311674610826102e-16 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [1, 1], [0, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -3.0970033528162176e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [1, 1], [0, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 1.480707202293723e-16 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [1, 0], [0, 1], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 0.09719277761097418 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 1], [1, 0], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 4.368834750980412e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 0], [1, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -0.13567449484666533 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 0], [1, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -1.4083641444028257e-32 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 1], [1, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -5.134239975171319e-33 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 1], [1, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 0.09719277761097418 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 0], [1, 1], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -2.499261495161532e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 1], [0, 0], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -2.7754728645695526e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 0], [0, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -1.3124047078228165e-33 ,0.))
        elif (x.numpy()==np.array( [[1, 1], [0, 0], [0, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -1.1417299728387712e-17 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 1], [0, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -0.19438555522194872 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 1], [0, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 3.522740024861627e-33 ,0.))
        elif (x.numpy()==np.array( [[1, 0], [0, 0], [0, 1], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 1.4523414059035787e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 1], [1, 0], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -2.81004076602937e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 0], [1, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -1.7104578436645768e-33 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 0], [1, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -0.1943855552219487 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 1], [1, 1], [0, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -1.1417299728387712e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 1], [1, 0], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 3.5007792558727506e-33 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 0], [1, 1], [0, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 1.3853099321371073e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 1], [0, 0], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 1.057895769000228e-16 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 0], [0, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 0.09719277761097417 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [1, 0], [0, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 2.645714947425075e-33 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 1], [0, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 2.6202689248838836e-33 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 1], [0, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -0.6518541365441235 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [1, 0], [0, 1], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -4.415313286242046e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [0, 1], [1, 0], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 0.09719277761097417 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [0, 0], [1, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( -2.1640019305810062e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 1], [0, 0], [1, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 1.0395441726756677e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [0, 1], [1, 1], [1, 0]] )).all():
            log_psi = tf.math.log(tf.complex( 1.0395441726756677e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [0, 1], [1, 0], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( -3.582380509639454e-17 ,0.))
        elif (x.numpy()==np.array( [[0, 0], [0, 0], [1, 1], [1, 1]] )).all():
            log_psi = tf.math.log(tf.complex( 0.1356744948466655 ,0.))
        log_psi = tf.expand_dims(log_psi,axis=0)
        '''
        
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