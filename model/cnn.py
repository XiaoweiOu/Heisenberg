import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNN(object):
    def __init__(self, length, dimension, loadpath = None):
        """
        Construct a convolutional neural network
        
        Args:
        length: length of lattice
        dimension: dimension of lattice
        loadpath: path to load model

        class variables:
        batch_size
        num_points: shape of input data, or say number of particles
        model_name
        net: keras model object
        """
        self.batch_size = 50
        self.num_points = length ** dimension 
        self.model_name = "CNN_MSR" #amp: cnn network;phi: marshall sign rule
        
        if loadpath == None:
            self.net = self.create_model(length,dimension)
            self.net.compile()
        else:
            self.net = tf.keras.models.load_model(loadpath)

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

        ## expand a dimension for channel at the end
        inputs_re = tf.expand_dims(inputs_re,-1)
        
        ## amplitude
        incep_A = self.inception(inputs_re,name='amp_inception',num_channels=10)
        pool_A,A_indices = tf.math.top_k(incep_A, k=1)
        pool_A = tf.math.reduce_mean(pool_A,axis=-1)
        pool_A = layers.Flatten()(pool_A)
        lnA = tf.math.reduce_sum(pool_A,axis=1)

        ## phase
        #Marshall sign rule
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

        # ### TEST ### lnA=0
        # batch_size=tf.shape(inputs)[0]
        # outputs = tf.complex(tf.tile([0.],[batch_size,]),phi)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def inception(self,inputs,name,num_channels=2048):
        """
        Inception layer: consist of a 3x3 conv layer, a 5x5 conv layer, and a 3x3 maxpooling layer
        """

        initializer = tf.keras.initializers.HeNormal()
        inputs_3 = self.PBCs(inputs,padding=int((3-1)/2))#kernel_size=3
        inputs_5 = self.PBCs(inputs,padding=int((5-1)/2))#kernel_size=5
        
        # dropout rate is suggested to between 0.2 and 0.5
        conv3 = layers.Conv2D(num_channels, kernel_size=3, padding='VALID', name=name+'_incep_3',kernel_initializer=initializer, activation='relu')(inputs_3)
        conv3 = layers.Dropout(rate=0.2)(conv3)
        conv5 = layers.Conv2D(num_channels, kernel_size=5, padding='VALID', name=name+'_incep_5',kernel_initializer=initializer, activation='relu')(inputs_5)
        conv5 = layers.Dropout(rate=0.2)(conv5)
        max3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='VALID',name=name+'max_3')(inputs_3)
       
        # concatenate final outputs
        outputs = layers.Concatenate(axis=3)([conv3, conv5, max3])
        return outputs

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
        log_psi = self.net.predict(x, batch_size = self.batch_size, verbose = 0)
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
        
        jacobians = g.jacobian(log_psi, self.net.trainable_variables)
        return [tf.cast(jacobian, tf.complex64) for jacobian in jacobians]