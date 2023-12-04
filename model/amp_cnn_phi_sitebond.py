import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import Model
import time

class AmpCNNPhiSiteBond(Model):
    """
    Amplitude network: one CNN layer
    Phase network: use the site and bond info as inputs

    site = 1 (spin up) or -1 (spin down)
    bond = 1 (two spins connected by this bond point in the same direction)
       or -1 (two spins connected by this bond point in the different direction)
    """
    def site2bond(self, site_samples):
        """
        Transfer a spin configuration from "site" to "bond".
        Both of their shapes = (num_samples, num_particles)
        """
        site_samples_2d = tf.reshape(site_samples,[tf.shape(site_samples)[0],self.length,self.length])
        site_samples_right = tf.concat([site_samples_2d[:,:,-1:],site_samples_2d[:,:,:-1]],axis=-1)
        site_samples_right = site_samples_right * site_samples_2d
        site_samples_down = tf.concat([site_samples_2d[:,-1:,:],site_samples_2d[:,:-1,:]],axis=-2)
        site_samples_down = site_samples_down * site_samples_2d

        site_samples_right_down = tf.concat([site_samples_right,site_samples_down],axis=1)
        indices = np.array([[i,i+self.length] for i in range(self.length)])
        indices = indices.flatten()
        selected_elements = tf.gather(site_samples_right_down,indices,axis=1)

        bond_samples = tf.reshape(selected_elements,[tf.shape(site_samples)[0],self.length*self.length*2])
        return bond_samples

    def log_val(self, x):
        """
            Calculate log(\Psi(x))

            With c4v symmetry: |S|=8
            log(\Psi(x)) = log(1/|S| \sum_{x' \in S} exp(net(x')))

            Args:
                x: the configuration needed to be calculated
        """
        if self.symmetry is None:
            log_psi = self.net([x, self.site2bond(x)])
            
        else:
            time_start = time.time()

            x_symmetrized = self.symmetry.symmetrize_config(x)

            time_symmetrized = time.time()
            # print('### TEST ### time to symmetrize',time_symmetrized - time_start)

            log_psi_symmetrized = self.net([x_symmetrized, self.site2bond(x_symmetrized)])

            time_log_psi_symmetrized = time.time()
            # print('### TEST ### time to calculate log_psi_symmetrized',time_log_psi_symmetrized - time_symmetrized)

            num_samples = x.shape[0] # Number of original configurations
            batch_indices_list = [[i+j*num_samples for j in range(self.symmetry.order)] for i in range(num_samples)]
            batch_indices_tensor = tf.constant(batch_indices_list, dtype=tf.int32)

            time_build_indices = time.time()
            # print('### TEST ### time to build indices',time_build_indices - time_log_psi_symmetrized)

            selected_elements = tf.gather(log_psi_symmetrized, batch_indices_tensor, axis=0)

            time_selected_elements = time.time()
            # print('### TEST ### time to gather selected elements',time_selected_elements - time_build_indices)

            ## v1: average the logarithmic wave function
            # log_psi = tf.reduce_mean(selected_elements, axis=1)

            ## v1.5: for amplitude, consider the average; for phase, keep the original value
            # log_psi = tf.complex(tf.math.real(tf.reduce_mean(selected_elements, axis=1)),tf.math.imag(log_psi_symmetrized[:num_samples]))

            ## v2: average the wave function
            selected_elements = selected_elements - 55 #tf.cast(tf.math.reduce_mean(tf.math.real(selected_elements)),tf.complex64)
            selected_elements = tf.math.exp(selected_elements)
            log_psi = tf.reduce_mean(selected_elements, axis=1)
            log_psi = tf.math.log(log_psi)

            time_end = time.time()
            # print('### TEST ### time to do final calculation for log_psi',time_end - time_selected_elements)

        return tf.expand_dims(log_psi,axis=-1)

    def create_model(self, length, dimension):
        """
        Create a cnn for the wave funtion.
        Return: log(psi)=lnA+i*phi
        """
        ## reshape input to be 2D data
        site_inputs = tf.keras.Input(shape=(length ** dimension,),dtype=tf.float64,name='site')
        bond_inputs = tf.keras.Input(shape=((length ** dimension)*2,),dtype=tf.float64,name='bond')
        site_inputs_re = layers.Reshape((length,length),input_shape=(length ** dimension,))(site_inputs)
        bond_inputs_re = layers.Reshape((2*length,length),input_shape=((length ** dimension)*2,))(bond_inputs)
        
        ## expand a dimension for feature at the end
        site_inputs_re = tf.expand_dims(site_inputs_re,-1)
        bond_inputs_re = tf.expand_dims(bond_inputs_re,-1)
        
        ## amplitude
        site_inputs_pbc = self.PBCs(site_inputs_re,padding=5)#kernel_size=6
        conv_A = layers.Conv2D(filters=64, kernel_size=6, padding='VALID', name='conv_amp', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='relu')(site_inputs_pbc)
        self.num_param_amp = (6*6+1)*64
        conv_A = tf.math.abs(conv_A)
        pool_A,A_indices = tf.math.top_k(conv_A, k=1)
        lnA = tf.math.reduce_sum(pool_A,axis=[-1,-2,-3])

        ## phase: let the network only represent the deviation from MSR
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

        phi_msr = layers.Lambda(function=marshall)(site_inputs_re)

        # Deviation from MSR: use bond_inputs
        # dense1_phi = layers.Dense(units=64, activation='relu', name = 'dense1_phi')(bond_inputs)
        # dense2_phi = layers.Dense(units=64, activation='relu', name = 'dense2_phi')(dense1_phi)
        # phi = tf.reduce_sum(dense2_phi,axis=-1)

        bond_inputs_pbc = self.PBCs(bond_inputs_re,padding=5)#kernel_size=6
        conv_phi = layers.Conv2D(filters=64, kernel_size=6, padding='VALID', name='conv_phi', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='relu')(bond_inputs_pbc)
        pool_phi,phi_indices = tf.math.top_k(conv_phi, k=1)
        phi = tf.math.reduce_sum(pool_phi,axis=[-1,-2,-3])

        # Add msr and deviation together
        phi = phi+phi_msr

        ## output
        # stack the values for output
        outputs = tf.complex(lnA,phi)

        return tf.keras.Model(inputs=[site_inputs,bond_inputs], outputs=outputs)