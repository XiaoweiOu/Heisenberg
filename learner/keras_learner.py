import numpy as np
import tensorflow as tf
import time
import csv
import os
from tensorflow import keras
from tensorflow.keras import layers

class KerasLearner(object):
    def __init__(self, hamiltonian, model, sampler, optimizer, num_epochs=1000,
                 minibatch_size=0, window_period=50, reference_energy=None, stopping_threshold=0.05,
                 store_model_freq=5, observables=[], observable_freq = 0, use_sr=True, transfer_sample = None, save_model = False):
        """
        Construct a learner objects
        Args:
            hamiltonian: Hamiltonian of the model
            model: the machine learning model used
            sampler: the sampler used to train
            optimizer: the optimizer for training
            num_epochs: the number of epochs for training (Default: 1000)
            minibatch_size: the number of minibatch training (Default: 0)
            window_period: the number of windows for logging purposes (Default: 50) 
            reference_energy: reference energy value if there is any (Default: None)
            stopping_threshold: stopping threshold for the training defined as mean(elocs)/std(elocs) (Default: 0.05)    
            store_model_freq: store the model only at epochs that is the multiplier of this value. By default the model at the first and last epochs are saved. (Default: 5)
            observables: observables value to compute (Default: [])
            observable_freq: compute the observables only at epochs that is the multiplier of this value. Zero means nothing is stored. By default observables are calculated at the last epoch. (Default: 0)
            
        """
        self.hamiltonian = hamiltonian
        self.model = model
        self.sampler = sampler
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.window_period = window_period
        self.reference_energy = reference_energy
        self.stopping_threshold = stopping_threshold
        self.store_model_freq = store_model_freq
        self.observables = observables
        self.observable_freq = observable_freq
        self.use_sr = use_sr
        self.transfer_sample = transfer_sample
        self.save_model = save_model

        self.ground_energy = []
        self.ground_energy_std = []
        self.energy_windows = []
        self.energy_windows_std = []
        self.rel_errors = []
        self.times = []
        self.observables_value = []
        self.model_params = []
        self.samples = []

        if self.minibatch_size == 0 or self.minibatch_size > self.sampler.num_samples:
            self.minibatch_size = self.sampler.num_samples

    def learn(self):
        ## Reset array
        self.reset_memory_array()
        
        ## Get initial sample
        self.samples = tf.convert_to_tensor(self.sampler.get_initial_random_samples(self.model.num_points))
        
        print ('===== Training start')
        print('===== Reference energy:',self.reference_energy)  
        for epoch in range(self.num_epochs):
            start = time.time()
            #####################################
            ####### TRAINING PROCESS ############
            #####################################

            ##### 1. Calculate local energy 
            elocs = self.get_local_energy(self.samples)    
            energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(elocs)

            ##### Some processing
            ## Print status
            print('### Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                epoch, energy, energy_std, energy_std / np.abs(energy), rel_error), end='')

            ## save energy
            result_file = 'result.csv'
            if not os.path.exists(result_file):
                csvfile = open(result_file,'w')
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['epoch','energy'])
                csvwriter.writerow([epoch,energy])
                csvfile.close()
            else:
                with open(result_file,'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([epoch,energy])

            ## save weight
            if epoch % self.store_model_freq == 0 and self.save_model == True:
                self.model.model.save('./result/'+self.model.model_name)

            ##### 2. Calculate gradient
            if self.use_sr:
                grads = self.get_gradient_sr(self.samples, self.minibatch_size, elocs, epoch)
            else:
                grads = self.get_gradient(self.samples, self.minibatch_size, elocs)
        
            ##### 3. Apply gradients
            self.optimizer.apply_gradients(zip([tf.cast(grad, tf.float32) for grad in grads], self.model.model.trainable_weights))

            ##### 4. Get new sample
            self.samples = self.sampler.sample(self.model, self.samples, self.minibatch_size)

            #####################################
            #####################################
            #####################################

            ### Calculating additional stuffs
            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)

            print(', time: %.5f' % time_interval)

        print ('===== Training finish')        
        ## save the last data
        if self.save_model == True:
            self.model.model.save('./result'+self.model.model_name)

    def get_local_energy(self, samples):
        """
            Calculate local energy from a given samples
            $E_{loc}(x) = \sum_{x'} H_{x,x'} \Psi(x') / \Psi(x)$
            In this part, we instead do $log(\Psi(x')) - log(\Psi(x))$
            Args:
                samples: samples that we want to calculate the local energy
            Return:
                The local energy of each given samples
        """
        ## Calculate $H_{x,x'}$
        hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(samples, len(samples))
        
        ## Calculate $log(\Psi(x')) - log(\Psi(x))$
        lvd = self.hamiltonian.calculate_ratio(samples, self.model, len(samples))

        ## Sum over x'
        eloc_array = tf.reduce_sum((tf.exp(lvd) * tf.cast(hamiltonian, tf.complex64)), axis=1, keepdims=True)

        return eloc_array
        
    def get_gradient(self):
        """
            Calculate the gradient of E[\Psi] defined as 
            $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
            where D_W is the gradient of the neural network w.r.t to its output defined as
            $D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
            Args:
                samples: the samples to calculate gradient
                sample_size:  the sample size
                eloc: the local energy E_{loc}
        """
        ## Get D_{W} from the model
        derlogs = self.model.derlog(samples) 

        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        grads = []
        for ii, derlog in enumerate(derlogs):
            old_shape = derlog.shape
            derlog = tf.reshape(derlog, (sample_size, -1))

            ## Calculate <D_{W}>
            derlog_mean = tf.reduce_mean(derlog, axis=0, keepdims=True)

            #### Calculate <E_loc D_{W}>
            ed = tf.reduce_mean(tf.math.conj(derlog) * eloc, axis = 0, keepdims = True)

            #### Calculate  $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
            grad = (ed - derlog_mean * eloc_mean)
        
            grads.append(tf.reshape(grad, old_shape[1:]))
       
        return grads

    def get_gradient_sr(self, samples, sample_size, eloc, epoch):
        """
            Calculate the gradient of E[\Psi] using the stochastic reconfiguration
            Args:
                samples: the samples to calculate gradient
                sample_size:  the sample size
                eloc: the local energy E_{loc}
        """
        ## Get D_{W} from the model
        derlogs = self.model.derlog(samples)
        old_shapes = [derlog.shape for derlog in derlogs]

        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        ## Calculate O_k
        all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)

        ## Calculate <O_k>
        all_derlogs_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        ## Calculate <O^*_k O_k>
        all_derlogs_derlogs_mean = tf.einsum('ij, ik->jk', tf.math.conj(all_derlogs), all_derlogs)/ len(samples)

        ## Calculate S_kk = <O^*_k O_k> - <O_k><O^*_k>
        S_kk = all_derlogs_derlogs_mean - tf.math.conj(all_derlogs_mean) * tf.transpose(all_derlogs_mean)

        ## Regularize S_kk to make sure it is invertible
        regularizer = 0.2

        S_kk_diag_reg = tf.linalg.tensor_diag(regularizer * tf.linalg.diag_part(S_kk))
        S_kk_reg = S_kk + S_kk_diag_reg

        ## Calculate <D_{W}>
        derlog_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        #### Calculate <E_loc D_{W}>
        ed = tf.reduce_mean(tf.math.conj(all_derlogs) * eloc, axis = 0, keepdims = True)

        #### Calculate  $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
        #grad = 2 * tf.math.real(ed - derlog_mean * eloc_mean)
        grad = ed - derlog_mean * eloc_mean

        ### inv(S_kk) * grad == final_grads or S_kk * final_grads == grad
        final_grads = tf.linalg.solve(S_kk_reg, tf.transpose(grad))

        grads = []
        prev = 0
        for old_shape in old_shapes:
            final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]
            
            prev += tf.reduce_prod(old_shape[1:])
 
            grads.append(tf.reshape(final_grad, old_shape[1:]))

        return grads

    def process_energy_and_error(self, elocs):
        """
            Process the energy and error by calculating the energy, energy over windows and relative error.
            Args:
                elocs: the local energies array of one epoch
        """
        ## Calculate ground state energy mean and std
        ground_energy = np.real(np.mean(elocs))
        ground_energy_std = np.real(np.std(elocs))

        self.ground_energy.append(ground_energy)
        self.ground_energy_std.append(ground_energy_std)

        ### Calculate energy over windows
        energy_window = np.mean(self.ground_energy[-self.window_period:])
        energy_window_std = np.std(self.ground_energy[-self.window_period:])

        self.energy_windows.append(energy_window)
        self.energy_windows_std.append(energy_window_std)

        ### Calculate relative error
        if self.reference_energy is None:
            rel_error = 0.0
        else:
            rel_error = np.abs((ground_energy - self.reference_energy) / self.reference_energy)
        self.rel_errors.append(rel_error)

        return ground_energy, ground_energy_std, energy_window, energy_window_std, rel_error

    def reset_memory_array(self):
        """
        Reset memory array to all empty
        """
        self.ground_energy = []
        self.ground_energy_std = []
        self.energy_windows = []
        self.energy_windows_std = []
        self.rel_errors = []
        self.times = []
        self.samples = []
        self.model_params = []
        self.observables_value = []