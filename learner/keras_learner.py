import numpy as np
import tensorflow as tf
import time
import csv
import os
import gc
from gradient import get_gradient,get_gradient_sr,get_gradient_minsr,get_gradient_amp_penalty

# Print everything
# np.set_printoptions(threshold=np.inf)

class KerasLearner:
    def __init__(self, hamiltonian, model, sampler, optimizer,
                 num_epochs=1000, minibatch_size=0, window_period=50, reference_energy=None,
                 stopping_threshold=0.05, store_model_freq=1, observables=[], observable_freq = 0,
                 use_gradient='sr', initial_sample_path=None, is_save = True, run_name = '', use_adaptive_step_size = False):
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
        self.use_gradient = use_gradient
        self.initial_sample_path = initial_sample_path
        self.is_save = is_save
        self.run_name = run_name
        self.use_adaptive_step_size = use_adaptive_step_size

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

        ## whether calculating local energy for one sample at a time.
        self.onebyone = 'Hubbard' in self.model.__class__.__name__

    def learn(self):
        ## Reset array
        self.reset_memory_array()
        
        ## Get initial sample
        ## Move initial sample for thousand of times to get to the equilibrium (default = self.model.num_points*10)
        default_equilibration = self.model.num_points*10
        if self.initial_sample_path is None:
            self.samples = tf.convert_to_tensor(self.sampler.get_initial_random_samples(self.model.num_points),dtype=np.float32)
            print('===== Equilibrate initial sample start')
            self.samples = self.sampler.sample(self.model, self.samples, self.minibatch_size, num_steps=default_equilibration)
        else:
            self.samples = tf.convert_to_tensor(self.load_initial_sample(self.initial_sample_path),dtype=np.float32)

        print('===== Training '+ self.run_name + ' start')
        print('===== Reference energy:',self.reference_energy)
        for epoch in range(self.num_epochs):
            start = time.time()
            #####################################
            ####### TRAINING PROCESS ############
            #####################################

            ##### 1. Calculate local energy
            elocs = self.get_local_energy(self.samples,self.model,self.onebyone)
            energy_imag, energy, energy_std, rel_error = self.process_energy_and_error(elocs)

            ## Post-selection of samples
            # energy_imag, energy, energy_std, rel_error = self.post_selection(epoch, energy_imag, energy, energy_std, rel_error)

            ## Confirm the current sample and append
            # self.append_energy_and_error(energy,energy_std,rel_error)

            ## Print status
            print('### Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                epoch, energy, energy_std, energy_std / np.abs(energy), rel_error), end='')

            ## Check nan
            if np.isnan(energy):
                print('### ERROR ### energy is NAN now!',flush=True)
                exit(0)

            ##### 2. Calculate gradient
            derlogs = self.model.derlog(self.samples)

            if self.use_gradient == 'plain':
                grads, grad_norm = get_gradient(derlogs, self.samples.shape[0], elocs)
            elif self.use_gradient == 'sr':
                grads, grad_norm, inner_product = get_gradient_sr(derlogs, self.samples.shape[0], elocs, self.model.num_param_amp)
            elif self.use_gradient == 'amp_penalty':
                grads, grad_norm, inner_product = get_gradient_amp_penalty(derlogs, self.samples.shape[0], elocs, self.model.num_param_amp)
            elif self.use_gradient == 'minsr':
                grads, grad_norm, inner_product = get_gradient_minsr(derlogs, self.samples.shape[0], elocs, self.model.num_param_amp, mass = 1.)
            else:
                print("### ERROR ### gradient type incorrect!")
                exit(0)

            ##### 3. Save energy, current MC samples, and neural network
            ## save energy and samples
            if self.is_save:
                self.save_energy(epoch,energy,energy_std,grad_norm,self.optimizer.lr.numpy())
                self.save_samples(self.samples.numpy().tolist())

            ## save model
            if epoch % self.store_model_freq == 0 and self.is_save:
                self.model.net.save('./result/'+self.run_name)

            ##### 4. Apply gradients
            ## apply gradients
            self.optimizer.apply_gradients(zip([tf.cast(grad, tf.float32) for grad in grads], self.model.net.trainable_weights))

            ## adaptive step size
            if self.use_adaptive_step_size:
                old_lr = self.optimizer.lr
                new_lr = self.adaptive_step_size(old_lr,grads,inner_product)
                self.optimizer.lr.assign(new_lr)

            ##### 5. Get new sample
            # If only use SR to train log-amp and phi together, you need to increase the equilibration steps
            equilibration_step = default_equilibration
            if self.use_gradient == 'sr' and energy<-5:
                equilibration_step = self.model.num_points*50
            if self.use_gradient == 'sr' and energy<-10:
                equilibration_step = self.model.num_points*80
            if self.use_gradient == 'sr' and energy<-15:
                equilibration_step = self.model.num_points*100

            if self.model.__class__.__name__ == 'AmpCNNPhiMultiLinearTriangle' and energy<-10:
                equilibration_step = self.model.num_points*50

            ### TEST ### for minSR
            equilibration_step = self.model.num_points*20

            self.samples = self.sampler.sample(self.model, self.samples, self.minibatch_size, num_steps=equilibration_step)

            #####################################
            #####################################
            #####################################

            ## Calculating additional stuffs
            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)
            print(', time: %.5f' % time_interval)

        print ('===== Training finish')
        ## save the last data
        if self.is_save:
            self.model.net.save('./result/'+self.run_name)

    def adaptive_step_size(self,old_lr,search_direction,old_inner_product):
        ##### Calculate original energy gradient with updated network parameters
        sample_size = self.samples.shape[0]
        
        ## Calculate new local energy array and gradient with old samples
        eloc = self.get_local_energy(self.samples,self.model,self.onebyone)
        derlogs = self.model.derlog(self.samples)

        ## Calculate O_k
        all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)

        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        ## Calculate <D_{W}>
        derlog_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        ## Calculate <E_loc D^*_{W}>
        ed = tf.reduce_mean(tf.math.conj(all_derlogs) * eloc, axis = 0, keepdims = True)

        ## Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
        gradE = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)

        ## Flatten search_direction
        search_direction_fl = tf.concat([tf.reshape(search_direction_ele,[-1]) for search_direction_ele in search_direction],0)
        new_inner_product = tf.math.real(tf.einsum('i,i',tf.squeeze(gradE),search_direction_fl))

        ##### New learning rate selection
        modified_coeff = 5/4
        max_lr = 0.2
        test_lr = old_lr * old_inner_product / (old_inner_product - new_inner_product)

        ## Switch test_lr
        if 0 < test_lr <= old_lr:
            new_lr = test_lr
        else:
            new_lr = min(max_lr, modified_coeff * old_lr)

        return new_lr

    def get_local_energy(self, samples, model, onebyone):
        """
            Calculate local energy from a given samples
            $E_{loc}(x) = \sum_{x'} H_{x,x'} \Psi(x') / \Psi(x)$
            In this part, we instead do $log(\Psi(x')) - log(\Psi(x))$
            Args:
                onebyone: whether calculating local energy for one sample at a time.
                samples: samples that we want to calculate the local energy
            Return:
                The local energy of each given samples (complex value).
        """
        if onebyone == True:
            eloc_array = []
            for state in samples:
                eloc = self.hamiltonian.local_energy(state,model)
                eloc_array.append(eloc)
            eloc_array = np.array(eloc_array)

        else:
            ## Calculate $H_{x,x'}$
            hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(samples, len(samples))

            ## Calculate $log(\Psi(x')) - log(\Psi(x))$
            lvd = self.hamiltonian.calculate_ratio(samples, self.model, len(samples))

            ## Sum over x'
            eloc_array = tf.reduce_sum((tf.exp(lvd) * tf.cast(hamiltonian, tf.complex64)), axis=1, keepdims=True)

        return eloc_array
        
    def post_selection(self, epoch, energy_imag, energy, energy_std, rel_error):
        """
        prevent the runaway or blow-up problem:
        ## |Re[E(epoch)-E(epoch-1)]| < alpha1
        ## |Im[E(epoch)]|< alpha2 * energy_std[epoch]
        ## energy_std[epoch] < alpha3 * energy_std[epoch-1]
        """
        alpha1, alpha2, alpha3 = 3,5,6
        while epoch >= 1 and (np.abs(energy - self.ground_energy[-1]) > alpha1 or \
        np.abs(energy_imag) > alpha2 * energy_std or \
        energy_std > alpha3 * self.ground_energy_std[-1]):
            self.samples = self.sampler.sample(self.model, self.samples, self.minibatch_size, num_steps=self.model.num_points)
            elocs = self.get_local_energy(self.samples,self.model,self.onebyone)
            energy_imag, energy, energy_std, rel_error = self.process_energy_and_error(elocs)

        return energy_imag, energy, energy_std, rel_error

    def save_energy(self, epoch, energy, energy_std, grad_norm, learning_rate):
        """
        Save all energy values and the standard deviation of the current local energy array.
        """
        result_file = './result/'+self.run_name+'_energy.csv'
        if os.path.isfile(result_file) == False or (self.initial_sample_path == None and epoch == 0):
            csvfile = open(result_file,'w')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['epoch','energy','energy_std','grad_amp_before','grad_phi_before','grad_amp_after','grad_phi_after','learning_rate'])
            csvwriter.writerow([epoch,energy,energy_std]+grad_norm+[learning_rate])
            csvfile.close()
        else:
            with open(result_file,'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([epoch,energy,energy_std]+grad_norm+[learning_rate])

    '''def save_samples(self, samples):
        """
        Save the latest Monte Carlo sample / Markov chain.
        """
        result_file = './result/'+self.run_name+'_samples.csv'
        with open(result_file,'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(samples)

    def load_initial_sample(self, initial_sample_path):
        with open(initial_sample_path,'r') as csvfile:
            csvreader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
            initial_sample = [line for line in csvreader]
        return initial_sample'''

    def save_samples(self, samples):
        """
        Save the latest Monte Carlo sample / Markov chain in binary format.
        """
        result_file = './result/'+self.run_name+'_samples.npy'
        np.save(result_file, samples)

    def load_initial_sample(self, initial_sample_path):
        initial_sample = np.load(initial_sample_path)
        return initial_sample

    def process_energy_and_error(self, elocs):
        """
            Process the energy and error by calculating the energy, energy over windows and relative error.
            Args:
                elocs: the local energies array of one epoch
        """
        ## Calculate the imaginary part of ground state energy
        ground_energy_imag = np.imag(np.mean(elocs))

        ## Calculate ground state energy mean and std
        ground_energy = np.real(np.mean(elocs))
        ground_energy_std = np.real(np.std(elocs))

        ### Calculate relative error
        if self.reference_energy is None:
            rel_error = 0.0
        else:
            rel_error = np.abs((ground_energy - self.reference_energy) / self.reference_energy)

        return ground_energy_imag, ground_energy, ground_energy_std, rel_error

    def append_energy_and_error(self,ground_energy,ground_energy_std,rel_error):
        self.ground_energy.append(ground_energy)
        self.ground_energy_std.append(ground_energy_std)
        self.rel_errors.append(rel_error)

        ### Calculate energy over windows
        energy_window = np.mean(self.ground_energy[-self.window_period:])
        energy_window_std = np.std(self.ground_energy[-self.window_period:])
        self.energy_windows.append(energy_window)
        self.energy_windows_std.append(energy_window_std)

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
