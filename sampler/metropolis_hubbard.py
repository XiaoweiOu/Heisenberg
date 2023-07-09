from sampler import Sampler
import tensorflow as tf
import numpy as np
import gc
import copy

class MetropolisHubbard():
    """
    This class is used to do a metropolis sampling for Hubbard model.
    """

    def __init__(self, num_samples, num_sites, total_sz=0):
        self.total_sz = total_sz
        self.num_samples = num_samples
        self.num_sites = num_sites

    def get_initial_random_samples(self, num_sites, num_samples=None):
        """
        Get initial random samples of shape (num_samples, num_sites, 2).
        Each entry is a occupation number, either 1 or 0.
        [:,:,0] is spin down, [:,:,1] is spin up.
        Assume total_sz==0

        Args:
            sample_size: the number of particles
            num_sites: number of samples
        Return:
            initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        assert self.total_sz == 0
        assert num_sites % 2 == 0

        one = np.ones(int(num_sites/2))
        zero = np.zeros(int(num_sites/2))
        base = np.concatenate((one,zero))

        ## shuffle them.
        # Both columns of spin down and spin up are a rearrangement of "base".
        init_data = []
        for i in range(num_samples):
            column1 = np.copy(base)
            column2 = np.copy(base)
            np.random.shuffle(column1)
            np.random.shuffle(column2)
            column1 = np.expand_dims(column1, axis=-1)
            column2 = np.expand_dims(column2, axis=-1)

            data = np.concatenate((column1,column2),axis=1)
            init_data.append(data)

        init_data = np.array(init_data, np.float32)
        return init_data

    def sample_once(self, model, starting_sample, starting_wfs, num_samples):
        """
            Do a one metropolis step from a given starting samples.
            Generate [num_samples * 1] new samples. 
            Model is used to calculate probability.
            Args:
                model: the model to calculate probability |\Psi(x)|^2
                starting_samples: the initial samples
                starting_wfs: wave functions array of the current sample
                num_samples: number of samples returned
            Return:
                new samples from one metropolis exchange
                new wave functions array
        """

        ## Get new configuration by flipping two random spins
        new_config = self.get_new_config(starting_sample)
        new_wfs = model.log_val(new_config)

        ## Calculate the ratio of the new configuration and old configuration probability by computing |psi(x') / psi(x)|^2
        # ratio = tf.abs(tf.exp(model.log_val_diff(new_config, starting_sample))) ** 2
        ratio = tf.abs(tf.exp(new_wfs - starting_wfs)) ** 2

        ## Random number
        random = tf.random.uniform((num_samples, 1), 0, 1)

        ## Calculate acceptance
        accept = tf.squeeze(tf.greater(ratio, random))
        accept = tf.broadcast_to(tf.reshape(accept, (accept.shape[0], 1, 1)), starting_sample.shape)

        ## Reject and accept samples
        sample = tf.where(accept, new_config, starting_sample)
        wfs = tf.where(tf.expand_dims(accept[:,0,0],axis=-1), new_wfs, starting_wfs)

        return sample, wfs

    def get_new_config(self, samples):
        """
        Move a spin-up electron with probability=1/2, spin-down electron with p=1/2.
        Pick any spin-up (down) electron, and then move into any site without spin-up electron.
        
        Args:
            samples: shape = (num_samples, num_sites,2)
        """
        new_samples = []

        for sample in samples:
            random = np.random.uniform(0.0,1.0)
            new_sample = sample.numpy()
            if random < 0.5: # hop a spin down electron
                occup_list = [i for i in range(self.num_sites) if sample[i,0]==1]
                empty_list = [i for i in range(self.num_sites) if sample[i,0]==0]
                occup_site = np.random.choice(occup_list)
                empty_site = np.random.choice(empty_list)
                new_sample[occup_site,0] -= 1
                new_sample[empty_site,0] += 1
            else: # hop a spin up electron
                occup_list = [i for i in range(self.num_sites) if sample[i,1]==1]
                empty_list = [i for i in range(self.num_sites) if sample[i,1]==0]
                occup_site = np.random.choice(occup_list)
                empty_site = np.random.choice(empty_list)
                new_sample[occup_site,1] -= 1
                new_sample[empty_site,1] += 1
            new_samples.append(new_sample)

        return tf.convert_to_tensor(np.array(new_samples))

    def sample(self, model, initial_sample, num_samples, num_steps):
        """
            Do a metropolis local sample from a given initial sample
            and model to get \Psi(x).
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample, shape = [num_samples,sample_size]
                num_samples: number of samples returned
            Return:
                new samples, only the final one, no intermediate configurations
                shape = [num_samples,sample_size]
        """
        sample = initial_sample
        wfs = model.log_val(initial_sample)
        for i in range(num_steps):
            sample, wfs = self.sample_once(model, sample, wfs, num_samples)
            if num_steps>50 and num_steps % 50 == 0:
                gc.collect()

        return sample