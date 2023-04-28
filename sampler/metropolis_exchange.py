from sampler import Sampler
import tensorflow as tf
import numpy as np
import gc


class MetropolisExchange(Sampler):
    """
    This class is used to do a metropolis exchange sampling. 
    In metropolis exchange, the next sample is determined by flipping two random spins.
    It is similar to metropolis local, the difference is only on the get_new_config.
    This sampling is used to maintain total sz to zero particularly in Heisenberg model.
    """

    def __init__(self, num_samples, total_sz=0):
        Sampler.__init__(self, num_samples)
        self.total_sz = total_sz

    # set total Sz to be 0
    def get_initial_random_samples(self, sample_size, num_samples=None):
        """
            Get initial random samples with size [num_samples, sample_size]
            from a random uniform. However it must conforms to the total sz specification
            Args:
                sample_size: the number of particles
                num_samples: number of samples
            Return:
                initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        assert self.total_sz <= sample_size
        assert (self.total_sz + sample_size) % 2 == 0

        ## mix +1 and -1 equally
        plus = np.ones(int((sample_size + self.total_sz)/2)) * 1.0
        minus = np.ones(int((sample_size - self.total_sz)/2)) * -1.0
        model = np.concatenate((plus, minus))

        ## shuffle them
        init_data = []
        for i in range(num_samples):
            data = np.copy(model)
            np.random.shuffle(data)
            init_data.append(data)

        init_data = np.array(init_data, np.float32)
        init_data = np.reshape(init_data, (num_samples, sample_size))

        return init_data

    def get_initial_sample_from_checkerboard(self, length, num_samples=None):
        """
        Since it usually takes a long time to equilibrate from the initial random sample, 
        here I generate the initial sample starting from checkerboard.
        (TODO) Now there is a problem that with this method,
        S_kk_reg in stochastic reconfiguration has too many zero eigenvalues even after the regulation.
        Args:
            length: the length of lattice (should always be even number)
            num_samples: number of samples
        Return:
            initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        small_1 = [1,-1]
        small_2 = [-1,1]
        checkerboard = []
        checkerboard.append(small_1 * int(length/2))
        checkerboard.append(small_2 * int(length/2))
        checkerboard = checkerboard * int(length/2)

        ## shuffle them
        init_data = []
        for i in range(num_samples):
            data = np.copy(checkerboard)
            np.random.shuffle(data)
            init_data.append(data)

        init_data = np.array(init_data, np.float32)
        init_data = np.reshape(init_data, (num_samples, length ** 2))

        return init_data

    def sample_once(self, model, starting_sample, num_samples):
        """
            Do a one metropolis step from a given starting samples.
            Generate [num_samples * 1] new samples. 
            Model is used to calculate probability.
            Args:
                model: the model to calculate probability |\Psi(x)|^2
                starting_samples: the initial samples
                num_samples: number of samples returned
            Return:
                new samples from one metropolis exchange
        """
        ## Get new configuration by flipping two random spins (?)
        new_config = self.get_new_config(starting_sample, num_samples)

        ## Calculate the ratio of the new configuration and old configuration probability by computing |psi(x') / psi(x)|^2
        ratio = tf.abs(tf.exp(model.log_val_diff(new_config, starting_sample))) ** 2
        
        ## Random number
        random = tf.random.uniform((num_samples, 1), 0, 1)

        ## Calculate acceptance
        accept = tf.squeeze(tf.greater(ratio, random))
        accept = tf.broadcast_to(tf.reshape(accept, (accept.shape[0], 1)), starting_sample.shape)

        ## Reject and accept samples
        sample = tf.where(accept, new_config, starting_sample)
        return sample

    def get_new_config(self, sample, num_samples):
        """
            Get a new configuration by flipping two random spins
            Args: 
                sample: the samples that want to be flipped randomly
                num_samples: the number of samples
            Return:
                new samples with two randomly flipped spins
        """
        num_points = int(sample.shape[1])
        position1 = np.random.randint(0, num_points, num_samples)
        position2 = np.random.randint(0, num_points, num_samples)
        row_indices = np.reshape(range(num_samples), (num_samples, 1))
        col_indices1 = np.reshape(position1, (num_samples, 1))
        col_indices2 = np.reshape(position2, (num_samples, 1))
        indices1 = tf.convert_to_tensor(value=np.concatenate((row_indices, col_indices1), axis=1))
        indices2 = tf.convert_to_tensor(value=np.concatenate((row_indices, col_indices2), axis=1))
        elements1 = tf.gather_nd(sample, indices1)
        elements2 = tf.gather_nd(sample, indices2)
        old1 = tf.scatter_nd(indices1, elements1, (num_samples, num_points))
        old2 = tf.scatter_nd(indices2, elements2, (num_samples, num_points))
        new1 = tf.scatter_nd(indices1, elements2, (num_samples, num_points))
        new2 = tf.scatter_nd(indices2, elements1, (num_samples, num_points))
        return sample - old1 - old2 + new1 + new2

    def sample(self, model, initial_sample, num_samples, num_steps):
        """
            Do a metropolis local sample from a given initial sample
            and model to get \Psi(x).
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample, shape = [num_sampels,sample_size]
                num_samples: number of samples returned
            Return:
                new samples, only the final one, no intermediate configurations
                shape = [num_sampels,sample_size]
        """
        sample = initial_sample
        for i in range(num_steps):
            sample = self.sample_once(model, sample, num_samples)
            if num_steps>10 and num_steps % 10 == 0:
                gc.collect()

        return sample

    def get_all_samples(self, model, initial_sample, num_samples, num_steps):
        """
        This is the same as the function sample, but it returns all samples rather than only the last one.
        """
        all_samples = []
        sample = initial_sample
        for i in range(num_steps):
            sample = self.sample_once(model, sample, num_samples)
            all_samples.append(sample)

        return all_samples