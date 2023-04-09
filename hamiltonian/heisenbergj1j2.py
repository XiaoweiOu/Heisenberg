import tensorflow as tf
from hamiltonian import Hamiltonian
import itertools
import numpy as np
import scipy
import scipy.sparse.linalg

class HeisenbergJ1J2(Hamiltonian): 
    """
    This class is used to define Heisenberg J1-J2 model.
    Nearest neighbor interaction along x-, y- and z-axis with magnitude J_1,
    next nearest neighbor interaction along x-, y- and z-axis with magntitude J_2,
    nearest neighbor interaction along z-axis with magnitude \Delta,.
    
    $H_{HJ} = J_1 \sum_{<i,j>} (\Delta \sigma^z_i \sigma^z_j + \sigma^y_i \sigma^y_j + \sigma^x_i \sigma^x_j) + J_2 \sum_{<i,j>} (\sigma^z_i \sigma^z_j + \sigma^y_i \sigma^y_j + \sigma^x_i \sigma^x_j)$
    """
    def __init__(self, graph, j1=1.0, delta=1.0, j2=1.0, total_sz = None):
        """
        Construct an Heisenberg J1-J2 model.
            
        Args:
            j1: magnitude of the nearest neighbor interaction along x,y,z-axis
            delta: magnitude of the nearest neighbor interaction along z-axis
            j2: magnitude of the next nearest neighbor interaction along x,y,z-axis
            total_sz: total_sz if we want to restrict the hilbert space
        """ 

        Hamiltonian.__init__(self, graph)
        self.j1 = j1
        self.delta = delta
        self.j2 = j2
        self.total_sz = total_sz

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from a given samples x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:(?)
            The Hamiltonian where the first column contains the diagonal, which is  $J_1 \sum_{<i,j>} x_i x_j + J_2 \sum_{<<i,j>>} x_i x_j$.
            The rest of the column contains the off-diagonal, which is (J_x - J_y * x_i * x_j). 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        """

        diagonal = tf.zeros((num_samples,))
        off_diagonal = None
        for (s, s_2) in self.graph.bonds:
            # Diagonal element of the hamiltonian
            # $J_1 \sum_{<i,j>} x_i x_j$
            diagonal += self.j1 * samples[:,s] * samples[:,s_2]

            # Off diagonal element of the hamiltonian
            # $J_1 * (1 - x_i * x_j)$
            if off_diagonal is None:
                off_diagonal = (self.j1 - samples[:,s] * samples[:, s_2])
                off_diagonal = tf.reshape(off_diagonal, (num_samples, 1))
            else:
                temp = (self.j1 - samples[:,s] * samples[:, s_2])
                temp = tf.reshape(temp, (num_samples, 1))
                off_diagonal = tf.concat((off_diagonal, temp ),
                                                 axis=1)
        
        for (s, s_2) in self.graph.bonds_next:
            # Diagonal element of the hamiltonian
            # $J_2 \sum_{<<i,j>>} x_i x_j$
            diagonal += self.j2 * samples[:,s] * samples[:,s_2]

            # Off diagonal element of the hamiltonian
            # $J_2 * (1 - x_i * x_j)$
            if off_diagonal is None:
                off_diagonal = (self.j2 - samples[:,s] * samples[:, s_2])
                off_diagonal = tf.reshape(off_diagonal, (num_samples, 1))
            else:
                temp = (self.j1 - samples[:,s] * samples[:, s_2])
                temp = tf.reshape(temp, (num_samples, 1))
                off_diagonal = tf.concat((off_diagonal, temp ),
                                                 axis=1)

        diagonal = tf.reshape(diagonal, (num_samples, 1))

        hamiltonian = tf.concat((diagonal, off_diagonal), axis=1)

        return hamiltonian
    
    def calculate_ratio(self, samples, model, num_samples):
        """
       Calculate the ratio of \Psi(x') and \Psi(x) from a given x 
        as log(\Psi(x')) - log(\Psi(x))
       \Psi is defined in the model. 
        However, the Hamiltonian determines which x' gives non-zero.
        
        Args:
            samples: the samples x
            model: the model used to define \Psi
            num_samples: the number of samples
        Return:(?) is this a 1D model?
            The ratio where the first column contains \Psi(x) / \Psi(x).
            The rest of the column contains the non-zero \Psi(x') / \Psi(x).
            In the Heisenberg model, this corresponds x' where two adjacent spins are flipped. 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        
        """

        lvd = model.log_val_diff(samples, samples)
        for (s, s2) in self.graph.bonds:
            ## Flip 2 adjacent spin
            flipped_s =  tf.reshape(samples[:,s] * -1 , (num_samples, 1))    
            flipped_s2 = tf.reshape(samples[:,s2] * -1, (num_samples, 1))
            if s == 0:
                new_config = tf.concat((flipped_s, flipped_s2, samples[:,s2+1:]), axis = 1)
            elif s2 == self.graph.num_points-1:  
                new_config = tf.concat((samples[:, :s], flipped_s, flipped_s2), axis = 1)
            else:
                new_config = tf.concat((samples[:, :s], flipped_s, flipped_s2, samples[:,s2+1:]), axis = 1)

            lvd = tf.concat((lvd, model.log_val_diff(new_config, samples)), axis=1)
        for (s, s2) in self.graph.bonds_next:
            ## Flip 2 adjacent spin
            flipped_s =  tf.reshape(samples[:,s] * -1 , (num_samples, 1))    
            flipped_s2 = tf.reshape(samples[:,s2] * -1, (num_samples, 1))
            ## Store the configuration between s and s2
            middle_conf =  tf.reshape(samples[:,s+1], (num_samples, 1))    
            if s == 0:
                new_config = tf.concat((flipped_s, middle_conf, flipped_s2, samples[:,s2+1:]), axis = 1)
            elif s2 == self.graph.num_points-1:  
                new_config = tf.concat((samples[:, :s], flipped_s, middle_conf, flipped_s2), axis = 1)
            else:
                new_config = tf.concat((samples[:, :s], flipped_s, middle_conf, flipped_s2, samples[:,s2+1:]), axis = 1)

            lvd = tf.concat((lvd, model.log_val_diff(new_config, samples)), axis=1)
        return lvd