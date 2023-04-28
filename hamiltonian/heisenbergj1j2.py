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
    next nearest neighbor interaction along x-, y- and z-axis with magntitude J_2.
    
    $H = J_1 \sum_{<i,j>} (S^z_i S^z_j + S^y_i S^y_j + S^x_i S^x_j) + J_2 \sum_{<<i,j>>} (S^z_i S^z_j + S^y_i S^y_j + S^x_i S^x_j)
    = J_1 \sum_{<i,j>} (S^z_i S^z_j + 1/2 * (S^+_i S^-_j + S^-_i S^+_j)) + J_2 \sum_{<<i,j>>} (S^z_i S^z_j + 1/2 * (S^+_i S^-_j + S^-_i S^+_j))$
    """
    def __init__(self, graph, j1=1.0, j2=0.0, total_sz = None):
        """
        Construct an Heisenberg J1-J2 model.
            
        Args:
            j1: magnitude of the nearest neighbor interaction along x,y,z-axis
            j2: magnitude of the next nearest neighbor interaction along x,y,z-axis
            total_sz: total_sz if we want to restrict the hilbert space
        """ 

        Hamiltonian.__init__(self, graph)
        self.j1 = j1
        self.j2 = j2
        self.total_sz = total_sz

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from a given samples x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:
            The Hamiltonian where the first column contains the diagonal, which is  $J_1 \sum_{<i,j>} z_i z_j / 4 + J_2 \sum_{<<i,j>>} z_i z_j / 4$.
            Because only S^z_i*S^z_j contribute to the diagonal term.
            
            The rest of the column contains the off-diagonal, which is $J_1 * (1 - z_i * z_j) / 4 + J_2 * (1 - z_i * z_j) / 4$.
            Considering the raising and lowering operators, it gives a 1/2 if z_i * z_j == -1, and 0 otherwise.

            Therefore, #columns = 1+ #nearest bond+ #next-nearest bond = 1+4*num_particles
            #rows = num_samples
        """

        diagonal = tf.zeros((num_samples,))
        off_diagonal = None
        for (s, s_2) in self.graph.bonds:
            # Diagonal element of the hamiltonian
            # $J_1 \sum_{<i,j>} z_i z_j / 4$
            diagonal += self.j1 * samples[:,s] * samples[:,s_2] / 4

            # Off diagonal element of the hamiltonian
            # $J_1 * (1 - z_i * z_j) / 4$
            if off_diagonal is None:
                off_diagonal = self.j1 * (1 - samples[:,s] * samples[:, s_2]) / 4
                off_diagonal = tf.reshape(off_diagonal, (num_samples, 1))
            else:
                temp = self.j1 * (1 - samples[:,s] * samples[:, s_2]) / 4
                temp = tf.reshape(temp, (num_samples, 1))
                off_diagonal = tf.concat((off_diagonal, temp), axis=1)
        
        for (s, s_2) in self.graph.bonds_next:
            # Diagonal element of the hamiltonian
            # $J_2 \sum_{<<i,j>>} z_i z_j / 4$
            diagonal += self.j2 * samples[:,s] * samples[:,s_2] / 4

            # Off diagonal element of the hamiltonian
            # $J_2 * (1 - z_i * z_j) / 4$
            if off_diagonal is None:
                off_diagonal = self.j2 * (1 - samples[:,s] * samples[:, s_2]) / 4
                off_diagonal = tf.reshape(off_diagonal, (num_samples, 1))
            else:
                temp = self.j2 * (1 - samples[:,s] * samples[:, s_2]) / 4
                temp = tf.reshape(temp, (num_samples, 1))
                off_diagonal = tf.concat((off_diagonal, temp), axis=1)

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
        Return:
            The ratio where the first column contains \Psi(x) / \Psi(x).
            The rest of the column contains the non-zero \Psi(x') / \Psi(x).
            In the Heisenberg model, this corresponds x' where two adjacent spins are exchanged. 
            Therefore, # column = 1 + num_bonds + num_bonds_next = 1 + 4 * num_particles
            # rows = num_samples
        
        """

        lvd = model.log_val_diff(samples, samples)

        for (s, s2) in self.graph.bonds:
            ## Swap 2 adjacent spin
            newcol_s =  tf.reshape(samples[:,s2], (num_samples, 1))    
            newcol_s2 = tf.reshape(samples[:,s], (num_samples, 1))

            new_config = tf.identity(samples)
            new_config = tf.concat((new_config[:,:s], newcol_s, new_config[:,s+1:]), axis = 1)
            new_config = tf.concat((new_config[:,:s2], newcol_s2, new_config[:,s2+1:]), axis = 1)

            lvd = tf.concat((lvd, model.log_val_diff(new_config, samples)), axis=1)

        for (s, s2) in self.graph.bonds_next:
            ## Swap 2 adjacent spin
            newcol_s =  tf.reshape(samples[:,s2], (num_samples, 1))    
            newcol_s2 = tf.reshape(samples[:,s], (num_samples, 1))

            new_config = tf.identity(samples)
            new_config = tf.concat((new_config[:,:s], newcol_s, new_config[:,s+1:]), axis = 1)
            new_config = tf.concat((new_config[:,:s2], newcol_s2, new_config[:,s2+1:]), axis = 1)

            lvd = tf.concat((lvd, model.log_val_diff(new_config, samples)), axis=1)
        return lvd
    
    def diagonalize(self):
        """
        Diagonalize hamiltonian with exact diagonalization.
        Only works for small systems (<= 10)!
        """
        num_particles = self.graph.num_points
        ## Initialize zeroes hamiltonian
        H = np.zeros((2 ** num_particles, 2 ** num_particles), dtype='complex')

        ## Calculate interaction energy
        for i, a in self.graph.bonds:
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            togg_vect[a] = 1

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_X)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j1 * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Y)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j1 * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Z)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j1 * temp

        ## Calculate interaction energy
        for i, a in self.graph.bonds_next:
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            togg_vect[a] = 1

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_X)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j2 * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Y)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j2 * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Z)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.j2 * temp

        ## Filter total sz
        if self.total_sz is not None:
            index = []
            num_confs = 2 ** num_particles
            for row in range(num_confs):
                ## configuration in binary 0 1
                conf_bin = format(row, '#0%db' % (num_particles + 2))
                ## configuration in binary -1 1
                conf = [1 if c == '1' else -1 for c in conf_bin[2:]]
                
                if np.sum(conf) == self.total_sz:
                    index.append(row)
            
            H = H[index] 
            H = H[:, index]
       

        ## Calculate the eigen value
        self.eigen_values, self.eigen_vectors = np.linalg.eig(H)
        self.hamiltonian = H