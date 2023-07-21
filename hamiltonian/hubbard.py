import tensorflow as tf
import numpy as np

class Hubbard():
    """
    This class is used to define Hubbard Hamiltonian.
    We consider bases in momentum space, instead of real space.
    """
    def __init__(self, graph, t_coeff, U_coeff):
        """
        Construct a Hubbard model.

        Args:
            t_coeff: magnitude of the hopping term
            U_coeff: magnitude of the Coulomb on-site interaction.
        """
        self.length = graph.length #length of the lattice
        self.t_coeff = t_coeff
        self.U_coeff = U_coeff

    def local_energy(self, in_state, model):
        """
        Calculate local energy for one in_state.
        """
        eloc = 0

        ## Hopping term
        eloc += self.calculate_hopping_term(in_state)
        eloc = tf.cast(eloc,tf.complex64)

        ## Coulomb term
        out_states, out_hamils = self.calculate_coulomb_term(in_state)
        for (out_state, out_hamil) in zip(out_states, out_hamils):
            # eloc += tf.cast(out_hamil, tf.complex64) * np.exp(model.log_val_diff_one_sample(out_state,in_state))
            eloc += tf.cast(out_hamil, tf.complex64) * model.log_val_one_sample(out_state) / model.log_val_one_sample(in_state)

        return eloc

    def calculate_hopping_term(self, in_state):
        """
        In k space, the hopping term is diagonal.
        -2t \sum_{k,spin} c^+_{k,spin} c_{k,spin}*[cos(kx*a)+cos(ky*a)]

        Args:
            in_state: input configuration,shape=(num_particles,2)=(num_kpoints,2)
                      This is not one-hot encoding, but occupation number for each k points.
                      [:,0] is spin down; [:,1] is spin up.

        Return:
        Hopping Hamiltonian matrix element
        """
        result = 0

        for kindex,(num_down,num_up) in enumerate(in_state):
            (kx,ky)=self.kindex2kvector(kindex) #unit: 2pi/a
            num = num_down + num_up # Total number of particles for this k point
            result += (np.cos(kx*2*np.pi)+np.cos(ky*2*np.pi))*num

        return result * 2 * self.t_coeff

    def kindex2kvector(self, kindex):
        """
        Relate the index of k points to the k vector.
        Set 2pi/a=1 for simplicity
        Args:
            kindex:
        Return:
            kvector:
        """
        ## (i,j) for the matrix in k space
        i = kindex // self.length
        j = kindex % self.length

        ## from (i,j) to (kx,ky)
        ky = -1/self.length*i + 0.5
        kx = 1/self.length*j - 0.5 + 1/self.length

        return [kx,ky]

    def kvector2kindex(self, kvector):
        """
        Relate the k vector to the index of k points.
        Set pi/a=1 for simplicity
        Args:
            kvector:
        Return:
            kindex:
        """
        i = int((-self.length)*(kvector[1]-0.5))
        j = int(self.length*(kvector[0]+0.5)-1)

        kindex = i*self.length + j
        return kindex

    def first_brillouin(self, kvector):
        """
        Reduce any kvector to the first Brillouin zone.
        """
        kx = kvector[0]%1
        ky = kvector[1]%1
        if kx > 0.5:
            kx -= 1
        if ky > 0.5:
            ky -= 1
        return [kx,ky]

    def calculate_coulomb_term(self, in_state):
        """
        In k space, the Coulomb term becomes off-diagonal.
        U*\sum_{k1,k2,k3,k4} c^+_{k1,up} c_{k2,up} c^+_{k3,down} c_{k4,down} delta(k1+k3-k2-k4)

        Args:
            in_state: input configuration,shape=(num_particles,2)=(num_kpoints,2)
                  This is not one-hot encoding, but occupation number for each k points.
                  [:,0] is spin down; [:,1] is spin up.

        Return:
            A dictionary: all possible output states, and the corresponding matrix elements (sign*U).
        """
        output={}

        ## loop through all possible out states
        for k4 in range(len(in_state)):
            for k3 in range(len(in_state)):
                for k2 in range(len(in_state)):
                    sign = 1
                    state = in_state.numpy()

                    #k1 = k2+k4-k3
                    k1=[]
                    k1.append(self.kindex2kvector(k2)[0]+self.kindex2kvector(k4)[0]-self.kindex2kvector(k3)[0])
                    k1.append(self.kindex2kvector(k2)[1]+self.kindex2kvector(k4)[1]-self.kindex2kvector(k3)[1])
                    k1 = self.kvector2kindex(self.first_brillouin(k1))

                    if(state[k4,0]==0):#spin down
                        continue
                    else:
                        state[k4,0]-=1
                        sign *= (-1)**sum(state[j,k] for j in range(k4) for k in range(2))
                                  
                    if(state[k3,0]==1):
                        continue
                    else:
                        state[k3,0]+=1
                        sign *= (-1)**sum(state[j,k] for j in range(k3) for k in range(2))
                                  
                    if(state[k2,1]==0):
                        continue
                    else:
                        state[k2,1]-=1
                        sign *= (-1)**(sum(state[j,k] for j in range(k2) for k in range(2)) + state[k2,0])

                    if(state[k1,1]==1):
                       continue
                    else:
                        state[k1,1]+=1
                        sign *= (-1)**(sum(state[j,k] for j in range(k1) for k in range(2)) + state[k1,0])

                    ## Add to the lists
                    if state.tobytes() in output:
                        output[state.tobytes()] += sign*self.U_coeff/len(in_state)
                    else:
                        output.update({state.tobytes():sign*self.U_coeff/len(in_state)})

        neighbor_states = [np.reshape(np.frombuffer(ele,state.dtype),state.shape) for ele in output.keys()]

        return tf.convert_to_tensor(neighbor_states),list(output.values())
