import copy
import numpy as np
np.set_printoptions(precision=4,threshold=np.inf)

def calculate_hopping_term(in_state, t_coeff):
    """
    In k space, the hopping term is diagonal.
    -2t \sum_{k,spin} c^+_{k,spin} c_{k,spin}*[cos(kx*a)+cos(ky*a)]

    Args:
        in_state: input configuration,shape=(num_particles,2)=(num_kpoints,2)
                  This is not one-hot encoding, but occupation number for each k points.
                  [:,0] is spin down; [:,1] is spin up.
        t_coeff

    Return:
        Diagonal Hamiltonian matrix element
    """
    length = int(np.sqrt(len(in_state)))
    result = 0

    for kindex,(num_down,num_up) in enumerate(in_state):
        (kx,ky)=kindex2kvector(kindex,length) #unit: 2pi/a
        num = num_down + num_up # Total number of particles for this k point
        result += (np.cos(kx*2*np.pi)+np.cos(ky*2*np.pi))*num

    return result * 2 * t_coeff

def kindex2kvector(kindex,length):
    """
    Relate the index of k points to the k vector.
    Set 2pi/a=1 for simplicity
    Args:
        kindex:
        length: length of lattice
    Return:
        kvector:
    """
    ## (i,j) for the matrix in k space
    i = kindex // length
    j = kindex % length

    ## from (i,j) to (kx,ky)
    ky = -1/length*i + 0.5
    kx = 1/length*j - 0.5 + 1/length

    return [kx,ky]

def kvector2kindex(kvector,length):
    """
    Relate the k vector to the index of k points.
    Set pi/a=1 for simplicity
    Args:
        kvector:
        length: length of lattice
    Return:
        kindex:
    """
    i = int((-length)*(kvector[1]-0.5))
    j = int(length*(kvector[0]+0.5)-1)

    kindex = i*length + j
    return kindex

def first_brillouin(kvector):
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

def calculate_coulomb_term(in_state, U_coeff):
    """
    In k space, the Coulomb term becomes off-diagonal.
    U*\sum_{k1,k2,k3,k4} c^+_{k1,up} c_{k2,up} c^+_{k3,down} c_{k4,down} delta(k1+k3-k2-k4)

    Args:
        in_state: input configuration,shape=(num_particles,2)=(num_kpoints,2)
                  This is not one-hot encoding, but occupation number for each k points.
                  [:,0] is spin down; [:,1] is spin up.
        U_coeff

    Return:
        All possible output states, and the corresponding matrix elements (sign*U).
    """
    length = int(np.sqrt(len(in_state)))
    output={}

    ## loop through all possible out states
    for k4 in range(len(in_state)):
        for k3 in range(len(in_state)):
            for k2 in range(len(in_state)):
                sign = 1
                state = copy.deepcopy(in_state)

                #k1 = k2+k4-k3
                k1=[]
                k1.append(kindex2kvector(k2,length)[0]+kindex2kvector(k4,length)[0]-kindex2kvector(k3,length)[0])
                k1.append(kindex2kvector(k2,length)[1]+kindex2kvector(k4,length)[1]-kindex2kvector(k3,length)[1])
                k1 = kvector2kindex(first_brillouin(k1),length)

                if(state[k4][0]==0):#spin down
                    continue
                else:
                    state[k4][0]-=1
                    sign *= (-1)**sum(state[j][k] for j in range(k4) for k in range(2))
                                  
                if(state[k3][0]==1):
                    continue
                else:
                    state[k3][0]+=1
                    sign *= (-1)**sum(state[j][k] for j in range(k3) for k in range(2))
                                  
                if(state[k2][1]==0):
                    continue
                else:
                    state[k2][1]-=1
                    sign *= (-1)**(sum(state[j][k] for j in range(k2) for k in range(2)) + state[k2][0])

                if(state[k1][1]==1):
                    continue
                else:
                    state[k1][1]+=1
                    sign *= (-1)**(sum(state[j][k] for j in range(k1) for k in range(2)) + state[k1][0])

                ## Add to the lists
                if str(state) in output:
                    output[str(state)] += sign*U_coeff/len(in_state)
                else:
                    output.update({str(state):sign*U_coeff/len(in_state)})

    return [eval(ele) for ele in list(output.keys())],list(output.values())

def generate_all_bases():
    """
    Generate all possible basis configurations allowed by total_sz=0 and Pauli exclusion principle.
    Half-filling
    We will follow the order of all these bases.

    Restriction: 
        total_sz=0: num_up = num_down
        half filling: num_particles = num_sites

    Shape of each configuration=(num_particles,2)=(num_kpoints,2)
    This is not one-hot encoding, but occupation number for each k points.

    Total number of bases=(Combination^{num_particles/2}_{num_particles}) ** 2

    Return:
        all basis
    """
    #simple implementation for 2 by 2 lattice: C^2_4
    C42 = [[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1]]
    all_states = []
    for ele_down in C42:
        for ele_up in C42:
            state=[]
            state.append(ele_down)
            state.append(ele_up)
            np_state = np.array(state)
            all_states.append(np_state.T.tolist())
    return all_states

def generate_all_bases_one_electron():
    all_states = [[[1, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [1, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [1, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 1], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [1, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 1]]]
    return all_states

##### Diagonalize Hamiltonian

## 1. Construct the Hamiltonian matrix
all_states = generate_all_bases()
#all_states = generate_all_bases_one_electron()

## 2. Setting parameters
t_coeff, U_coeff = 1,8
hamiltonian = np.zeros((len(all_states),len(all_states)))

## 3. Calculate the diagonal term
for i,state in enumerate(all_states):
    hamiltonian[i][i]=calculate_hopping_term(state,t_coeff)

    ## 4. Calculate the off-diagonal term
    out_states, out_hamils = calculate_coulomb_term(state,U_coeff)
    for (out_state,out_hamil) in zip(out_states,out_hamils):
        j = all_states.index(out_state)
        hamiltonian[i][j]+=out_hamil

assert (hamiltonian == hamiltonian.T).all()

print('### INFO ### diagonal elements')
for i,diag in enumerate(np.diagonal(hamiltonian)):
    print(i,all_states[i],diag)

print('### INFO ### hamiltonian matrix')
print(hamiltonian)

## The column eigenvectors[:,i] is corresponding to the eigenvalue[i].
## not row!
eigvals, eigvecs = np.linalg.eig(hamiltonian)
zipped = zip(eigvals,eigvecs.T)
result = sorted(zipped, key = lambda x:x[0])

## ground state info
print('### INFO ### ground state')
print('E:',result[0][0])
print('eigvec: basis --- probability --- wave function')
eigvec_gs = result[0][1]
# zip basis and the corresponding probability
zipped_eigvec_gs = zip(all_states,np.real(eigvec_gs*np.conjugate(eigvec_gs)),eigvec_gs)
sorted_eigvec_gs = sorted(zipped_eigvec_gs, key = lambda x:x[1])
for ele in sorted_eigvec_gs[::-1]:
    print(ele)