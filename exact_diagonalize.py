from scipy.sparse.linalg import eigs,eigsh 
from scipy.sparse import dok_matrix
import numpy as np

def Diagonalize_test(num_confs,ham_idx,ham_ele):
    sparse_array = dok_matrix((num_confs, num_confs), dtype=np.complex64)
    length = len(ham_ele)
    for i in range(length):
        sparse_array[ham_idx[i][0],ham_idx[i][1]] = ham_ele[i]
    vals, vecs = eigs(sparse_array,k=1,which='SR')
    return vals, vecs