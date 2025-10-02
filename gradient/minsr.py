import tensorflow as tf
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def pseudo_inverse(matrix, r_pinv = 1e-7, a_pinv = 0, soft = False):
    """
       Require the matrix to be real and double-precision.
    """
    matrix = tf.math.real(matrix)
    D, U = tf.linalg.eigh(matrix)

    # threshold = tf.math.abs(tf.math.reduce_max(D)) * r_pinv + a_pinv
    threshold = 1e-8
    if soft:
        D_inv = 1 / (D * (1 + (threshold / tf.math.abs(D)) ** 6))
    else:
        D_inv = tf.where(tf.math.abs(D) >= threshold, 1 / D, tf.convert_to_tensor(0.0, dtype=matrix.dtype))
    matrix_inv = U @ tf.linalg.diag(D_inv) @ tf.linalg.adjoint(U)

    return matrix_inv

def get_gradient_minsr(derlogs, sample_size, eloc, num_param_amp):
    """
        Calculate the gradient of E[\Psi] using the minSR method
        Args:
            derlogs: $D_{W}(x) = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
            sample_size: the sample size
            eloc: the local energy E_{loc}
    """
    mass = 4.
    regularizer = 0.2

    old_shapes = [derlog.shape for derlog in derlogs]

    ## Calculate O_mat, shape = (N_sample, N_param)
    all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)
    all_derlogs_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)
    O_mat = (all_derlogs - all_derlogs_mean) / tf.math.sqrt(tf.cast(sample_size,tf.complex64))

    ## Calculate R_vec, shape = (N_sample)
    eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)
    R_vec = (eloc - eloc_mean) / tf.math.sqrt(tf.cast(sample_size,tf.complex64))

    ## Stack real and imaginary parts, 
    ## when the neural network parameters are real but the network outputs can be complex
    ## O_mat_stack shape = (2 N_sample, N_param) dtype:tf.float
    ## R_vec_stack shape = (2 N_sample) dtype:tf.float
    O_mat_stack = tf.concat([tf.math.real(O_mat), tf.math.imag(O_mat)], axis = 0).numpy()
    O_mat_weighted_stack = tf.concat([mass * tf.math.real(O_mat), tf.math.imag(O_mat)], axis = 0).numpy()
    R_vec_stack = tf.concat([tf.math.real(R_vec), tf.math.imag(R_vec)], axis = 0).numpy()

    ## Raw gradient
    grad = tf.cast(tf.transpose(O_mat_stack) @ R_vec_stack, tf.complex64)
    grad_norm = []
    grad_norm.append(np.real(tf.norm(grad[:num_param_amp,:]).numpy()))
    grad_norm.append(np.real(tf.norm(grad[num_param_amp:,:]).numpy()))

    ## Calculate preconditioned gradient with mass term.
    T_mat = O_mat_weighted_stack @ O_mat_weighted_stack.T
    T_mat_reg = T_mat + np.eye(T_mat.shape[1])*0.09
    T_mat_reg2 = T_mat + np.eye(T_mat.shape[1])*0.09  # Small correction
    T_reg_inv = np.linalg.inv(T_mat_reg)
    T_reg_inv2 = np.linalg.inv(T_mat_reg2)
    final_grads = tf.convert_to_tensor(O_mat_weighted_stack.T @ T_reg_inv @ T_reg_inv2 @ O_mat_weighted_stack @ O_mat_stack.T @ R_vec_stack, tf.complex64)

    ## Save norm of preconditioned gradient
    grad_norm.append(np.real(tf.norm(final_grads[:num_param_amp,:]).numpy()))
    grad_norm.append(np.real(tf.norm(final_grads[num_param_amp:,:]).numpy()))

    ## inner product <grad|final_grads>
    inner_product = tf.einsum('i,i',tf.squeeze(grad),tf.squeeze(final_grads))

    grads = []
    prev = 0
    for old_shape in old_shapes:
        final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]   
        prev += tf.reduce_prod(old_shape[1:])
        grads.append(tf.reshape(final_grad, old_shape[1:]))

    return grads,grad_norm,tf.math.real(inner_product)