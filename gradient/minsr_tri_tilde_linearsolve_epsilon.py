import tensorflow as tf
import numpy as np
import scipy
def get_gradient_minsr_tri_tilde_linearsolve_epsilon(derlogs, sample_size, eloc, num_param_amp):
    """
        Calculate the gradient of E[\Psi] using the modifed stochastic reconfiguration with mass term
        G_ij = < mass**2 * Re[O_i(sk)] Re[O_j(sk)] + Im[O_i(sk)] Im[O_j(sk)] >
        We use the coefficient "mass" to control the training velocity of amplitude

        Args:
            derlogs: $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
            sample_size: the sample size
            eloc: the local energy E_{loc}
    """
    mass = 11.
    regularizer = 0.1

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
    O_mat_stack = tf.concat([tf.math.real(O_mat), tf.math.imag(O_mat)], axis = 0).numpy().astype(np.float64)
    R_vec_stack = tf.concat([tf.math.real(R_vec), tf.math.imag(R_vec)], axis = 0).numpy().astype(np.float64)
    R_vec_weighted_stack = tf.concat([tf.math.real(R_vec), mass * tf.math.imag(R_vec)], axis = 0).numpy().astype(np.float64)

    ## Raw gradient
    grad = tf.cast(O_mat_stack.T @ R_vec_stack, tf.float32)

    ## Calculate preconditioned gradient with mass term.
    T_mat = O_mat_stack @ O_mat_stack.T
    T_mat_reg = T_mat + np.eye(T_mat.shape[1])*regularizer
    cholesky = scipy.linalg.cholesky(T_mat_reg, lower=True)
    intermediate = scipy.linalg.cho_solve((cholesky, True),R_vec_weighted_stack)
    final_grads = tf.cast(O_mat_stack.T @ intermediate, tf.float32)

    ## inner product <grad|final_grads>
    inner_product = tf.einsum('i,i',tf.squeeze(grad),tf.squeeze(final_grads))

    grads = []
    prev = 0
    for old_shape in old_shapes:
        final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]   
        prev += tf.reduce_prod(old_shape[1:])
        grads.append(tf.reshape(final_grad, old_shape[1:]))

    return grads,tf.math.real(inner_product)