import tensorflow as tf
import numpy as np
def get_gradient_minsr(derlogs, sample_size, eloc, num_param_amp, mass = 1.):
        """
            Calculate the gradient of E[\Psi] using the minSR method
            Args:
                derlogs: $D_{W}(x) = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
                sample_size: the sample size
                eloc: the local energy E_{loc}
                mass: reweighting factor for amplitude part
        """
        old_shapes = [derlog.shape for derlog in derlogs]

        ## Calculate O_mat, shape = (N_batch, N_param)
        all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)
        all_derlogs_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)
        O_mat = (all_derlogs - all_derlogs_mean) / tf.math.sqrt(tf.cast(sample_size,tf.complex64))

        ## Calculate weighted O_mat_weighted
        # all_derlogs_weighted = tf.complex(mass * tf.math.real(all_derlogs), tf.math.imag(all_derlogs))
        # all_derlogs_weighted_mean = tf.reduce_mean(all_derlogs_weighted, axis=0, keepdims=True)
        # O_mat_weighted = (all_derlogs_weighted - all_derlogs_weighted_mean) / tf.math.sqrt(tf.cast(sample_size,tf.complex64))

        ## Calculate R_vec, shape = (N_batch)
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)
        R_vec = (eloc - eloc_mean) / tf.math.sqrt(tf.cast(sample_size,tf.complex64))

        ## Stack real and imaginary parts, 
        ## when the neural network parameters are real but the network outputs can be complex
        ## O_mat shape = (2 N_batch, N_param)
        ## R_vec shape = (2 N_batch)
        # O_mat_stack = tf.concat([tf.math.real(O_mat), tf.math.imag(O_mat)], axis = 0)
        # O_mat_weighted_stack = tf.concat([tf.math.real(O_mat_weighted), tf.math.imag(O_mat_weighted)], axis = 0)
        # R_vec_stack = tf.concat([tf.math.real(R_vec), tf.math.imag(R_vec)], axis = 0)

        ## Raw gradient
        grad = tf.cast(tf.linalg.adjoint(O_mat) @ R_vec, tf.complex64)

        ## Save norm of raw gradient
        grad_norm = []
        grad_norm.append(np.real(tf.norm(grad[:num_param_amp,:]).numpy()))
        grad_norm.append(np.real(tf.norm(grad[num_param_amp:,:]).numpy()))

        ## Calculated preconditioned gradient. 
        lambd = 1e-3
        r_pinv = 1e-12
        a_pinv = 0
        soft = True

        T_mat = O_mat @ tf.linalg.adjoint(O_mat)
        T_mat_reg = tf.linalg.set_diag(T_mat, tf.linalg.diag_part(T_mat) + lambd)
        D, U = tf.linalg.eigh(T_mat_reg)

        ## For a self-adjoint matrix, the eigenvalues and eigenvectors are real-valued.
        D, U = tf.math.real(D), tf.math.real(U)

        threshold = tf.math.abs(tf.math.reduce_max(D)) * r_pinv + a_pinv
        if soft:
            D_inv = 1 / (D * (1 + (threshold / tf.math.abs(D)) ** 6))
        else:
            D_inv = tf.where(tf.math.abs(D) >= threshold, 1 / D, tf.convert_to_tensor(0.0, dtype=O_mat.dtype))
        T_inv = (U * D_inv) @ tf.linalg.adjoint(U)
        # final_grads = tf.cast(tf.linalg.adjoint(O_mat_stack) @ T_inv @ R_vec_stack, tf.complex64)

        T_inv = tf.cast(T_inv, tf.complex64)
        final_grads = tf.linalg.adjoint(O_mat) @ T_inv @ R_vec

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