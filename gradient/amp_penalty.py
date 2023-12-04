import tensorflow as tf
import numpy as np
def get_gradient_amp_penalty(derlogs, sample_size, eloc, num_param_amp):
        """
            Calculate the gradient of E[\Psi] using the modifed stochastic reconfiguration with mass term
            G_ij = < mass**2 * Re[O_i(sk)] Re[O_j(sk)] + Im[O_i(sk)] Im[O_j(sk)] >
            We use the coefficient "mass" to control the training velocity of amplitude

            Args:
                derlogs: $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
                sample_size: the sample size
                eloc: the local energy E_{loc}
        """
        ##### 1. Calculate precondition matrix
        old_shapes = [derlog.shape for derlog in derlogs]

        ## Calculate O_k
        all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)

        ## Calculate <O_k>
        all_derlogs_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        ## Calculate G_ij = < mass**2 * Re[O_i(sk)] Re[O_j(sk)] + Im[O_i(sk)] Im[O_j(sk)] >
        #                   - mass**2 * <Re[O_i(sk)]> * <Re[O_j(sk)]> - <Im[O_i(sk)]> * <Im[O_j(sk)]>
        mass = 4.
        all_derlogs_derlogs_mean = (mass*mass * tf.einsum('ki, kj->ij', tf.math.real(all_derlogs), tf.math.real(all_derlogs)) \
                + tf.einsum('ki, kj->ij', tf.math.imag(all_derlogs), tf.math.imag(all_derlogs)) )/ sample_size \
                - mass*mass * tf.math.real(tf.transpose(all_derlogs_mean)) * tf.math.real(all_derlogs_mean) \
                - tf.math.imag(tf.transpose(all_derlogs_mean)) * tf.math.imag(all_derlogs_mean)

        G_ij = all_derlogs_derlogs_mean
        G_ij = tf.cast(G_ij,tf.complex64)

        ## Regularize G_ij to make sure it is invertible
        regularizer = 0.2
        G_ij_reg = tf.linalg.set_diag(G_ij,tf.linalg.diag_part(G_ij) + regularizer)

        ##### 2. Calculate original energy gradient
        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        ## Calculate <D_{W}>
        derlog_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        ## Calculate <E_loc D^*_{W}>
        ed = tf.reduce_mean(tf.math.conj(all_derlogs) * eloc, axis = 0, keepdims = True)

        ## Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
        grad = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)

        ## Save norm of raw gradient
        grad_norm = [] #order : grad_amp_before, grad_phi_before, grad_amp_after, grad_phi_after
        grad_norm.append(np.real(tf.norm(grad[:,:num_param_amp]).numpy()))
        grad_norm.append(np.real(tf.norm(grad[:,num_param_amp:]).numpy()))

        ##### 3. Calculated modified gradient
        ## inv(G_ij) * grad == final_grads or G_ij * final_grads == grad
        final_grads = tf.linalg.solve(G_ij_reg, tf.transpose(grad))
        
        ## make the norm of wfs more stable:<\psi+\delta \psi | \psi+\delta \psi>=<\psi | \psi>
        # See meeting notes for detail
        '''Ai = tf.math.real(derlog_mean)
        Ai = Ai / tf.norm(Ai) # unit vector
        Ai = tf.transpose(tf.cast(Ai, tf.complex64))
        final_grads = final_grads - tf.reduce_sum(Ai*final_grads) * Ai'''

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

        return grads, grad_norm, tf.math.real(inner_product)