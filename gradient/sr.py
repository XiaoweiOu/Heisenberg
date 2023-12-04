import tensorflow as tf
import numpy as np
def get_gradient_sr(derlogs, sample_size, eloc, num_param_amp):
        """
            Calculate the gradient of E[\Psi] using the stochastic reconfiguration
            S_kk = <O^*_k O_k> - <O^*_k><O_k>
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

        ## Calculate <O^*_k O_k>
        all_derlogs_derlogs_mean = tf.einsum('ki, kj -> ij', tf.math.conj(all_derlogs), all_derlogs)/ sample_size

        ## Calculate S_kk = <O^*_k O_k> - <O^*_k><O_k>
        S_kk = all_derlogs_derlogs_mean - tf.math.conj(tf.transpose(all_derlogs_mean)) * all_derlogs_mean

        ## Regularize S_kk to make sure it is invertible
        regularizer = 0.2

        # S_kk_diag_reg = tf.linalg.tensor_diag(regularizer * tf.linalg.diag_part(S_kk))
        # S_kk_reg = S_kk + S_kk_diag_reg
        S_kk_reg = tf.linalg.set_diag(S_kk,tf.linalg.diag_part(S_kk) + regularizer)

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
        grad_norm = []
        grad_norm.append(np.real(tf.norm(grad[:,:num_param_amp]).numpy()))
        grad_norm.append(np.real(tf.norm(grad[:,num_param_amp:]).numpy()))

        ##### 3. Calculated modified gradient
        ## inv(S_kk) * grad == final_grads or S_kk * final_grads == grad
        final_grads = tf.linalg.solve(S_kk_reg, tf.transpose(grad))

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

        return grads,grad_norm,tf.math.real(inner_product)

def _test_get_gradient_sr_distinct_config(derlogs, sample_size, eloc, prob, wfs):
        """
            Calculate the gradient of E[\Psi] using the stochastic reconfiguration
            S_kk = <O^*_k O_k> - <O^*_k><O_k>
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
        all_derlogs_mean = tf.reduce_sum(all_derlogs * prob, axis=0, keepdims=True)

        ## Calculate <O^*_k O_k>
        all_derlogs_derlogs_mean = tf.einsum('k, ki, kj -> ij', tf.squeeze(prob), tf.math.conj(all_derlogs), all_derlogs)

        ## Calculate S_kk = <O^*_k O_k> - <O^*_k><O_k>
        S_kk = all_derlogs_derlogs_mean - tf.math.conj(tf.transpose(all_derlogs_mean)) * all_derlogs_mean

        ## Regularize S_kk to make sure it is invertible
        regularizer = 0.2

        # S_kk_diag_reg = tf.linalg.tensor_diag(regularizer * tf.linalg.diag_part(S_kk))
        # S_kk_reg = S_kk + S_kk_diag_reg
        S_kk_reg = tf.linalg.set_diag(S_kk,tf.linalg.diag_part(S_kk) + regularizer)

        ##### 2. Calculate original energy gradient
        ## Calculate <E_{loc}>
        eloc_mean = tf.cast(tf.reduce_sum(eloc*prob, axis=0, keepdims=True),tf.complex64)

        ## Calculate <D_{W}>
        derlog_mean = tf.reduce_sum(all_derlogs*prob, axis=0, keepdims=True)

        #### Calculate <E_loc D^*_{W}>
        ed = tf.reduce_sum(tf.math.conj(all_derlogs) * eloc * prob, axis = 0, keepdims = True)

        #### Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
        grad = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)

        ### inv(S_kk) * grad == final_grads or S_kk * final_grads == grad
        final_grads = tf.linalg.solve(S_kk_reg, tf.transpose(grad))

        ### make the norm of wfs more stable:<\psi+\delta \psi | \psi+\delta \psi>=<\psi | \psi>
        Ai = tf.math.real(tf.einsum('k, k, ki -> i',tf.math.conj(tf.squeeze(wfs)),tf.squeeze(wfs),all_derlogs))
        Ai = Ai / tf.norm(Ai) # unit vector
        Ai = tf.expand_dims(Ai,axis=-1)
        Ai = tf.cast(Ai, tf.complex64)
        final_grads = final_grads - tf.reduce_sum(Ai*final_grads) * Ai

        grads = []
        prev = 0
        for old_shape in old_shapes:
            final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]   
            prev += tf.reduce_prod(old_shape[1:])
            grads.append(tf.reshape(final_grad, old_shape[1:]))

        return grads