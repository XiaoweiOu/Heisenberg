import tensorflow as tf
def get_gradient_sr(derlogs, sample_size, eloc):
        """
            Calculate the gradient of E[\Psi] using the stochastic reconfiguration
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

        #### Calculate <E_loc D^*_{W}>
        ed = tf.reduce_mean(tf.math.conj(all_derlogs) * eloc, axis = 0, keepdims = True)

        #### Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
        grad = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)

        ### inv(S_kk) * grad == final_grads or S_kk * final_grads == grad
        final_grads = tf.linalg.solve(S_kk_reg, tf.transpose(grad))

        grads = []
        prev = 0
        for old_shape in old_shapes:
            final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]   
            prev += tf.reduce_prod(old_shape[1:])
            grads.append(tf.reshape(final_grad, old_shape[1:]))

        return grads