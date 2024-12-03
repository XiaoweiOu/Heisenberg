import tensorflow as tf
import numpy as np
def get_gradient(derlogs, sample_size, eloc):
        """
            Calculate the gradient of E[\Psi] defined as 
            $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
            where D_W is the gradient of the neural network w.r.t to its output defined as
            $D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
            
            Args:
                derlogs: $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
                sample_size: the sample size
                eloc: the local energy E_{loc}
        """
        ##### 1. Calculate precondition matrix
        old_shapes = [derlog.shape for derlog in derlogs]

        ## Calculate O_k
        all_derlogs = tf.concat([tf.reshape(derlog, (sample_size, -1)) for derlog in derlogs], 1)

        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        ## Calculate <D_{W}>
        derlog_mean = tf.reduce_mean(all_derlogs, axis=0, keepdims=True)

        ## Calculate <E_loc D^*_{W}>
        ed = tf.reduce_mean(tf.math.conj(all_derlogs) * eloc, axis = 0, keepdims = True)

        ## Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
        final_grads = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)
        final_grads = tf.transpose(final_grads)

        grad_norm = [np.real(tf.norm(final_grads).numpy())]

        grads = []
        prev = 0

        for old_shape in old_shapes:
            final_grad = final_grads[prev:prev+tf.reduce_prod(old_shape[1:])]
            prev += tf.reduce_prod(old_shape[1:])
            grads.append(tf.reshape(final_grad, old_shape[1:]))
            
        return grads, grad_norm


def _test_get_gradient_distinct_config(derlogs, sample_size, eloc, prob):
        """
            Calculate the gradient of E[\Psi] defined as 
            $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
            where D_W is the gradient of the neural network w.r.t to its output defined as
            $D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
            
            Note: all mean values need to use |psi(x)|^2 as the probability distribution.

            Args:
                derlogs: $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = dlog(Psi(x))/dW$
                sample_size: the sample size
                eloc: the local energy E_{loc}
        """
        ## Calculate <E_{loc}>
        eloc_mean = tf.cast(tf.reduce_sum(eloc*prob, axis=0, keepdims=True),tf.complex64)

        grads = []
        for ii, derlog in enumerate(derlogs):
            old_shape = derlog.shape
            derlog = tf.reshape(derlog, (sample_size, -1))

            ## Calculate <D_{W}>
            derlog_mean = tf.reduce_sum(derlog*prob, axis=0, keepdims=True)

            #### Calculate <E_loc D^*_{W}>
            ed = tf.reduce_sum(tf.math.conj(derlog) * eloc * prob, axis = 0, keepdims = True)

            #### Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
            grad = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)
        
            grads.append(tf.reshape(grad, old_shape[1:]))
       
        return grads