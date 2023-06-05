import tensorflow as tf
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
        ## Calculate <E_{loc}>
        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)

        grads = []
        for ii, derlog in enumerate(derlogs):
            old_shape = derlog.shape
            derlog = tf.reshape(derlog, (sample_size, -1))

            ## Calculate <D_{W}>
            derlog_mean = tf.reduce_mean(derlog, axis=0, keepdims=True)

            #### Calculate <E_loc D^*_{W}>
            ed = tf.reduce_mean(tf.math.conj(derlog) * eloc, axis = 0, keepdims = True)

            #### Calculate gradient $2Re[  <E_{loc}D^*_{W}> - <E_{loc}><D^*_{W}> ]$
            grad = 2 * tf.cast(tf.math.real(ed - eloc_mean * tf.math.conj(derlog_mean)), tf.complex64)
        
            grads.append(tf.reshape(grad, old_shape[1:]))
       
        return grads