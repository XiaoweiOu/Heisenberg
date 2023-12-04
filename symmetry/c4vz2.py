import tensorflow as tf
from symmetry import Symmetry

class C4vZ2(Symmetry):
    def __init__(self,length):
        self.order = 8
        self.length = length

    def symmetrize_config(self, x):
        """
            Transform a few configurations under c4v group.
            (excluding identity operation)
            Args:
                x: configurations, dim = (num_samples, num_particles)
                idx: index of the group operation (1-7 for c4v group)
        """
        ## 4D image shape [batch, height, width, channels]
        x_reshape = tf.reshape(x, [-1, self.length, self.length])
        x_reshape = tf.expand_dims(x_reshape,-1)

        x_transform = x_reshape
        x_transform = tf.concat([x_transform,tf.image.rot90(x_reshape,k=1)],0)
        x_transform = tf.concat([x_transform,tf.image.rot90(x_reshape,k=2)],0)
        x_transform = tf.concat([x_transform,tf.image.rot90(x_reshape,k=3)],0)
        x_transform = tf.concat([x_transform,tf.image.flip_left_right(x_reshape)],0)
        x_transform = tf.concat([x_transform,tf.image.flip_up_down(x_reshape)],0)
        x_transform = tf.concat([x_transform,tf.image.rot90(tf.image.flip_left_right(x_reshape))],0)
        x_transform = tf.concat([x_transform,tf.image.rot90(tf.image.flip_up_down(x_reshape))],0)
        
        ## Z2 spin flip
        x_transform = tf.concat([x_transform,(-1)*x_transform],0)

        x_transform = tf.squeeze(x_transform, -1)
        x_transform = tf.reshape(x_transform, [-1, self.length*self.length])
        return x_transform