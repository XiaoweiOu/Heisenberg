import tensorflow as tf
from tensorflow.keras import layers

def transform(x, length, idx):
    """
        Transform a few configurations under c4v group.
        Args:
            x: configurations, dim = (num_samples, num_particles)
             idx: index of the group operation (1-7 for c4v group)
    """
    ## 4D image shape [batch, height, width, channels]
    x_reshape = tf.reshape(x, [-1, length, length])
    x_reshape = tf.expand_dims(x_reshape,-1)

    if idx==0:
        x_transform = tf.image.rot90(x_reshape)
    elif idx==1:
        x_transform = tf.image.rot90(x_reshape,k=2)
    elif idx==2:
        x_transform = tf.image.rot90(x_reshape,k=3)
    elif idx==3:
        x_transform = tf.image.flip_left_right(x_reshape)
    elif idx==4:
        x_transform = tf.image.flip_up_down(x_reshape)
    elif idx==5:
        x_transform = tf.image.flip_left_right(x_reshape)
        x_transform = tf.image.rot90(x_transform)
    elif idx==6:
        x_transform = tf.image.flip_up_down(x_reshape)
        x_transform = tf.image.rot90(x_transform)
        
    x_transform = tf.squeeze(x_transform, axis=-1)
    x_transform = tf.reshape(x_transform, [-1, length*length])
    return x_transform