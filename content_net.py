import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

def net(images):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(
            images,
            reuse=tf.AUTO_REUSE,
            is_training=False
        )
    return logits
