import tensorflow as tf
import inception_resnet

slim = tf.contrib.slim

def net(images):
    with slim.arg_scope(inception_resnet.resnet_arg_scope()):
        logits, _ = inception_resnet.resnet_v2_101(
            images,
            is_training=False
        )
    return logits
