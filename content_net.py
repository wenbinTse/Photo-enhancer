import tensorflow as tf
import inception_resnet

slim = tf.contrib.slim

def net(images):
    with slim.arg_scope(inception_resnet.vgg_arg_scope()):
        logits, _ = inception_resnet.inception_resnet_v2(
            images,
            is_training=False
        )
    return logits
