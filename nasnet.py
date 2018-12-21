import tensorflow as tf

def net(images, weights_path='./NasNet_pretrain/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
    # images = tf.keras.applications.nasnet.preprocess_input(images)
    # model = tf.keras.applications.nasnet.NASNetLarge(
    #     weights=weights_path,
    #     input_tensor=images,
    #     include_top=False
    # )

    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        weights=weights_path,
        input_tensor=tf.keras.applications.inception_resnet_v2.preprocess_input(images),
        include_top=False,
        pooling='max'
    )

    for layer in model.layers:
        layer.trainable = False

    result = model.layers[-1].output

    return result
