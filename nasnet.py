import tensorflow as tf

def net(images, weights_path='./NasNet_pretrain/NASNet-large-no-top.h5'):
    images = tf.keras.applications.nasnet.preprocess_input(images)
    model = tf.keras.applications.nasnet.NASNetLarge(
        weights=weights_path,
        input_tensor=images,
        include_top=False
    )

    result = model.layers[-1].output

    return result
