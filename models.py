import tensorflow as tf

def new_leaky_relu(tensor, alpha=0.2):
    return tf.nn.leaky_relu(tensor, alpha)

def new_conv_layer(inputs, filters, kernel_size, stride, name, norm=True, relu=True):
    result = tf.layers.conv2d(
        inputs,
        filters,
        kernel_size,
        stride=stride,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
        bias_initializer=tf.initializers.constant(0.01),
        name=name,
        padding='SAME'
    )

    if norm:
        result = _instance_norm(result)
    if relu:
        result = new_leaky_relu(result)
    return result

def global_concat_layer(inputs, concated):
    h = tf.shape(tensor)[1]
    w = tf.shape(tensor)[2]
    concat_t = tf.squeeze(concated, [1, 2])
    dims = concat_t.get_shape()[-1]
    batch_l = tf.unstack(concat_t, axis=0)
    bs = []
    for batch in batch_l:
        batch = tf.tile(batch, [h * w])
        batch = tf.reshape(batch, [h, w, -1])
        bs.append(batch)
    concat_t = tf.stack(bs)
    concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
    tensor = tf.concat(3, [inputs, concat_t])
    return tensor

def concat_layer(inputs, concated, norm: True, relu: True):
    tensor = tf.concat(3, [inputs, concated])
    if norm:
        tensor = _instance_norm(tensor)
    if relu:
        tensor = new_leaky_relu(tensor)
    return tensor

#我觉得可以直接用反卷积
def resize_layer(inputs, scale=2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False):
    t_shape = tf.shape(inputs)
    t_size = [t_shape[1]*scale, t_shape[2]*scale]
    tensor = tf.image.resize_images(inputs, t_size, method, align_corners)
    return tensor

def add_layer(inputs, added):
    tensor = tf.add(inputs, added)
    return tensor

def u_net(input_image):
    with tf.variable_scope('generator'):
        with tf.variable_scope('net1'):
            l1 = new_conv_layer(input_image, 16, 3, 1, 'conv1')
            l2 = new_conv_layer(l1, 32, 5, 2, 'conv2')
            l3 = new_conv_layer(l2, 64, 5, 2, 'conv3')
            l4 = new_conv_layer(l3, 128, 5, 2, 'conv4')
            l5 = new_conv_layer(l4, 128, 5, 2, name='conv4')
        
        # global feature
        with tf.variable_scope('net2'):
            l6 = new_conv_layer(l5, 128, 5, 2, name='conv1')
            l7 = new_conv_layer(l6, 128, 5, 2, name='conv2')
            l8 = new_conv_layer(l7, 128, 8, 1, name='conv3', norm=False)
            l9 = new_conv_layer(l8, 128, 1, 1, nam3='conv4', norm=False, relu=False)
    
        with tf.variable_scope('net3'):
            l10 = new_conv_layer(l5, 128, 3, 1, name='conv1')
            l10_concat = global_concat_layer(l10, l9)

            l11 = new_conv_layer(l10_concat, 128, 1, 1, name='conv2')
            
            l12 = new_conv_layer(l11, 128, 3, 1, name='conv3', norm=False, relu=False)
            l12_resize = resize_layer(l12)
            l12_concat = concat_layer(l12_resize, l4)

            l13 = new_conv_layer(l12_concat, 128, 3, 1, name='conv4', norm=False, relu=False)
            l13_resize = resize_layer(l13)
            l13_concat = concat_layer(l13_resize, l3)

            l14 = new_conv_layer(l13_concat, 64, 3, 1, name='conv5', norm=False, relu=False)
            l14_resize = resize_layer(l14)
            l14_concat = concat_layer(l14_resize, l2)

            l15 = new_conv_layer(l14_concat, 32, 3, 1, name='conv6', norm=False, relu=False)
            l15_resize = resize_layer(l15)
            l15_concat = concat_layer(l15_resize, l1)

            l16 = new_conv_layer(l15_concat, 16, 3, 1, name='conv7')
            l17 = new_conv_layer(l16, 3, 3, 1, name='conv8', norm=False, relu=False)

            l18 = add_layer(l17, input_image)

            return l18

# def resnet(input_image):

#     with tf.variable_scope("generator"):

#         W1 = weight_variable([9, 9, 3, 64], name="W1"); b1 = bias_variable([64], name="b1");
#         c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

#         # residual 1

#         W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2");
#         c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

#         W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3");
#         c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

#         # residual 2

#         W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4");
#         c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

#         W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5");
#         c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

#         # residual 3

#         W6 = weight_variable([3, 3, 64, 64], name="W6"); b6 = bias_variable([64], name="b6");
#         c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

#         W7 = weight_variable([3, 3, 64, 64], name="W7"); b7 = bias_variable([64], name="b7");
#         c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

#         # residual 4

#         W8 = weight_variable([3, 3, 64, 64], name="W8"); b8 = bias_variable([64], name="b8");
#         c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

#         W9 = weight_variable([3, 3, 64, 64], name="W9"); b9 = bias_variable([64], name="b9");
#         c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

#         # Convolutional

#         W10 = weight_variable([3, 3, 64, 64], name="W10"); b10 = bias_variable([64], name="b10");
#         c10 = tf.nn.relu(conv2d(c9, W10) + b10)

#         W11 = weight_variable([3, 3, 64, 64], name="W11"); b11 = bias_variable([64], name="b11");
#         c11 = tf.nn.relu(conv2d(c10, W11) + b11)

#         # Final

#         W12 = weight_variable([9, 9, 64, 3], name="W12"); b12 = bias_variable([3], name="b12");
#         enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

#     return enhanced

def adversarial(image_):

    with tf.variable_scope("discriminator"):

        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
