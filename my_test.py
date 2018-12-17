"""
支持不同尺度输入，并只加载一次模型
"""
import tensorflow as tf
from scipy import misc
import numpy as np
from models import u_net

model_name = './history/u_net_remove_the_last_concat_layer/models/iphone_iteration_25000.ckpt'

inited = False
images_plh = tf.placeholder(tf.float32, [None, None, None, 3])
training_plh = tf.placeholder(tf.bool)
enhanced = u_net(images_plh, training_plh)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, model_name)

def test(file_name, result_name='result.jpg'):
    image = misc.imread(file_name) / 255
    assert len(image.shape) == 3, '只处理彩色照片'
    images = np.expand_dims(image, 0)
    print(images.shape)

    results = sess.run(enhanced, feed_dict={images_plh: images, training_plh: False})
    misc.imsave(result_name, results[0])
    print('result is saved to {}'.format(result_name))
    return results[0]
