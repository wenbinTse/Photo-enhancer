"""
支持不同尺度输入，并只加载一次模型
"""
import tensorflow as tf
from scipy import misc
import numpy as np
from models import u_net
from load_dataset import preprocess
import argparse

model_name = './pretrained_models/iphone_iteration_22000.ckpt'

inited = False
images_plh = tf.placeholder(tf.float32, [None, None, None, 3])
training_plh = tf.placeholder(tf.bool)
enhanced = u_net(images_plh, training_plh)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, model_name)

image_net_process = True

def test(file_name, result_name='result.png'):
    image = misc.imread(file_name, mode='RGB')
    assert len(image.shape) == 3, '只处理彩色照片'
    if image_net_process:
        image = preprocess(image)
    else:
        image = image / 255

    images = np.expand_dims(image, 0)
    print(images.shape)

    results = sess.run(enhanced, feed_dict={images_plh: images, training_plh: False})
    misc.imsave(result_name, results[0])
    print('result is saved to {}'.format(result_name))
    return results[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='图像增强')
    parser.add_argument('file_name', type=str, help='待处理图片路径')
    parser.add_argument('--result_name', type=str, help='结果保存路径', default='result.png')
    args = parser.parse_args()

    test(args.file_name, args.result_name)
