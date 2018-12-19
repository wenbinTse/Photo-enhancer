# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
from scipy import misc
import numpy as np
import sys
import os

from load_dataset import load_test_data, load_batch, postprocess
from ssim import MultiScaleSSIM
import models
import utils
import vgg

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step, last_step = utils.process_command_args(sys.argv)

np.random.seed(666)
tf.set_random_seed(666)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
#wb: load_batch并不是load batch，而是获取所有train data
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:

    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    training = tf.placeholder(tf.bool)
    global_step = tf.Variable(0)

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    # get processed enhanced image

    # enhanced = models.resnet(phone_image)
    enhanced = models.u_net(phone_image, training)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    # # push randomly the enhanced or dslr image to an adversarial CNN-discriminator
    #
    # adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    # adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    # 之前是随机选取，然后混合判断，现在采用跟DCGAN一样的策略，分开判断，再使用交叉熵

    logits_phone, probs_phone = models.adversarial(phone_image)
    logits_dslr, probs_dslr = models.adversarial(dslr_image)
    logits_enhanced, _ = models.adversarial(enhanced)

    # losses
    # 1) texture (adversarial) loss

    # discrim_target = tf.concat([adv_, 1 - adv_], 1)
    #
    # loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    # loss_texture = -loss_discrim
    #
    # correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    # discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    d_loss_real = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_dslr, tf.ones_like(logits_dslr)))
    d_loss_fake = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_phone, tf.zeros_like(logits_phone)))
    loss_discrim = d_loss_fake + d_loss_real

    half = 0.5
    phone_accuracy = tf.reduce_mean(tf.cast(tf.less_equal(probs_phone, half), tf.float32))
    dslr_accuracy = tf.reduce_sum(tf.cast(tf.greater(probs_dslr, half), tf.float32))
    discim_accuracy = (phone_accuracy + dslr_accuracy) / 2

    loss_texture = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_dslr, logits_enhanced))

    # 2) content loss

    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, enhanced)
    dslr_vgg = vgg.net(vgg_dir, dslr_image)

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

    learning_rate = tf.train.exponential_decay(5e-4, global_step, decay_steps=1000, decay_rate=0.88,
                                               staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_gen = tf.train.AdamOptimizer(learning_rate)\
            .minimize(loss_generator, var_list=generator_vars, global_step=global_step)
    train_step_disc = tf.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    if last_step != -1:
        print('Loading model')
        saver.restore(sess, 'models/' + str(phone) + '_iteration_' + str(last_step) + '.ckpt')

    print('Training network')

    train_loss_gen = 0.0
    train_acc_discrim = 0.0

    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()

    #################################################################
    # summary
    _ = tf.summary.scalar('d_loss_real/train', tensor=d_loss_real, collections=['train'])
    _ = tf.summary.scalar('d_loss_fake/train', tensor=d_loss_fake, collections=['train'])
    _ = tf.summary.scalar('loss_discrim/train', tensor=loss_discrim, collections=['train'])
    _ = tf.summary.scalar('phone_accuracy/train', tensor=phone_accuracy, collections=['train'])
    _ = tf.summary.scalar('dslr_accuracy/train', tensor=dslr_accuracy, collections=['train'])
    _ = tf.summary.scalar('loss_texture/train', tensor=loss_texture, collections=['train'])
    _ = tf.summary.scalar('loss_content/train', tensor=loss_content, collections=['train'])
    _ = tf.summary.scalar('loss_color/train', tensor=loss_color, collections=['train'])
    _ = tf.summary.scalar('loss_generator/train', tensor=loss_generator, collections=['train'])
    _ = tf.summary.scalar('learning_rate/train', tensor=learning_rate, collections=['train'])

    _ = tf.summary.scalar('d_loss_real/val', tensor=d_loss_real, collections=['val'])
    _ = tf.summary.scalar('d_loss_fake/val', tensor=d_loss_fake, collections=['val'])
    _ = tf.summary.scalar('loss_discrim/val', tensor=loss_discrim, collections=['val'])
    _ = tf.summary.scalar('phone_accuracy/val', tensor=phone_accuracy, collections=['val'])
    _ = tf.summary.scalar('dslr_accuracy/val', tensor=dslr_accuracy, collections=['val'])
    _ = tf.summary.scalar('loss_texture/val', tensor=loss_texture, collections=['val'])
    _ = tf.summary.scalar('loss_content/val', tensor=loss_content, collections=['val'])
    _ = tf.summary.scalar('loss_color/val', tensor=loss_color, collections=['val'])
    _ = tf.summary.scalar('loss_generator/val', tensor=loss_generator, collections=['val'])

    summaries_op = tf.summary.merge_all('train')
    summaries_val_op = tf.summary.merge_all('val')

    folder_summary = 'summary'
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)
    #################################################################

    for i in range(last_step, num_train_iters + last_step + 1):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp, learning_rate_tmp] = sess.run([loss_generator, train_step_gen, learning_rate],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, training: True})
        train_loss_gen += loss_temp / eval_step

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [accuracy_temp, temp, summaries_val] = sess.run([discim_accuracy, train_step_disc, summaries_op],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, training: True})
        summary_writer.add_summary(summaries_val, i)

        train_acc_discrim += accuracy_temp / eval_step

        if i % eval_step == 0:

            print('learning rate: ', learning_rate_tmp)

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(num_test_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, accuracy_disc, summaries_val, losses] =\
                    sess.run([enhanced, discim_accuracy, summaries_val_op,
                                [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr]],
                                feed_dict={phone_: phone_images, dslr_: dslr_images, training: False})
                summary_writer.add_summary(summaries_val, j)
                test_losses_gen += np.asarray(losses) / num_test_batches
                test_accuracy_disc += accuracy_disc / num_test_batches

                loss_ssim += MultiScaleSSIM(postprocess(np.reshape(dslr_images, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]), np.float32),
                                                    postprocess(enhanced_crops, np.float32) ) / num_test_batches

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                  (i, phone, train_acc_discrim, test_accuracy_disc)

            logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                  (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                   test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], loss_ssim)

            print(logs_disc)
            print(logs_gen)

            # save the results to log file

            logs = open('models/' + phone + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, training: False})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data
            if i != 0 and i % (eval_step * 5) == 0:
                del train_data
                del train_answ
                train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
