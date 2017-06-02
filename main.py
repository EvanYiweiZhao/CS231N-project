import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import math
from random import randint
import random

from utils import *
from Unet_util import *
from uNet import *
import time
import datetime

clamp_lower = -0.01
clamp_upper = 0.01
lam = 10

class Color():
    def __init__(self, imgsize=256, batchsize=4, mode='gp'):
        self.time  = time.time()
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = imgsize
        self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.input_colors2 = 3
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        # combined_preimage = self.line_images
        with tf.variable_scope('generator'):
            self.generated_images = self.generator(combined_preimage)

        self.real_AB = tf.concat([combined_preimage, self.real_images], 3)
        self.fake_AB = tf.concat([combined_preimage, self.generated_images], 3)

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss = tf.reduce_mean(disc_fake_logits - disc_true_logits) # WGAN
        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake

        if mode is 'gp':
            alpha_dist = tf.contrib.distributions.Uniform(0.0, 1.0)
            alpha = alpha_dist.sample((self.batch_size, 1, 1, 1))
            interpolated = self.real_images + alpha*(self.generated_images-self.real_images)
            interpolated_AB = tf.concat([combined_preimage, interpolated], 3)
            _, inte_logit = self.discriminator(interpolated_AB, reuse=True)
            gradients = tf.gradients(inte_logit, [interpolated_AB,])[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
            gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
            gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
            grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
            self.d_loss += lam*gradient_penalty

        self.g_loss = tf.reduce_mean(-disc_fake_logits) \
                       + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        d_loss_sum = tf.summary.scalar("c_loss", self.d_loss)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        if mode is 'regular':
            clipped_var_d = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in self.d_vars]
            # merge the clip operations on critic variables
            with tf.control_dependencies([self.d_optim]):
                self.d_optim = tf.tuple(clipped_var_d)
 
        if not mode in ['gp', 'regular']:
            raise(NotImplementedError('Only two modes'))


    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d2(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d2(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d2(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d2(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        X = img_in

        inputX = X
        cacheList = [None for _ in range(6)]
        for i in range(1, 6):
            filterNum = int(math.pow(2, 5 + i))
            cacheList[i] = conv2d(inputX, "g_layer" + str(i), i, Config.conv_kernel_size, Config.conv_stride,
                                  Config.pool_kernel_size, Config.pool_stride, True, filterNum)
            inputX = cacheList[i][0]

        for i in range(1,6):
            print(cacheList[i][0].get_shape())
        deCacheList = [None for _ in range(6)]
        for i in range(1, 5):
            #deconv_shape =tf.pack([cacheList[5 - i][0].get_shape()[_] for _ in range(4)])
            #print(deconv_shape)
            deCacheList[i] = deconv2d(inputX, cacheList[5 - i][0], "g_delayer" + str(i), i, Config.conv_kernel_size,
                                      Config.conv_stride, Config.deconv_kernel_size, Config.deconv_stride,
                                      tf.shape(cacheList[5 - i][0]), int(math.pow(2, 10 - i)), True)
            inputX = deCacheList[i][0]

        deCacheList[5] = deconv2d(inputX, None, "g_delayer" + str(5), 5, Config.conv_kernel_size, Config.conv_stride,
                                  Config.deconv_kernel_size, Config.deconv_stride, tf.shape(self.real_images),
                                  self.real_images.get_shape()[3], False)#Config.color_num+1, False)

        return tf.nn.tanh(deCacheList[5][0])
        '''s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)


        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = bn(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = bn(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)'''




    def imageblur(self, cimg, sampling=False):
        if sampling:
            cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
        else:
            for i in xrange(30):
                randx = randint(0,205)
                randy = randint(0,205)
                cimg[randx:randx+50, randy:randy+50] = 255
        return cv2.blur(cimg,(100,100))

    def train(self):
        self.loadmodel()
        #data = glob(os.path.join("img", "*.jpg"))
        data = glob(os.path.join("/commuter/chatbot/ersanyi/deepcolor/imgs1000", "*.jpg"))
        print data[0]
        val_data = glob(os.path.join("val","*.jpg"))
        
        base = np.array([get_image(sample_file) for sample_file in data[0:self.batch_size]])
        base_normalized = base/255.0

        base_edge = np.array([edge_detection(ba) for ba in base]) / 255.0
        base_edge = np.expand_dims(base_edge, 3)

        base_colors = np.array([self.imageblur(ba) for ba in base]) / 255.0

        val = np.array([get_image(sample_file) for sample_file in val_data[0:self.batch_size]])
        val_normalized = val/255.0

        val_edge = np.array([edge_detection(ba) for ba in val]) / 255.0
        val_edge = np.expand_dims(val_edge, 3)

        val_colors = np.array([self.imageblur(ba) for ba in val]) / 255.0

        ims("results/base.png",merge_color(base_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_line.jpg",merge(base_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_colors.jpg",merge_color(base_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))


        ims("fourthResults/val.jpg",merge_color(val_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("fourthResults/val_line.jpg",merge(val_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("fourthResults/val_colors.jpg",merge_color(val_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        def next_feed_dict():
            #batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
            batch_files = [data[j] for j in random.sample(xrange(datalen), 4) ]
            batch = np.array([get_image(batch_file) for batch_file in batch_files])
            batch_normalized = batch/255.0

            batch_edge = np.array([edge_detection(ba) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            batch_colors = np.array([self.imageblur(ba) for ba in batch]) / 255.0
            
            feed_dict = {self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors}
            return feed_dict


        with tf.device("/gpu:0"):
            log_dir = './log/'
            summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            for t in xrange(20000):
                d_iters = 5
                if t % 500 == 0 or t < 25:
                    d_iters = 100
                for j in range(d_iters):
                    feed_dict = next_feed_dict()
                    if t % 100 == 99 and j == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        d_loss, _, merged = self.sess.run([self.d_loss, self.d_optim, self.merged_all], feed_dict=feed_dict,
                                             options=run_options, run_metadata=run_metadata)
                        summary_writer.add_summary(merged, t)
                        summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(t), t)
                    else:
                        d_loss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict=feed_dict)   

                feed_dict = next_feed_dict()
                if t % 100 == 99:
                    g_loss, _, merged = self.sess.run([self.g_loss, self.g_optim, self.merged_all], feed_dict=feed_dict,
                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, t)
                    summary_writer.add_run_metadata(
                        run_metadata, 'generator_metadata {}'.format(t), t)
                else:
                    g_loss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict=feed_dict)

                print "%d: [%d] d_loss %f, g_loss %f" % (t, (datalen/self.batch_size), d_loss, g_loss)


                if t % 500 == 499:
                    self.save("./checkpoint", t)

                if t % 200 == 0:
                    recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: val_normalized, self.line_images: val_edge, self.color_images: val_colors})
                    ims("fourthResults/"+str(e) + 'turn' + str(i)+ ".jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                #start validate

                print('total time '+ str(datetime.timedelta(seconds=(time.time()-self.time))))
                print('average time '+ str(datetime.timedelta(seconds=((time.time()-self.time)/(e+1)))))

    def test(self):
        self.loadmodel()
        with tf.device("/gpu:0"):
            val_data = glob(os.path.join("val","*.jpg"))
            val = np.array([get_image(sample_file) for sample_file in val_data[0:self.batch_size]])
            val_normalized = val/255.0

            val_edge = np.array([edge_detection(ba) for ba in val]) / 255.0
            val_edge = np.expand_dims(val_edge, 3)

            #val_colors = np.array([self.imageblur(ba) for ba in val]) / 255.0
            val_colors = np.array([cv2.threshold(ba,255,255,cv2.THRESH_BINARY) for ba in val]) / 255.0

            ims("fourthResults/val.jpg",merge_color(val_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("fourthResults/val_line.jpg",merge(val_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("fourthResults/val_colors.jpg",merge_color(val_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))


            recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: val_normalized, self.line_images: val_edge, self.color_images: val_colors})
            ims("fourthResults/NoHint.jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

        #start validate

        #print('total time '+ str(datetime.timedelta(seconds=(time.time()-self.time))))
        #print('average time '+ str(datetime.timedelta(seconds=((time.time()-self.time)/(e+1)))))


    def loadmodel(self, load_discrim=True):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)
        self.merged_all = tf.summary.merge_all()

        if self.load("./checkpoint"):
            print "Loaded"
        else:
            print "Load failed"

    def sample(self):
        self.loadmodel(False)

        data = glob(os.path.join("img", "*.jpg"))

        datalen = len(data)

        for i in range(min(100,datalen / self.batch_size)):
            batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
            batch = np.array([cv2.resize(imread(batch_file), (512,512)) for batch_file in batch_files])
            batch_normalized = batch/255.0

            batch_edge = np.array([edge_detection(ba) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            batch_colors = np.array([self.imageblur(ba,True) for ba in batch]) / 255.0

            recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})
            ims("results/sample_"+str(i)+".jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_origin.jpg",merge_color(batch_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_line.jpg",merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_color.jpg",merge_color(batch_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python main.py [train, sample]"
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            c = Color()
            c.train()
        elif cmd == "sample":
            c = Color(512,1)
            c.sample()
        elif cmd == "test":
            c = Color()
            c.test()
        else:
            print "Usage: python main.py [train, sample]"
