import tensorflow as tf
import numpy as np
import cv2
from Unet_util import *


class Config():
    lr = 0.01
    batch_size = 128
    conv_kernel_size = 3
    conv_stride = 1
    deconv_kernel_size = 3
    deconv_stride = 2
    pool_kernel_size = 2
    pool_stride = 2
    l2_lambda = 0.0000001
    color_num = 3



class UNetModel():
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.float32, (Config.batch_size, 256, 256, 3))
        self.outputs_placeholder = tf.placeholder(tf.float32, (Config.batch_size, 256, 256, 3))

    def create_feed_dict(self, inputs_batch, outputs_batch):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.outputs_placeholder: outputs_batch,
        }
        return feed_dict

    def add_prediction_op(self):
        X = self.inputs_placeholder

        inputX = X
        cacheList = [None for _ in range(6)]
        for i in range(1, 6):
            filterNum = int(math.pow(2, 5 + i))
            cacheList[i] = conv2d(inputX, "layer" + str(i), i, Config.conv_kernel_size, Config.conv_stride,
                                  Config.pool_kernel_size, Config.pool_stride, True, filterNum)
            inputX = cacheList[i][0]
        '''filterNum = int(math.pow(2, 5 + 5))
        cacheList[5] = conv2d(inputX, "layer" + str(5), 5, Config.conv_kernel_size, Config.conv_stride,
                              Config.pool_kernel_size, Config.pool_stride, False, filterNum)
        inputX = cacheList[5][0]'''

        for i in range(1,6):
            print(cacheList[i][0].get_shape())
        deCacheList = [None for _ in range(6)]
        for i in range(1, 5):
            #deconv_shape =tf.pack([cacheList[5 - i][0].get_shape()[_] for _ in range(4)])
            #print(deconv_shape)
            deCacheList[i] = deconv2d(inputX, cacheList[5 - i][0], "delayer" + str(i), i, Config.conv_kernel_size,
                                      Config.conv_stride, Config.deconv_kernel_size, Config.deconv_stride,
                                      tf.shape(cacheList[5 - i][0]), int(math.pow(2, 10 - i)), True)
            inputX = deCacheList[i][0]

        deCacheList[5] = deconv2d(inputX, None, "delayer" + str(5), 5, Config.conv_kernel_size, Config.conv_stride,
                                  Config.deconv_kernel_size, Config.deconv_stride, tf.shape(self.outputs_placeholder),
                                  Config.color_num, False)

        self.result = tf.nn.tanh(deCacheList[5][0])
        print('finally', self.result.get_shape(), self.outputs_placeholder.get_shape())

    def add_loss_op(self):

        cost = tf.reduce_sum(tf.nn.l2_loss(self.outputs_placeholder - self.result))

        tv = tf.trainable_variables()
        l2_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if 'bconv' not in v.name])

        #print(cost.get_shape(), l2_cost.get_shape())
        self.loss = Config.l2_lambda * l2_cost + cost

    def add_training_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()


    def train_on_batch(self, session, train_inputs_batch, train_outputs_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_outputs_batch)
        batch_cost = session.run([self.loss], feed)
        if math.isnan(batch_cost): # basically all examples in this batch have been skipped
            return 0
        if train:
            _ = session.run([self.optimizer], feed)
        return batch_cost

    def __init__(self):
        self.build()


if __name__ == "__main__":
    with tf.Graph().as_default():
        fuck = UNetModel()
