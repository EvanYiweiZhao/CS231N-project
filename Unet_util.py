import tensorflow as tf
import math


def conv2d(input, name, order, conv_kernel_size, conv_stride, pool_kernel_size, pool_stride, whether_pool, filterNum):
    with tf.variable_scope(name):
        # filterNum = int(math.pow(2, 5 + order))
        #print(conv_kernel_size, conv_kernel_size, input.get_shape()[-1], filterNum)
        Wconv1 = tf.get_variable("Wconv" + str(order),
                                 shape=[conv_kernel_size, conv_kernel_size, input.get_shape()[-1], filterNum])
        bconv1 = tf.get_variable("bconv" + str(order), shape=[filterNum])
        conv1 = tf.nn.conv2d(input, Wconv1, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + bconv1
        act1 = tf.nn.relu(conv1)

        Wconv2 = tf.get_variable("Wconv" + str(order + 1),
                                 shape=[conv_kernel_size, conv_kernel_size, filterNum, filterNum])
        bconv2 = tf.get_variable("bconv" + str(order + 1), shape=[filterNum])
        conv2 = tf.nn.conv2d(act1, Wconv2, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + bconv2
        act2 = tf.nn.relu(conv2)

        if whether_pool:
            maxPool1 = tf.nn.max_pool(act2, (1, pool_kernel_size, pool_kernel_size, 1),
                                               (1, pool_stride, pool_stride, 1), padding='SAME')
            return (maxPool1, Wconv1, bconv1, Wconv2, bconv2)
        else:
            return (act2, Wconv1, bconv1, Wconv2, bconv2)


def deconv2d(input, concat_input, name, order, conv_kernel_size, conv_stride, deconv_kernel_size, deconv_stride,
             output, filterNum, whether_conv):
    with tf.variable_scope(name):
        #filterNum = int(math.pow(2, 10 - order))
        Wdeconv1 = tf.get_variable("Wdeconv" + str(order),
                                   shape=[deconv_kernel_size, deconv_kernel_size, filterNum, input.get_shape()[-1]])
        bdeconv1 = tf.get_variable("bdeconv" + str(order), shape=[filterNum])
        #print(output)
        deconv1 = tf.nn.conv2d_transpose(input, Wdeconv1, output_shape = output, strides=[1, deconv_stride, deconv_stride, 1]) + bdeconv1

        if whether_conv:
            print("concatenate", deconv1.get_shape(), concat_input.get_shape())
            concat_output = tf.concat([deconv1, concat_input], axis = 3)
            print("fuck", concat_output.get_shape())

            tmp_output, Wconv1, bconv1, Wconv2, bconv2 = conv2d(concat_output, 'layer' + str(order), 10 + order,
                                                            conv_kernel_size, conv_stride, 0, 0, False, filterNum)

            return (tmp_output, Wconv1, bconv1, Wconv2, bconv2, Wdeconv1, bdeconv1)

        else:
            return (deconv1, Wdeconv1, bdeconv1)