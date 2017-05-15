import tensorflow as tf
import math


def batch_norm(x, n_out, phase_train = tf.constant(True), scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv2d(input, name, order, conv_kernel_size, conv_stride, pool_kernel_size, pool_stride, whether_pool, filterNum):
    with tf.variable_scope(name):
        # filterNum = int(math.pow(2, 5 + order))
        #print(conv_kernel_size, conv_kernel_size, input.get_shape()[-1], filterNum)
        Wconv1 = tf.get_variable("Wconv" + str(order),
                                 shape=[conv_kernel_size, conv_kernel_size, input.get_shape()[-1], filterNum])
        bconv1 = tf.get_variable("bconv" + str(order), shape=[filterNum])
        conv1 = tf.nn.conv2d(input, Wconv1, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + bconv1
        conv1 = batch_norm(conv1, conv1.get_shape()[3])
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
