import tensorflow as tf
from numpy.random import seed
seed(1)

def att_img(input, weights):
    W_hid = weights[0]
    b_hid = weights[1]
    W_out = weights[2]
    b_out = weights[3]
    hidden = tf.nn.relu(tf.matmul(input, W_hid) + b_hid)
    output = tf.nn.relu(tf.matmul(hidden,W_out) + b_out)
    return output

def weight_distribute(img, att, weights):
    gen_img = att_img(att, weights)
    hidden = tf.matmul(gen_img, tf.transpose(img))
    value = tf.nn.softmax(hidden, 1)
    output = tf.matmul(value, img)
    return output

def contrastive_loss(x1, x2, y, margin):
    """
    :param x1: first input
    :param x2: second input
    :param y:  if x1 and x2 are a pair, y =1, otherwise, y=0
    :param margin: a constant value
    :return:
    """
    distance = tf.reduce_mean(tf.square(x1-x2))

    similarity = y * distance
    dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    return tf.reduce_mean(dissimilarity + similarity) / 2