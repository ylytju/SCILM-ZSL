import tensorflow as tf
import scipy.io as sio
import numpy as np
from numpy.random import seed
seed(1)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-2)  #1e-2
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def load_data(dataroot):
    image_embedding = 'res101'
    class_embedding = 'att'
    matcontent = sio.loadmat(dataroot + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + class_embedding + "_splits.mat")
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['att'].T  # 85 x 50
    x = feature[trainval_loc]    # train feature
    train_label = label[trainval_loc].astype(int)

    att = attribute[train_label]
    train_data = {
        'img_fea': x,
        'att_fea': att,
        'tr_label': train_label,
        'all_pro': attribute
    }
    x_test = feature[test_unseen_loc]
    test_label = label[test_unseen_loc].astype(int)
    x_test_seen = feature[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    test_id = np.unique(test_label)
    att_pro = attribute[test_id]

    test_data = {
        'img_fea': x_test,
        'te_label': test_label,
        'te_id': test_id,
        'te_pro': att_pro
    }
    test_seen_data = {
        'img_fea': x_test_seen,
        'te_label': test_label_seen,
        'te_pro': attribute,
        'te_id': np.arange(50)
    }
    test_unseen_data = {
        'img_fea': x_test,
        'te_label': test_label,
        'te_id': np.arange(50),
        'te_pro': attribute
    }
    return train_data, test_data, test_seen_data, test_unseen_data