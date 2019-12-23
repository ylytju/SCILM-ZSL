import tensorflow as tf
from config import *
from model import *
from utils import *
import argparse
from numpy.random import seed
import os
from dataset import LoadDataset

seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = argparse.ArgumentParser('Run the code for CYB')
    parser.add_argument('--iter_num', type=int,default=1000, help='the iteration number')
    parser.add_argument('--hid_dim', type=int, default=1600, help='the hidden dimension')
    parser.add_argument('--data_dir', type=str, default='../data/CUB', help='the train directory')
    parser.add_argument('--selected_num',type=int,default=5, help='the selected number for each class')
    parser.add_argument('--lr', type=float32, default=1e-4, help='the learning rate')
    parser.add_argument('--lambda_p', type=float32, default=0.1, help='the parameter one')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--lambda_q', type=float32, default=1e-5, help='the balance parameter, default: 2e-4')
    parser.add_argument('--manualSeed', type=int, default=1000, help='the random seed')
    args = parser.parse_args()
    return args

class Model(object):
    def __init__(self, args):
        self.iter_num = args.iter_num
        self.data_dir = args.data_dir
        self.hid_dim = args.hid_dim
        self.lambda_p = args.lambda_p
        self.lambda_q = args.lambda_q
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.selected_num = args.selected_num

    def create_model(self):
        self.train_data, self.test_data, self.test_seen_data, self.test_unseen_data = load_data(self.data_dir)

        self.img_dim = self.train_data['img_fea'].shape[1]
        self.att_dim = self.train_data['att_fea'].shape[1]

        # define the placeholder
        self.att_pl = tf.placeholder(tf.float32, [None, self.att_dim])
        self.img_pl = tf.placeholder(tf.float32, [None, self.img_dim])
        self.ave_img_pl = tf.placeholder(tf.float32, [None, self.img_dim])
        self.ave_att_pl = tf.placeholder(tf.float32, [None, self.att_dim])
        self.con_att_pl = tf.placeholder(tf.float32, [None, self.att_dim])
        self.con_img_pl = tf.placeholder(tf.float32, [None, self.img_dim])
        self.lr_pl = tf.placeholder(tf.float32)

        similarity = tf.placeholder(tf.float32, [None, 1], name='similarity')
        self.similarity_float = tf.to_float(similarity)

        ## Parameters
        # att_img parameters
        with tf.variable_scope('att_image') as scope:
            W_hid = self.weight_variable([self.att_dim, self.hid_dim])
            b_hid = self.bias_variable([self.hid_dim])
            W_out = self.weight_variable([self.hid_dim, self.img_dim])
            b_out = self.bias_variable([self.img_dim])

            self.Weights_encoder = [W_hid, b_hid, W_out, b_out]
            self.gen_img = self.att_img(self.att_pl, self.Weights_encoder)
            scope.reuse_variables()
            self.gen_con_img = self.att_img(self.con_att_pl, self.Weights_encoder)

        with tf.name_scope('loss'):
            margin = 2
            hinge_loss = contrastive_loss(self.gen_con_img, self.con_img_pl, self.similarity_float, margin)
            loss_lse = tf.reduce_mean(tf.square(self.gen_img - self.img_pl))
            vars = tf.trainable_variables()
            regularisers = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        self.loss = loss_lse + self.lambda_p*hinge_loss + self.lambda_q*regularisers

        # # Network
        self.optimizer = tf.train.AdamOptimizer(self.lr_pl).minimize(self.loss)

    def train(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # # Run
        for i in range(self.iter_num+1):
            att_batch, img_batch, lab_batch = data_iterator(self.sess, self.train_data, self.selected_num,
                                                            self.ave_img_pl, self.ave_att_pl, self.Weights_encoder)
            dataset = LoadDataset(att_batch, img_batch, lab_batch)
            next_batch = dataset.get_batch
            con_att_batch, con_img_batch, con_lab_batch = next_batch(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.att_pl: att_batch, self.img_pl: img_batch, self.con_img_pl: con_img_batch,
                                                    self.con_att_pl: con_att_batch,self.similarity_float: con_lab_batch, self.lr_pl: self.lr})
            if i >= 2000:
                self.lr = 1e-5
            if i % 50 == 0:
                print('The iter is %d' % i)
                self.test()

    def test(self):
        unseen_pro = self.test_data['te_pro']
        att_pre = self.sess.run(self.gen_img, feed_dict={self.att_pl: unseen_pro})
        acc_zsl = compute_accuracy(att_pre, self.test_data)
        all_pro = self.test_seen_data['te_pro']
        att_pre = self.sess.run(self.gen_img, feed_dict={self.att_pl: all_pro})
        acc_seen_gzsl = compute_accuracy(att_pre, self.test_seen_data)
        acc_unseen_gzsl = compute_accuracy(att_pre, self.test_unseen_data)

        H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
        print('zsl:', acc_zsl)
        print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_seen_gzsl, acc_unseen_gzsl, H))

    def att_img(self, input, weights):
        W_hid = weights[0]
        b_hid = weights[1]
        W_out = weights[2]
        b_out = weights[3]
        hidden = tf.nn.tanh(tf.matmul(input, W_hid) + b_hid)
        output = tf.nn.relu(tf.matmul(hidden, W_out) + b_out)
        return output

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=1e-3)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(1e-3, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def Dense(input, weights, bias, activate_fc=None):
        output = tf.matmul(input, weights) + bias
        if activate_fc:
            output = activate_fc(output)
        return output


def main():
    args = parse_args()
    if args is None:
        exit()
    random.seed(args.manualSeed)
    tf.set_random_seed(args.manualSeed)
    model = Model(args)
    model.create_model()
    model.train()
    print("Training finished!")
    model.test()
    print("Test finished!")

if __name__ == '__main__':
    main()
