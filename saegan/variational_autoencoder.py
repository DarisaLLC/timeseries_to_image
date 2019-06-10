import sys
sys.path.append("..")
from utils import *

class Variational_AutoEncoder_Image(BasicBlock):
    def __init__(self, len_latent, name=None):
        super(Variational_AutoEncoder_Image, self).__init__(name=name or 'VAEI')
        self.len_latent = len_latent

    def encode(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            net = lrelu(conv2d(x, 32, 4, 4, 2, 2, padding='SAME', name='c1'), name='l1')
            net = lrelu(bn(conv2d(net, 64, 4, 4, 2, 2, padding='SAME', name='c2'), is_training, name='bn1'), name='l2')
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, padding='SAME', name='c3'), is_training, name='bn2'), name='l3')
            net = tf.reshape(net, [-1, 4*4*128])
            net = lrelu(bn(dense(net, 256, name='fc1'), is_training, name='bn3'), name='l4')

            mean_code = dense(net, self.len_latent, name='fc_mean')
            std_code = dense(net, self.len_latent, name='fc_std')

        return mean_code, std_code
    
    def decode(self, z, mean_code, std_code, noised_z=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            if noised_z is None:
                noised_z = mean_code + tf.multiply(tf.exp(std_code), z)

            net = lrelu(dense(noised_z, 256, name='fc1'), name='l1')
            net = lrelu(bn(dense(net, 128*4*4, name='fc2'), is_training, name='bn1'), name='l2')
            net = tf.reshape(net, (-1, 4, 4, 128))
            
            net = lrelu(bn(deconv2d(net, 64, 4, 4, 1, 1, padding='VALID', name='dc1'), is_training, name='bn2'), name='l3')
            net = lrelu(bn(deconv2d(net, 32, 4, 4, 2, 2, padding='SAME', name='dc2'), is_training, name='bn3'), name='l4')
            net = tf.nn.sigmoid(deconv2d(net, 1, 4, 4, 2, 2, padding='SAME', name='dc3'))
        return noised_z, net


class Variational_AutoEncoder_Seq(BasicBlock):
    def __init__(self, len_latent, name=None):
        super(Variational_AutoEncoder_Seq, self).__init__(name=name or 'AES')
        self.len_latent = len_latent
    
    def encode(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            net = lrelu(conv2d(x, 32, 4, 1, 2, 1, padding='SAME', name='c1'), name='l1')
            net = lrelu(bn(conv2d(net, 64, 4, 1, 2, 1, padding='SAME', name='c2'), is_training, name='bn1'), name='l2')
            net = lrelu(bn(conv2d(net, 128, 4, 2, 3, 2, padding='SAME', name='c3'), is_training, name='bn2'), name='l3')
            net = tf.reshape(net, [-1, 15*128])
            net = lrelu(bn(dense(net, 256, name='fc1'), is_training, name='bn3'), name='l4')

            mean_code = dense(net, self.len_latent, name='fc_mean')
            std_code = dense(net, self.len_latent, name='fc_std')
        return mean_code, std_code
    
    def decode(self, z, mean_code, std_code, noised_z=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            if noised_z is None:
                noised_z = mean_code + tf.multiply(tf.exp(std_code), z)

            net = lrelu(dense(noised_z, 256, name='fc1'), name='l1')
            net = lrelu(bn(dense(net, 128*15, name='fc2'), is_training, name='bn1'), name='l2')
            net = tf.reshape(net, (-1, 15, 1, 128))

            net = lrelu(bn(deconv2d(net, 64, 4, 2, 3, 2, padding='SAME', name='dc1'), is_training, name='bn2'), name='l3')
            net = lrelu(bn(deconv2d(net, 32, 4, 1, 2, 1, padding='SAME', name='dc2'), is_training, name='bn3'), name='l4')
            net = deconv2d(net, 1, 4, 1, 2, 1, padding='SAME', name='dc3')
        return noised_z, net