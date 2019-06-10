import sys
sys.path.append("..")
from utils import *

class AutoEncoder_Image(BasicBlock):
    def __init__(self, len_latent, name=None):
        super(AutoEncoder_Image, self).__init__(name=name or 'AEI')
        self.len_latent = len_latent
    
    def encode(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            net = lrelu(conv2d(x, 32, 4, 4, 2, 2, padding='SAME', name='c1'), name='l1')
            net = lrelu(bn(conv2d(net, 64, 4, 4, 2, 2, padding='SAME', name='c2'), is_training, name='bn1'), name='l2')
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, padding='SAME', name='c3'), is_training, name='bn2'), name='l3')
            net = tf.reshape(net, [-1, 4*4*128])
            net = lrelu(bn(dense(net, 256, name='fc1'), is_training, name='bn3'), name='l4')
            net = dense(net, self.len_latent, name='fc2')
        return net 
    
    def decode(self, emb, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            net = lrelu(dense(emb, 256, name='fc1'), name='l1')
            net = lrelu(bn(dense(net, 128*4*4, name='fc2'), is_training, name='bn1'), name='l2')
            net = tf.reshape(net, (-1, 4, 4, 128))
            net = lrelu(bn(deconv2d(net, 64, 4, 4, 1, 1, padding='VALID', name='dc1'), is_training, name='bn2'), name='l3')
            net = lrelu(bn(deconv2d(net, 32, 4, 4, 2, 2, padding='SAME', name='dc2'), is_training, name='bn3'), name='l4')
            net = tf.nn.sigmoid(deconv2d(net, 1, 4, 4, 2, 2, padding='SAME', name='dc3'))
        return net

class AutoEncoder_Seq(BasicBlock):
    def __init__(self, len_latent, name=None):
        super(AutoEncoder_Seq, self).__init__(name=name or 'AES')
        self.len_latent = len_latent
    
    def encode(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            net = lrelu(conv2d(x, 32, 4, 1, 2, 1, padding='SAME', name='c1'), name='l1')
            net = lrelu(bn(conv2d(net, 64, 4, 1, 2, 1, padding='SAME', name='c2'), is_training, name='bn1'), name='l2')
            net = lrelu(bn(conv2d(net, 128, 4, 2, 3, 2, padding='SAME', name='c3'), is_training, name='bn2'), name='l3')
            net = tf.reshape(net, [-1, 15*128])
            net = lrelu(bn(dense(net, 256, name='fc1'), is_training, name='bn3'), name='l4')
            net = dense(net, self.len_latent, name='fc2')
        return net
    
    def decode(self, emb, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            net = lrelu(dense(emb, 256, name='fc1'), name='l1')
            net = lrelu(bn(dense(net, 128*15, name='fc2'), is_training, name='bn1'), name='l2')
            net = tf.reshape(net, (-1, 15, 1, 128))
            net = lrelu(bn(deconv2d(net, 64, 4, 2, 3, 2, padding='SAME', name='dc1'), is_training, name='bn2'), name='l3')
            net = lrelu(bn(deconv2d(net, 32, 4, 1, 2, 1, padding='SAME', name='dc2'), is_training, name='bn3'), name='l4')
            net = deconv2d(net, 1, 4, 1, 2, 1, padding='SAME', name='dc3')
        return net

# X = tf.ones(dtype=tf.float32, shape=(5,28,28,1))
# AE = AutoEncoder_Image(10)
# r = AE.encode(X)
# d = AE.decode(r)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(r).shape
#     print sess.run(d).shape
#     vs = AE.vars 
#     for v in vs:
#         print v.name, v.shape

# X = tf.ones(dtype=tf.float32, shape=(5,180,2,1))
# AE = AutoEncoder_Seq(10)
# r = AE.encode(X)
# d = AE.decode(r)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(r).shape
#     print sess.run(d).shape