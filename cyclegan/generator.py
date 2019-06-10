import tensorflow as tf 
from utils import * 

class CNN_Generator_SeqtoImg(BasicBlock):
    def __init__(self, output_dim, name=None):
        super(CNN_Generator_SeqtoImg, self).__init__(None, name or "CNN_Generator")
        self.output_dim = output_dim

    def __call__(self, x, y=None, decode_mode=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            if y is not None:
                batch_size, ydim = y.get_shape().as_list()
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y) # [bz, 180, 2, 20+1]
            
            if decode_mode:
                emb = x
            else:
                pad_x = tf.pad(x, [[0,0],[3,3],[1,1],[0,0]], "REFLECT")
                c1 = tf.nn.relu(bn(conv2d(pad_x, 32, 7, 1, 1, 1, padding="VALID", name="g_c1"), is_training, name='g_bn1'))
                c2 = tf.nn.relu(bn(conv2d(c1, 64, 4, 1, 2, 1, padding="SAME", name="g_c2"), is_training, name='g_bn2'))
                c3 = tf.nn.relu(bn(conv2d(c2, 32, 4, 1, 2, 1, padding="SAME", name='g_c3'), is_training, name='g_bn3'))

                r1 = resnet_block_seq(c3, 32, is_training, name='r1')
                r2 = resnet_block_seq(r1, 32, is_training, name='r2') 
                r3 = resnet_block_seq(r2, 32, is_training, name='r3') 

                emb = r3 
            
            emb = tf.layers.flatten(emb)
            emb = tf.nn.relu(bn(dense(emb, 49*32), is_training, name='emb_bn'))
            emb = tf.reshape(emb, (-1, 7, 7, 32))

            d1 = tf.nn.relu(bn(deconv2d(emb, 64, 3, 3, 2, 2, padding="SAME", name='g_dc1'), is_training, name='g_bn4'))
            d2 = tf.nn.relu(bn(deconv2d(d1, 32, 3, 3, 2, 2, padding="SAME", name='g_dc2'), is_training, name='g_bn5'))
            d2_pad = tf.pad(d2, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            c4 = bn(conv2d(d2_pad, self.output_dim, 7, 7, 1, 1, padding="VALID", name="g_c4"), is_training, name='g_bn6')
            
            out = tf.nn.sigmoid(c4) # [N, 28, 28, 1]
        return out, emb

class CNN_Generator_ImgtoSeq(BasicBlock):
    def __init__(self, output_dim, name=None):
        super(CNN_Generator_ImgtoSeq, self).__init__(None, name or "CNN_Generator")
        self.output_dim = output_dim

    def __call__(self, x, y=None, decode_mode=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            if y is not None:
                batch_size, ydim = y.get_shape().as_list()
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y) # [bz, 28, 28, 20+1]
            
            if decode_mode:
                emb = x
            else:
                pad_x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
                c1 = tf.nn.relu(bn(conv2d(pad_x, 32, 7, 7, 1, 1, padding="VALID", name="g_c1"), is_training, name='g_bn1'))
                c2 = tf.nn.relu(bn(conv2d(c1, 64, 3, 3, 2, 2, padding="SAME", name="g_c2"), is_training, name='g_bn2'))
                c3 = tf.nn.relu(bn(conv2d(c2, 32, 3, 3, 2, 2, padding="SAME", name='g_c3'), is_training, name='g_bn3'))

                r1 = resnet_block_img(c3, 32, is_training, name='r1')
                r2 = resnet_block_img(r1, 32, is_training, name='r2') 
                r3 = resnet_block_img(r2, 32, is_training, name='r3') 

                emb = r3 
            
            emb = tf.layers.flatten(emb)
            emb = bn(dense(emb, 45*32), is_training, name='emb_bn')
            emb = tf.reshape(emb, (-1, 45, 1, 32))

            d1 = tf.nn.relu(bn(deconv2d(emb, 64, 4, 2, 2, 2, padding="SAME", name='g_dc1'), is_training, name='g_bn4'))
            d2 = tf.nn.relu(bn(deconv2d(d1, 32, 4, 1, 2, 1, padding="SAME", name='g_dc2'), is_training, name='g_bn5'))
            d2_pad = tf.pad(d2, [[0,0],[3,3],[0,0],[0,0]], "REFLECT")
            c4 = conv2d(d2_pad, self.output_dim, 7, 1, 1, 1, padding="VALID", name="g_c4")
            
            out = c4
        return out, emb

# x = tf.ones(dtype=tf.float32, shape=[64,180,2,1])
# G1 = CNN_Generator_SeqtoImg(1, name='G1')
# G2 = CNN_Generator_ImgtoSeq(1, name='G2')

# y = G1(x)
# z = G2(y[0])

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(y)[0].shape
#     print sess.run(z)[0].shape