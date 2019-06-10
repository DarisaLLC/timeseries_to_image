# coding:utf-8
import sys 
sys.path.append("..")
from utils import * 

class Latent_Classifier(BasicBlock):
    def __init__(self, class_num, name=None):
        super(Latent_Classifier, self).__init__(name or "LC")
        self.class_num = class_num
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            net = lrelu(bn(dense(x, 512, name='fc1'), is_training, name='bn1'), name='l1')
            net = lrelu(bn(dense(net, 256, name='fc2'), is_training, name='bn2'), name='l2')
            net = lrelu(bn(dense(net, 128, name='fc3'), is_training, name='bn3'), name='l3')
            net = dense(net, self.class_num, name='fc4')
        
        return net