# coding:utf-8
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf 
import numpy as np
import os
from generator import CNN_Generator_ImgtoSeq, CNN_Generator_SeqtoImg
from discriminator import CNN_Discriminator_Img, CNN_Discriminator_Seq
from utils import BasicTrainFramework, event_reader
from datamanager import datamanager
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CycleGAN(BasicTrainFramework):
    def __init__(self, batch_size, gan_type='gan', version="cyclegan"):
        self.gan_type = gan_type
        super(CycleGAN, self).__init__(batch_size, version)

        self.data_Seq = datamanager('CT', train_ratio=0.8, expand_dim=3, seed=0)
        self.data_Img = datamanager('CT_img', train_ratio=0.8, expand_dim=3, seed=1)
        self.sample_data_Seq = self.data_Seq(self.batch_size, phase='test', var_list=['data'])
        self.sample_data_Img = self.data_Img(self.batch_size, phase='test', var_list=['data'])

        self.critic_iter = 3

        self.generator_SeqtoImg = CNN_Generator_SeqtoImg(output_dim=1, name="cnn_generator_SeqtoImg")
        self.generator_ImgtoSeq = CNN_Generator_ImgtoSeq(output_dim=1, name="cnn_generator_ImgtoSeq")
        self.discriminator_Seq = CNN_Discriminator_Seq(name="cnn_discriminator_Seq")
        self.discriminator_Img = CNN_Discriminator_Img(name="cnn_discriminator_Img")

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        # gray image
        self.source_Seq = tf.placeholder(shape=(self.batch_size, 180, 2, 1), dtype=tf.float32)
        # colored image
        self.source_Img = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)

    def build_network(self):
        # cyclegan
        self.fake_Img, _ = self.generator_SeqtoImg(self.source_Seq, is_training=True, reuse=False)
        self.fake_Seq, _ = self.generator_ImgtoSeq(self.source_Img, is_training=True, reuse=False)
        self.fake_Img_test, _ = self.generator_SeqtoImg(self.source_Seq, is_training=False, reuse=True)
        self.fake_Seq_test, _ = self.generator_ImgtoSeq(self.source_Img, is_training=False, reuse=True)

        self.logit_real_Seq, _ = self.discriminator_Seq(self.source_Seq, is_training=True, reuse=False)
        self.logit_real_Img, _ = self.discriminator_Img(self.source_Img, is_training=True, reuse=False)
        self.logit_fake_Seq, _ = self.discriminator_Seq(self.fake_Seq, is_training=True, reuse=True)
        self.logit_fake_Img, _ = self.discriminator_Img(self.fake_Img, is_training=True, reuse=True)

        self.cyc_Seq, _ = self.generator_ImgtoSeq(self.fake_Img, is_training=True, reuse=True)
        self.cyc_Img, _ = self.generator_SeqtoImg(self.fake_Seq, is_training=True, reuse=True)

    def build_optimizer(self):
        # self.reconstruct_loss = mse(self.fake_A, self.source_A, self.batch_size)
        # if self.gan_type == 'gan':
        #     self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        #     self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        #     self.D_loss = self.D_loss_real + self.D_loss_fake
        #     self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        # elif self.gan_type == 'wgan':
        #     self.D_loss_real = -tf.reduce_mean(self.logit_real)
        #     self.D_loss_fake = tf.reduce_mean(self.logit_fake)
        #     self.D_loss = self.D_loss_real + self.D_loss_fake
        #     self.G_loss = - self.D_loss_fake
        #     self.D_clip = [v.assign(tf.clip_by_value(v, -0.1, 0.1)) for v in self.discriminator_B.vars]
        self.D_loss_real_Seq = tf.reduce_mean(tf.squared_difference(self.logit_real_Seq, 1))
        self.D_loss_real_Img = tf.reduce_mean(tf.squared_difference(self.logit_real_Img, 1))
        self.D_loss_fake_Seq = tf.reduce_mean(tf.square(self.logit_fake_Seq))
        self.D_loss_fake_Img = tf.reduce_mean(tf.square(self.logit_fake_Img))
        self.D_loss_Seq = self.D_loss_real_Seq + self.D_loss_fake_Seq
        self.D_loss_Img = self.D_loss_real_Img + self.D_loss_fake_Img 
        self.reconstruct_loss = tf.reduce_mean(tf.abs(self.source_Seq-self.cyc_Seq)) \
            + tf.reduce_mean(tf.abs(self.source_Img-self.cyc_Img))

        self.G_loss_Seq = tf.reduce_mean(tf.squared_difference(self.logit_fake_Img, 1)) + 10*self.reconstruct_loss
        self.G_loss_Img = tf.reduce_mean(tf.squared_difference(self.logit_fake_Seq, 1)) + 10*self.reconstruct_loss
        
        self.D_solver_Seq = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss_Seq, var_list=self.discriminator_Seq.vars)
        self.D_solver_Img = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss_Img, var_list=self.discriminator_Img.vars)
        self.G_solver_Seq = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss_Seq, var_list=self.generator_SeqtoImg.vars)
        self.G_solver_Img = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss_Img, var_list=self.generator_ImgtoSeq.vars)

    def build_summary(self):
        R_sum = tf.summary.scalar('reconstruct_loss', self.reconstruct_loss)
        D_sum_Seq = tf.summary.scalar('D_loss_Seq', self.D_loss_Seq)
        G_sum_Seq = tf.summary.scalar('G_loss_Seq', self.G_loss_Seq)
        D_sum_Img = tf.summary.scalar('D_loss_Img', self.D_loss_Img)
        G_sum_Img = tf.summary.scalar('G_loss_Img', self.G_loss_Img)
        self.summary = tf.summary.merge([R_sum, D_sum_Seq, G_sum_Seq, D_sum_Img, G_sum_Img])
    
    def plot(self, imgs, savepath, plot_type='img'):
        if plot_type == 'img':
            # imgs [bz, 28, 28, 3 or 1]
            tmp = [[] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    tmp[i].append(imgs[i*5+j])
                tmp[i] = np.concatenate(tmp[i], 1)
            tmp = np.concatenate(tmp, 0)
            if tmp.shape[-1] == 1:
                plt.imshow(tmp[:,:,0], cmap=plt.cm.gray, origin='lower')
            else:
                plt.imshow(tmp, origin='lower')
            plt.savefig(savepath)
            plt.clf()
        elif plot_type == 'seq':
            for i in range(5):
                for j in range(5):
                    idx = i*5 + j
                    plt.subplot(5, 5, idx+1)
                    plt.plot(imgs[idx, :, 0, 0], imgs[idx, :, 1, 0], linewidth=2)
                    plt.xticks([])
                    plt.yticks([])
            plt.savefig(savepath)
            plt.clf()


    def sample(self, epoch):
        print "sample at epoch {}".format(epoch)
        feed_dict = {self.source_Seq : self.sample_data_Seq['data']}
        G = self.sess.run(self.fake_Img_test, feed_dict=feed_dict)
        self.plot(G, os.path.join(self.fig_dir, "AtoB_epoch_{}.png".format(epoch)), 'img')

        feed_dict = {self.source_Img : self.sample_data_Img['data']}
        G = self.sess.run(self.fake_Seq_test, feed_dict=feed_dict)
        self.plot(G, os.path.join(self.fig_dir, "BtoA_epoch_{}.png".format(epoch)), 'seq')

        if epoch == 0:
            self.plot(self.sample_data_Seq['data'], os.path.join(self.fig_dir, "ori_Seq.png"), 'seq')
            self.plot(self.sample_data_Img['data'], os.path.join(self.fig_dir, "ori_Img.png"), 'img')

    def plot_loss(self):
        res = event_reader(self.log_dir, 
            names=['reconstruct_loss', 'G_loss_Seq', 'G_loss_Img', 'D_loss_Seq', 'D_loss_Img'])
        total_iters = len(res['reconstruct_loss'])
        plt.clf()
        plt.gca().set_ylim([0, 1])
        plt.plot(range(total_iters), res['reconstruct_loss'], label='reconstruct_loss')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "reconstruct_loss.png"))
        plt.clf()

        plt.gca().set_ylim([0, 9])
        plt.plot(range(total_iters), res['G_loss_Seq'], label='G_loss_Seq')
        plt.plot(range(total_iters), res['D_loss_Seq'], label='D_loss_Seq')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "GD_loss_Seq.png"))
        plt.clf()

        plt.gca().set_ylim([0, 9])
        plt.plot(range(total_iters), res['G_loss_Img'], label='G_loss_Img')
        plt.plot(range(total_iters), res['D_loss_Img'], label='D_loss_Img')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "GD_loss_Img.png"))
        plt.clf()

    
    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data_Seq.train_num // self.batch_size

        for epoch in range(epoches):
            self.data_Seq.shuffle_train(seed=epoch)
            self.data_Img.shuffle_train(seed=epoch+1)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                Seq = self.data_Seq(self.batch_size, var_list=['data'])
                Img = self.data_Img(self.batch_size, var_list=['data'])

                feed_dict = {
                    self.source_Seq : Seq['data'], 
                    self.source_Img : Img['data']
                }

                self.sess.run([self.G_solver_Seq, self.D_solver_Img], feed_dict=feed_dict)
                self.sess.run([self.G_solver_Img, self.D_solver_Seq], feed_dict=feed_dict)

                if cnt % 10 == 0:
                    da, db, ga, gb, r, sum_str = self.sess.run([self.D_loss_Seq, self.D_loss_Img, self.G_loss_Seq, self.G_loss_Img, self.reconstruct_loss, self.summary], feed_dict=feed_dict)
                    print self.version + " Epoch [%3d/%3d] Iter [%3d/%3d] Dseq=%.3f Dimg=%.3f Gseq=%.3f Gimg=%.3f R=%.3f" % (epoch, epoches, idx, batches_per_epoch, da, db, ga, gb, r)
                    self.writer.add_summary(sum_str, cnt)
            if epoch % 10 == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)


cyclegan = CycleGAN(64, 'gan', 'cyclegan')
# cyclegan.load_model()
cyclegan.train(100)