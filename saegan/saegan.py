
import sys
sys.path.append("..")

from utils import *
from autoencoder import AutoEncoder_Image, AutoEncoder_Seq
from variational_autoencoder import Variational_AutoEncoder_Image, Variational_AutoEncoder_Seq
from latent_discriminator import Latent_Discriminator
from latent_classifier import Latent_Classifier
from datamanager import datamanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def imshow(imgs, save_path):
    tmp = [[] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            tmp[i].append(imgs[i*5+j])
        tmp[i] = np.concatenate(tmp[i], 1)
    tmp.reverse()
    tmp = np.concatenate(tmp, 0)
    plt.imshow(tmp[:,:,0], cmap=plt.cm.gray, origin='lower')
    plt.savefig(save_path)
    plt.clf()
def plot(seqs, save_path):
    for i in range(5):
        for j in range(5):
            idx = i*5 + j
            plt.subplot(5, 5, idx+1)
            plt.plot(seqs[idx][:, 0, 0], seqs[idx][:, 1, 0], linewidth=2, color='b')
            plt.xticks([])
            plt.yticks([])
    plt.savefig(save_path)
    plt.clf()

class SAEGAN(BasicTrainFramework):
    def __init__(self, batch_size, version='saegan', gpu='0'):
        super(SAEGAN, self).__init__(batch_size, version, gpu)

        self.data_img = datamanager('CT_img', train_ratio=0.8, expand_dim=3, seed=0)
        self.data_seq = datamanager('CT_seq', train_ratio=0.8, expand_dim=3, seed=1)

        self.sample_data_Img = self.data_img(self.batch_size, phase='test', var_list=['data'])
        self.sample_data_Seq = self.data_seq(self.batch_size, phase='test', var_list=['data'])

        self.len_latent = 64
        # self.autoencoder_img = AutoEncoder_Image(self.len_latent, name='AEI')
        # self.autoencoder_seq = AutoEncoder_Seq(self.len_latent, name='AES')
        self.autoencoder_img = Variational_AutoEncoder_Image(self.len_latent, name='VAEI')
        self.autoencoder_seq = Variational_AutoEncoder_Seq(self.len_latent, name='VAES')
        self.latent_discriminator = Latent_Discriminator(name='LD')

        self.class_num = 20
        self.latent_classifier = Latent_Classifier(class_num=self.class_num, name='LC')

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()

        self.build_dirs(appendix="../")
        self.build_sess()

    def build_placeholder(self):
        self.source_img = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.source_seq = tf.placeholder(shape=(self.batch_size, 180, 2, 1), dtype=tf.float32)

        self.target_img = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.target_seq = tf.placeholder(shape=(self.batch_size, 180, 2, 1), dtype=tf.float32)

        self.labels_img = tf.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)
        self.labels_seq = tf.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)
    
    def build_network(self):
        # self.emb_img = self.autoencoder_img.encode(self.source_img, True, False)
        # self.cyc_img = self.autoencoder_img.decode(self.emb_img, True, False)
        # self.emb_seq = self.autoencoder_seq.encode(self.source_seq, True, False)
        # self.cyc_seq = self.autoencoder_seq.decode(self.emb_seq, True, False)

        # self.emb_img2seq = self.autoencoder_seq.encode(self.autoencoder_seq.decode(self.emb_img, True, True), True, True)
        # self.emb_seq2img = self.autoencoder_img.encode(self.autoencoder_img.decode(self.emb_seq, True, True), True, True)

        # self.emb_img_test = self.autoencoder_img.encode(self.source_img, False, True)
        # self.cyc_img_test = self.autoencoder_img.decode(self.emb_img_test, False, True)
        # self.emb_seq_test = self.autoencoder_seq.encode(self.source_seq, False, True)
        # self.cyc_seq_test = self.autoencoder_seq.decode(self.emb_seq_test, False, True)

        # self.cross_img2seq_test = self.autoencoder_seq.decode(self.emb_img_test, False, True)
        # self.cross_seq2img_test = self.autoencoder_img.decode(self.emb_seq_test, False, True)

        mean_code, std_code = self.autoencoder_img.encode(self.source_img, True, False)
        gaussian_noise = tf.random_normal(tf.shape(mean_code), 0.0, 1.0, dtype=tf.float32)
        self.emb_img, self.cyc_img = self.autoencoder_img.decode(gaussian_noise, mean_code, std_code, None, True, False)

        mean_code_test, std_code_test = self.autoencoder_img.encode(self.source_img, False, True)
        self.emb_img_test, self.cyc_img_test = self.autoencoder_img.decode(gaussian_noise, mean_code_test, std_code_test, None, False, True)

        mean_code, std_code = self.autoencoder_seq.encode(self.source_seq, True, False)
        gaussian_noise = tf.random_normal(tf.shape(mean_code), 0.0, 1.0, dtype=tf.float32)
        self.emb_seq, self.cyc_seq = self.autoencoder_seq.decode(gaussian_noise, mean_code, std_code, None, True, False)

        mean_code_test, std_code_test = self.autoencoder_seq.encode(self.source_seq, False, True)
        self.emb_seq_test, self.cyc_seq_test = self.autoencoder_seq.decode(gaussian_noise, mean_code_test, std_code_test, None, False, True)

        _, tmp = self.autoencoder_seq.decode(None, None, None, self.emb_img, True, True)
        gaussian_noise = tf.random_normal(tf.shape(mean_code), 0.0, 1.0, dtype=tf.float32)
        mean_code, std_code = self.autoencoder_seq.encode(tmp, True, True)
        self.emb_img2seq, _ = self.autoencoder_seq.decode(gaussian_noise, mean_code, std_code, None, True, True)
        _, tmp = self.autoencoder_img.decode(None, None, None, self.emb_seq, True, True)
        gaussian_noise = tf.random_normal(tf.shape(mean_code), 0.0, 1.0, dtype=tf.float32)
        mean_code, std_code = self.autoencoder_img.encode(tmp, True, True)
        self.emb_seq2img, _ = self.autoencoder_img.decode(gaussian_noise, mean_code, std_code, None, True, True)
        
        _, self.cross_img2seq_test = self.autoencoder_seq.decode(None, None, None, self.emb_img_test, False, True)
        _, self.cross_seq2img_test = self.autoencoder_img.decode(None, None, None, self.emb_seq_test, False, True)



        self.latent_logit_img = self.latent_discriminator(self.emb_img, True, False)
        self.latent_logit_seq = self.latent_discriminator(self.emb_seq, True, True)

        self.cls_img = self.latent_classifier(self.emb_img, True, False)
        self.cls_seq = self.latent_classifier(self.emb_seq, True, True)


    
    def build_optimizer(self):
        self.cycloss_img = tf.reduce_mean(tf.squared_difference(self.cyc_img, self.target_img))
        self.cycloss_seq = tf.reduce_mean(tf.squared_difference(self.cyc_seq, self.target_seq))

        self.embloss_img = tf.reduce_mean(tf.squared_difference(self.emb_img, self.emb_img2seq))
        self.embloss_seq = tf.reduce_mean(tf.squared_difference(self.emb_seq, self.emb_seq2img))

        self.latent_D_loss = tf.reduce_mean(tf.squared_difference(self.latent_logit_seq, 1.0)) + tf.reduce_mean(tf.square(self.latent_logit_img))
        self.latent_G_loss = tf.reduce_mean(tf.squared_difference(self.latent_logit_img, 1.0)) 

        self.clsloss_img = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_img, labels=self.labels_img))
        self.clsloss_seq = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_seq, labels=self.labels_seq))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.cycsolver_img = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.cycloss_img, var_list=self.autoencoder_img.vars)
            self.cycsolver_seq = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.cycloss_seq, var_list=self.autoencoder_seq.vars)

            self.embsolver_img = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.embloss_img, var_list=self.autoencoder_seq.vars)
            self.embsolver_seq = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.embloss_seq, var_list=self.autoencoder_img.vars)

            self.latent_D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.latent_D_loss, var_list=self.latent_discriminator.vars)
            self.latent_G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.latent_G_loss, var_list=self.autoencoder_img.vars)

            self.clssolver_img = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.clsloss_img, var_list=self.autoencoder_img.vars + self.latent_classifier.vars)
            self.clssolver_seq = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.clsloss_seq, var_list=self.autoencoder_seq.vars + self.latent_classifier.vars)
            
    
    def sample(self, epoch):
        print "sample at epoch {}".format(epoch)

        feed_dict = {self.source_img : self.sample_data_Img['data']}
        gen = self.sess.run(self.cyc_img, feed_dict=feed_dict)
        imshow(gen, os.path.join(self.fig_dir, "cyc_img_{}.png".format(epoch)))

        feed_dict = {self.source_seq : self.sample_data_Seq['data']}
        gen = self.sess.run(self.cyc_seq, feed_dict=feed_dict)
        plot(gen, os.path.join(self.fig_dir, "cyc_seq_{}.png".format(epoch)))

        if epoch == 0:
            imshow(self.sample_data_Img['data'], os.path.join(self.fig_dir, "real_img.png"))
            plot(self.sample_data_Seq['data'], os.path.join(self.fig_dir, "real_seq.png"))
    
    def tsne(self):
        data_img = datamanager('CT_img', train_ratio=0.8, expand_dim=3, seed=0)
        data_seq = datamanager('CT_seq', train_ratio=0.8, expand_dim=3, seed=1)

        img_embs, seq_embs, img_labels, seq_labels = [], [], [], []

        for i in range(data_img.train_num // self.batch_size + 1):
            img = data_img(self.batch_size, phase='train', var_list=['data', 'labels'])
            seq = data_seq(self.batch_size, phase='train', var_list=['data', 'labels'])
            feed_dict = {self.source_img : img['data'],
                        self.source_seq : seq['data']}
            emb_img, emb_seq = self.sess.run([self.emb_img_test, self.emb_seq_test], feed_dict=feed_dict)
            img_embs.append(emb_img)
            seq_embs.append(emb_seq)
            
            img_labels.append(np.argmax(img['labels'], 1))
            seq_labels.append(np.argmax(seq['labels'], 1))
        
        embs = np.concatenate(img_embs + seq_embs, axis=0)
        img_labels = np.concatenate(img_labels)
        seq_labels = np.concatenate(seq_labels)
        domain_labels = np.array([0] * (len(img_embs)*self.batch_size) + [1] * (len(seq_embs)*self.batch_size))
        print embs.shape, domain_labels.shape

        from sklearn.manifold import TSNE 
        model = TSNE(n_components=2, random_state=0)
        embs = model.fit_transform(embs)
        # np.save(os.path.join(self.fig_dir, "tsne", 'emb.npy'), embs)

        # embs = np.load(os.path.join(self.fig_dir, "tsne", 'emb.npy'))

        plt.scatter(embs[:,0], embs[:,1], c=domain_labels)
        plt.colorbar()

        keys = ['a','b','c','d','e','g','h','l','m','n','o','p','q','r','s','u','v','w','y','z' ]
        img_labels_dict = {}
        for i in range(200):
            img_labels_dict[img_labels[i]] = i
        seq_labels_dict = {}
        for i in range(200):
            seq_labels_dict[seq_labels[i]] = i + len(img_labels)
        for i,v in img_labels_dict.iteritems():
            plt.text(embs[v][0]-5, embs[v][1]+5, s=keys[i], fontsize=10, color='b')
        for i,v in seq_labels_dict.iteritems():
            plt.text(embs[v][0]+5, embs[v][1]-5, s=keys[i], fontsize=10, color='r')

        plt.savefig(os.path.join(self.fig_dir, "tsne", "tsne.png"))
        plt.clf()
    
    def cross(self):
        img = self.data_img(self.batch_size, 'train', var_list=['data'])
        seq = self.data_seq(self.batch_size, 'train', var_list=['data'])

        feed_dict = {
            self.source_img : img['data'],
            self.source_seq : seq['data']
        }
        cross_img2seq, cross_seq2img, cyc_img, cyc_seq = self.sess.run([self.cross_img2seq_test, self.cross_seq2img_test, self.cyc_img_test, self.cyc_seq_test], feed_dict=feed_dict)
        print cross_img2seq.shape, cross_seq2img.shape
        imshow(img['data'], os.path.join(self.fig_dir, 'cross', 'img.png'))
        plot(cross_img2seq, os.path.join(self.fig_dir, 'cross', 'cross_img2seq.png'))

        plot(seq['data'], os.path.join(self.fig_dir, 'cross', 'seq.png'))
        imshow(cross_seq2img, os.path.join(self.fig_dir, 'cross', 'cross_seq2img.png'))

        imshow(cyc_img, os.path.join(self.fig_dir, 'cross', 'cyc_img.png'))
        plot(cyc_seq, os.path.join(self.fig_dir, 'cross', 'cyc_seq.png'))

    def test(self):
        for _ in range(10):
            img = self.data_img(self.batch_size, 'train', var_list=['data'])
            seq = self.data_seq(self.batch_size, 'train', var_list=['data'])

            feed_dict = {
                self.source_img : img['data'],
                self.source_seq : seq['data']
            }

            a,b = self.sess.run([self.latent_D_loss, self.latent_G_loss], feed_dict=feed_dict)
            print "D=%.4f G=%.4f" % (a,b)

    def train(self, epoches=1):
        batches_per_epoch = self.data_img.train_num // self.batch_size

        for epoch in range(epoches):
            self.data_img.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx

                img = self.data_img(self.batch_size, 'train', var_list=['data', 'labels'])
                seq = self.data_seq(self.batch_size, 'train', var_list=['data', 'labels'])

                feed_dict = {
                    self.source_img : img['data'],
                    self.target_img : img['data'],
                    self.labels_img : img['labels'],

                    self.source_seq : seq['data'],
                    self.target_seq : seq['data'],
                    self.labels_seq : seq['labels']
                }

                self.sess.run(self.cycsolver_img, feed_dict=feed_dict)
                self.sess.run(self.cycsolver_seq, feed_dict=feed_dict)

                self.sess.run(self.clssolver_img, feed_dict=feed_dict)
                self.sess.run(self.clssolver_seq, feed_dict=feed_dict)

                self.sess.run(self.embsolver_img, feed_dict=feed_dict)
                self.sess.run(self.embsolver_seq, feed_dict=feed_dict)

                for _ in range(3):
                    self.sess.run(self.latent_D_solver, feed_dict=feed_dict)
                self.sess.run(self.latent_G_solver, feed_dict=feed_dict)

                if cnt % 25 == 0:
                    cycloss_img, cycloss_seq = self.sess.run([self.cycloss_img, self.cycloss_seq], feed_dict=feed_dict)
                    print self.version + " Epoch [%3d/%3d] Iter [%3d] img_loss=%.4f seq_loss=%.4f" % (epoch, epoches, idx, cycloss_img, cycloss_seq)
                
            if epoch % 200 == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))

saegan = SAEGAN(64, version='svaegan_ldlcle', gpu='0')
# saegan.train(1000)
# saegan.sample(0)
saegan.load_model(ckpt_name='model.ckpt')

# saegan.tsne()
saegan.test()
# saegan.cross()
        

        

