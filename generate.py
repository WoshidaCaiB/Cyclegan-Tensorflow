import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import scipy.misc
import argparse
from model import CycleGan,read_img,shuffle

parser_=argparse.ArgumentParser(description='Inference')
parser_.add_argument('--model_file',dest='model',default=os.path.join(os.getcwd(),'CycleGan.ckpt'),help='model path')
parser_.add_argument('--test file',dest='test',default=os.path.join(os.getcwd(),'horse2zebra'),help='test path')
param=parser_.parse_args()

C=read_img(os.path.join(param.test,'testA'))
D=read_img(os.path.join(param.test,'testB'))

def inference(reuse=True):
    model=CycleGan()
	i=np.random.randint(min(len(C),len(D)))
    gen_B=model.Generator(model.A,'G_AtoB',dim=model.g_dim,reuse=reuse)
    gen_A=model.Generator(model.B,'G_BtoA',dim=model.g_dim,reuse=reuse)
    cyc_A=model.Generator(gen_B,'G_BtoA',dim=model.g_dim,reuse=reuse)
    cyc_B=model.Generator(gen_A,'G_AtoB',dim=model.g_dim,reuse=reuse)
    saver=tf.train.Saver()
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,param.model)
        fake_B,cycA=sess.run([gen_B,cyc_A],feed_dict={model.A:C[i:i+1]})
        fake_A,cycB=sess.run([gen_A,cyc_B],feed_dict={model.B:D[i:i+1]})
        fake_B=((fake_B+1.)*127.5).astype(np.uint8)
        fake_A=((fake_A+1.)*127.5).astype(np.uint8)
        cycA=((cycA+1.)*127.5).astype(np.uint8)
        cycB=((cycB+1.)*127.5).astype(np.uint8)
		plt.subplot(3,2,1)
        plt.imshow(((C[i]+1.)*127.5).astype(np.uint8))
		plt.title('true A')
        plt.axis('off')
        plt.subplot(3,2,2)
        plt.imshow(fake_B[0])
		plt.title('Fake B')
        plt.axis('off')
        plt.subplot(3,2,3)
        plt.imshow(((D[i]+1.)*127.5).astype(np.uint8))
		plt.title('True B')
        plt.axis('off')
        plt.subplot(3,2,4)
        plt.imshow(fake_A[0])
		plt.title('Fake A')
        plt.axis('off')
        plt.subplot(3,2,5)
        plt.imshow(cycA[0])
		plt.title('cyc A')
        plt.axis('off')
        plt.subplot(3,2,6)
        plt.imshow(cycB[0])
		plt.title('cyc B')
        plt.axis('off')
        plt.show()
