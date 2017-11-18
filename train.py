import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import scipy.misc
import argparse

def train(a,b,epoch_num):
    total_step=min(len(a),len(b))
    cyc=CycleGan()
    cyc.model()
    cyc.loss()
    lr=tf.placeholder(dtype=tf.float32,shape=[])
    var_list=tf.trainable_variables()
    g_A_var=[var for var in var_list if 'G_AtoB' in var.name]
    g_B_var=[var for var in var_list if 'G_BtoA' in var.name]
    d_A_var=[var for var in var_list if 'Dis_A' in var.name]
    d_B_var=[var for var in var_list if 'Dis_B' in var.name]
    
    g_var=[var for var in var_list if 'G_AtoB' in var.name or 'G_BtoA' in var.name]
    
    op=tf.train.AdamOptimizer(lr)
    
    op_ga=op.minimize(cyc.g_loss_A,var_list=g_A_var)
    op_gb=op.minimize(cyc.g_loss_B,var_list=g_B_var)
    op_da=op.minimize(cyc.D_A_loss,var_list=d_A_var)
    op_db=op.minimize(cyc.D_B_loss,var_list=d_B_var)
    
    op_g=op.minimize(cyc.g_loss,var_list=g_var)
    
    saver=tf.train.Saver(tf.global_variables())
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        curr_lr=0.0002
        for epoch in range(epoch_num):
            A=shuffle(a)
            B=shuffle(b)
            if epoch<=100:
                curr_lr=0.0002
            else:
                curr_lr=0.0002-0.0002*(epoch-100)/100
            for step in range(total_step):
                curr_A=np.array([A[step]])
                curr_B=np.array([B[step]])
                
                curr_gen_A,curr_gen_B=sess.run([cyc.gen_A,cyc.gen_B],feed_dict={cyc.A:curr_A,cyc.B:curr_B,lr:curr_lr})
                fake_b=cyc.fake_image(curr_gen_B,cyc.fake_B_pool)
                fake_a=cyc.fake_image(curr_gen_A,cyc.fake_A_pool)
                _,curr_g_loss,curr_cyc_loss=sess.run([op_g,cyc.g_loss,cyc.cyc_loss],feed_dict={cyc.A:curr_A,cyc.B:curr_B,lr:curr_lr})
                _,curr_dA_loss=sess.run([op_da,cyc.D_A_loss],feed_dict={cyc.A:curr_A,cyc.gen_A:fake_a,lr:curr_lr})
                _,curr_dB_loss=sess.run([op_db,cyc.D_B_loss],feed_dict={cyc.B:curr_B,cyc.gen_B:fake_b,lr:curr_lr})
        
                if epoch%5==0 and step%500==0:
                    print('### epoch {}, step {} ### \n *** g loss: {} \n *** cyc loss: {} \n *** dA loss: {} \n *** dB loss: {} \n'.format(epoch,step,curr_g_loss,curr_cyc_loss,curr_dA_loss,curr_dB_loss))
               
        saver.save(sess,os.path.join(os.getcwd(), 'CycleGan_model/cyclegan1.ckpt'))