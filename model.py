import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import scipy.misc

'''
Simple implementation of cyclegan: https://arxiv.org/abs/1703.10593

Python: 3.6 
Tensorflow: 1.4

Encoder structure:
3 convolution layers (3*3 kernel) + 9 residual block (2 3*3 convolution + instance norm)+ 3 convolution layers all with instance norm layer + relu

Decoder structure:
5 convolution layers (4*4 kernel size) + leaky relu

Loss function follows paper using the least square distance 
'''

class CycleGan:
    def __init__(self,**param):
        self.g_dim=param.get('generator_dim',32)
        self.d_dim=param.get('discriminator_dim',64)
        self.batch_size=param.get('batch_size',1)
        self.image_size=param.get('image_size',256)
        self.pool_size=param.get('pool_size',50)
        self.fake_A_pool=[]
        self.fake_B_pool=[]
        self.A=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.image_size,self.image_size,3])
        self.B=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.image_size,self.image_size,3])
    
    def C7(self,x,dim,stddev,name,act='relu'):
        with tf.variable_scope(name):
            padded=tf.pad(x,[[0,0],[3,3],[3,3],[0,0]],'REFLECT')
            out=tf.layers.conv2d(padded,dim,(7,7),(1,1),padding='VALID',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='c7')
            out=self.instance_norm(out,'norm_c7')
            if act=='relu':
                out=tf.nn.relu(out)
            if act=='tanh':
                out=tf.nn.tanh(out)
        return out
    
    def C3(self,x,dim,stddev,name):
        with tf.variable_scope(name):
            out=tf.layers.conv2d(x,dim,(3,3),(2,2),padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='c3')
            out=self.instance_norm(out,'norm_c3')
            out=tf.nn.relu(out)
        return out
    
    def DC3(self,x,dim,stddev,name):
        with tf.variable_scope(name):
            out=tf.layers.conv2d_transpose(x,dim,(3,3),(2,2),padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='dc3')
            out=self.instance_norm(out,'norm_dc3')
            out=tf.nn.relu(out)
        return out
    
    #Discriminator layer
    def C4(self,x,dim,name,stride=2,stddev=0.02):
        with tf.variable_scope(name):
            out=tf.layers.conv2d(x,dim,(4,4),(stride,stride),padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='dc3')
            out=self.instance_norm(out,'norm_dc3')
            out=self.lrelu(out)
        return out
    
    def RB(self,x,dim,stddev,name):
        with tf.variable_scope(name):
            out=tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],'REFLECT')
            out1=tf.layers.conv2d(out,dim,(3,3),(1,1),padding='VALID',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='re_1')
            out1=self.instance_norm(out1,'norm1')
            out1=tf.nn.relu(out1)
            out1=tf.pad(out1,[[0,0],[1,1],[1,1],[0,0]],'REFLECT')
            out2=tf.layers.conv2d(out1,dim,(3,3),(1,1),padding='VALID',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),use_bias=False,name='re_2')
            out2=self.instance_norm(out2,'norm2')
        return x+out2
            
    def Generator(self,x,name,dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            oc_1=self.C7(x,dim,0.02,'c7')
            oc_2=self.C3(oc_1,dim*2,0.02,'c3_1')
            oc_3=self.C3(oc_2,dim*4,0.02,'c3_2')
            
            re_1=self.RB(oc_3,dim*4,0.02,'re_1')
            re_2=self.RB(re_1,dim*4,0.02,'re_2')
            re_3=self.RB(re_2,dim*4,0.02,'re_3')
            re_4=self.RB(re_3,dim*4,0.02,'re_4')
            re_5=self.RB(re_4,dim*4,0.02,'re_5')
            re_6=self.RB(re_5,dim*4,0.02,'re_6')
            re_7=self.RB(re_6,dim*4,0.02,'re_7')
            re_8=self.RB(re_7,dim*4,0.02,'re_8')
            re_9=self.RB(re_8,dim*4,0.02,'re_9')
            
            od_1=self.DC3(re_9,dim*2,0.02,'dc3_1')
            od_2=self.DC3(od_1,dim,0.02,'dc3_2')
            #output=self.C7(od_2,3,0.02,'final_layer',act='tanh')
            output=tf.pad(od_2,[[0,0],[3,3],[3,3],[0,0]],'REFLECT')
            output=tf.layers.conv2d(output,3,(7,7),(1,1),padding='VALID',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='final_layer')
            output=tf.nn.tanh(output)
        return output
    
    def Discriminator(self,x,name,dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            d_1=self.C4(x,dim,'d_1')
            d_2=self.C4(d_1,dim*2,'d_2')
            d_3=self.C4(d_2,dim*4,'d_3')
            d_4=self.C4(d_3,dim*8,'d_4',stride=1)
            output=tf.layers.conv2d(d_4,1,(4,4),(1,1),padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='final_layer')
            output=tf.nn.sigmoid(output)
        return output
    
    def model(self):
        self.gen_B=self.Generator(self.A,'G_AtoB',dim=self.g_dim)
        self.gen_A=self.Generator(self.B,'G_BtoA',dim=self.g_dim)
        self.dis_A=self.Discriminator(self.A,'Dis_A',dim=self.d_dim)
        self.dis_B=self.Discriminator(self.B,'Dis_B',dim=self.d_dim)
        
        self.dis_gen_A=self.Discriminator(self.gen_A,'Dis_A',dim=self.d_dim,reuse=True)
        self.dis_gen_B=self.Discriminator(self.gen_B,'Dis_B',dim=self.d_dim,reuse=True)
        self.cyc_A=self.Generator(self.gen_B,'G_BtoA',dim=self.g_dim,reuse=True)
        self.cyc_B=self.Generator(self.gen_A,'G_AtoB',dim=self.g_dim,reuse=True)
        
    def loss(self):
        D_A_loss_1=tf.reduce_mean(tf.squared_difference(self.dis_A,1.))
        D_B_loss_1=tf.reduce_mean(tf.squared_difference(self.dis_B,1.))
    
        D_A_loss_2=tf.reduce_mean(tf.square(self.dis_gen_A))
        D_B_loss_2=tf.reduce_mean(tf.square(self.dis_gen_B))

        self.D_A_loss=(D_A_loss_1+D_A_loss_2)/2.
        self.D_B_loss=(D_B_loss_1+D_B_loss_2)/2.
    
        g_loss_B_1=tf.reduce_mean(tf.squared_difference(self.dis_gen_A,1))
        g_loss_A_1=tf.reduce_mean(tf.squared_difference(self.dis_gen_B,1))
    
        self.cyc_loss=tf.reduce_mean(tf.abs(self.A-self.cyc_A))+tf.reduce_mean(tf.abs(self.B-self.cyc_B))
    
        self.g_loss_A=g_loss_A_1+10*self.cyc_loss
        self.g_loss_B=g_loss_B_1+10*self.cyc_loss
        self.g_loss=g_loss_A_1+g_loss_B_1+10*self.cyc_loss

    def fake_image(self,img,img_pool):
        if len(img_pool)<self.pool_size:
            img_pool.append(img)
            return img
        else:
            p=np.random.random()
            if p>0.5:
                idx=np.random.randint(0,self.pool_size)
                tmp=img_pool[idx]
                img_pool[idx]=img
                return tmp
            else:
                return img
        
    def instance_norm(self,x,name):
        with tf.variable_scope(name):
            epsilon=1e-5
            mean,var=tf.nn.moments(x,[1,2],keep_dims=True)
            scale=tf.get_variable('scale',[x.get_shape()[-1]],initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.02))
            offset=tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
            standard_x=(x-mean)/(tf.sqrt(var+epsilon))
            output=scale*standard_x+offset
            return output
        
    def batch_norm(self,x,name):
        return tf.contrib.layers.batch_norm(x,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,scope=name)
    
    def lrelu(self,x,leak=0.2):
        return tf.maximum(x,leak*x)
		
def read_img(path_list):
    img_path=glob(path_list+'/*.jpg')
    img_pool=[(scipy.misc.imread(path,mode='RGB').astype(np.float32))/127.5-1. for path in img_path]
    return np.array(img_pool)

def shuffle(img_pool):
    total_batch=len(img_pool)
    idx=np.random.permutation(total_batch)
    shuffle_pool=img_pool[idx]
    return shuffle_pool

