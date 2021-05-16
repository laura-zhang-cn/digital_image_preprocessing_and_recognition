# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:54:08 2021

@author: yxzha
"""

import os
import numpy as npy
from matplotlib import pyplot as plt
from datetime import datetime as dtm


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose as C2DT
from tensorflow.keras.layers import Conv2D as C2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras import optimizers

import dataread



#定义生成模型
def define_generator_model(k):
    model = Sequential() # 初始化一个序列（网络），接下里会向序列中add层
    #全连接层，
    model.add(Dense(7*7*256, use_bias=False, input_shape=(k,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU()) # 添加了一个很小的γ斜率,避免x<0的永远不会被激活
    #
    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # 不限制batch size
    #第一个转置卷积层，s=1
    model.add(C2DT(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU()) 
    #第二个转置卷积层，s=2,做0填充
    model.add(C2DT(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    #第三个转置卷积层，filter=32,kernal_size=5,s=2,做0填充
    model.add(C2DT(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    #第四个转置卷积层，
    model.add(C2DT(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 16)
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    #第五个转置卷积层，filter=1,kernal_size=5,s=2,做0填充
    model.add(C2DT(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 112, 112, 1)

    return model

#定义判别模型
def define_discriminator_model():
    model = Sequential()
    #第一层：常规的卷积,s=2
    model.add(C2D(32, (5, 5), strides=(2, 2), padding='same',input_shape=[112, 112, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    #第二层：常规的卷积,s=2
    model.add(C2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    #第三层：常规卷积，s=2
    model.add(C2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    #第四层：扁平化展开
    model.add(Flatten()) 
    #第五层：输出层，1位 ，是，或者不是
    model.add(Dense(1))

    return model



    
if __name__=='__main__':
    k=300
    #1 生成器生成一个图
    g_mdl=define_generator_model(k=k)
    noise=npy.random.normal(0,1.0,k).reshape(1,-1) 
    g_img=g_mdl(noise,training=False)
    #g_img.shape
    plt.imshow(g_img[0,:,:,0],cmap='gray')
    
    #2 判别器判断图的真假
    d_mdl=define_discriminator_model()
    rst=d_mdl(g_img)
    rst.numpy() # 0.00023501
    
    #3  01分类的交叉熵损失函数 lossfunc-define
    BCE=BinaryCrossentropy(from_logits=True)
    def discriminator_loss(real_rst,fake_rst):
        #判别器的损失函数
        realloss=BCE(tf.ones_like(real_rst),real_rst) # 对真的判断尽量都为真
        fakeloss=BCE(tf.zeros_like(fake_rst),fake_rst) # 对假的判断尽量都为假
        loss_total=realloss+fakeloss
        return loss_total
    
    def generator_loss(fake_rst):
        gloss=BCE(tf.ones_like(fake_rst),fake_rst) #对假的判断尽量为真 =》生成器越优秀
        return gloss
    
    #4 优化方法optimizer 
    g_optimizer=optimizers.Adam(1e-4)
    d_optimizer=optimizers.Adam(1e-4)
    
    #保存模型
    checkpoint_dir = './training_checkpoints1000'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                     discriminator_optimizer=d_optimizer,
                                     generator=g_mdl,
                                     discriminator=d_mdl)

    #5 训练
    epoch=20
    k=300 # noise dim
    g_examples=3 # generate examples' number
    BATCH_SIZE = 64 # 真实dataset的批大小
    seedx=tf.random.normal([g_examples,k]) # [-1,1]
    
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, k]) # 与待训练的真实dataset的批大小一致
        #梯度带计算梯度
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            g_img = g_mdl(noise, training=True)
            real_rst = d_mdl(images, training=True)
            fake_rst = d_mdl(g_img, training=True)
            
            gen_loss = generator_loss(fake_rst)
            disc_loss = discriminator_loss(real_rst, fake_rst)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, g_mdl.trainable_variables)  #生成器的梯度
        gradients_of_discriminator = disc_tape.gradient(disc_loss, d_mdl.trainable_variables) #判别器的梯度
        
        g_optimizer.apply_gradients(zip(gradients_of_generator, g_mdl.trainable_variables)) # 用梯度更新参数
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_mdl.trainable_variables)) # 用梯度更新参数

    def train(dataset, epochs):
        print(dtm.now())
        g_imgs=[]
        for epochx in range(epochs):
            st= dtm.now()
            print(epochx,'\t',st)
            #
            for image_batch in dataset:
                #print('batch is ',i);i+=1
                train_step(image_batch)
            if (epochx+1)%5==0:
                g_imgx=g_mdl(seedx,training=False) # 每100次迭代生成3张图片进行观察
                g_imgs.append(g_imgx)
                checkpoint.save(file_prefix=checkpoint_prefix) # 保存模型
                #print('epoch {0} run time: {1}s'.format(epochx+1,dtm.now()-st))
        return g_imgs
    
    datasets,error_imgs=dataread.read_images_data(rs=112,batchsize=BATCH_SIZE)
    g_imgs=train(datasets,epoch)
    #看看效果
    plt.figure(figsize=(12,15))
    rows=len(g_imgs);cols=g_examples
    for ix,gx in enumerate(g_imgs):
        for jx in range(gx.shape[0]):
            plt.subplot(rows,cols,ix*3+jx+1)
            plt.imshow(gx[jx,:,:,0]*225/2+225/2,cmap='gray')
            plt.axis('off')
    plt.show()
    
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) #调用最近一次保存的模型

    
    
    
    
    
    
    