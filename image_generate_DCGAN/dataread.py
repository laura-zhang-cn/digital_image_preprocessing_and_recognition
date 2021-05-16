# -*- coding:utf-8 -*-
'''
使用DCGAN生成头像图片
DeepConverlutional GAN
'''

import os
import cv2 
import numpy as npy
from matplotlib import pyplot as plt

import tensorflow as tf

def read_images_data(rs=112,batchsize=64):
    '''
    与main中一样，供算法模块中去调用
    '''
    flp='E:\\google_download\\image_wang\\'
    flns=os.listdir(flp)
    imgs=[];error_imgs=[]
    for flx in flns:
        try:
            imgx=cv2.imread(flp+flx)
            imgy=cv2.cvtColor(imgx,cv2.COLOR_BGR2GRAY)
            imgz=cv2.resize(imgy,(rs,rs))    
            imgs.append(imgz)
        except:
            error_imgs.append(flx)
    train_data=npy.array(imgs)
    #
    n=train_data.shape[0]
    train_data=train_data.reshape(n,rs,rs,1).astype('float32')
    scl=255/2
    train_data=(train_data-scl)/scl #[-1,1]
    #
    train_set=tf.data.Dataset.from_tensor_slices(train_data).shuffle(n).batch(batchsize)
    return train_set,error_imgs

if __name__=='__main__':
    flp='E:\\google_download\\image_wang\\'
    flns=os.listdir(flp)
    imgs=[]
    for flx in flns:
        flx='u=2191650517,1787497015&fm=26&gp=0.jpg'
        imgx=cv2.imread(flp+flx)
        imgy=cv2.cvtColor(imgx,cv2.COLOR_BGR2GRAY)
        imgz=cv2.resize(imgy,(112,112))    
        
        plt.imshow(imgx)
        plt.imshow(imgy)
        plt.imshow(imgz)
        imgs.append(imgz)
    train_data=npy.array(imgs)
    train_data.shape
