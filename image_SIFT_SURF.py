# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:14:22 2018

@author: zhangyaxu

SIFT  SURF 等，是物体辨识、手势辨识、地图感知的

python3的安装：
pip install opencv-contrib-python

如果已经安装OpenCv2，则需要先卸载pip uninstall opencv-python再安装
"""

import cv2
from matplotlib import pyplot as plt 

import pandas as pds
from sklearn.metrics import pairwise_distances
import numpy as npy

from datetime import datetime as dtm


### 第一步 使用 sift 或surf 提取特征矩阵算法调用

def keypoints_feature(imgx,method='sift',ht=5000,spare):
    if method=='sift':
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(imgx,None)
        imgy=cv2.drawKeypoints(imgx,kp,None,(255,0,0),4)
    elif method=='surf':
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(ht)
        kp, des = surf.detectAndCompute(imgx,None)
        imgy=cv2.drawKeypoints(imgx,kp,None,(255,0,0),4) 
    return kp,des,imgy

### 第二步 计算特征矩阵相似度算法
def sim(des0x):
    #生成余弦相似度矩阵
    #sim=1-pairwise_distances(npy.concatenate((des0x.reshape(1,-1),des1)),metric='cosine')
    sim=pairwise_distances(npy.concatenate((des0x.reshape(1,-1),des1)))
    sim=sim[0,1:]
    sim=1.0-sim*1.0/(sim.max()+0.1)
    return sim

def sim_total(sim,padding=False):
    '''
    sim ： m×n维的特征点相似度矩阵
    迭代提取sim矩阵中最相似的值，每次提取后从原sim矩阵中剔除该最大值对应的特征点，不在进入下一次提取
    pad 是否补全，因为图片提取的特征点数量是不统一的，
    若padding=False 特征点多的图片就直接过滤掉那部分，
    若padding=True  多出来的那部分无法比较相似度的特征点需要进行补全，这里使用剩余相似度的最小值补全
    '''
    keypoints_sim=[]
    minsp=min(sim.shape)
    maxsp=max(sim.shape)
    shape_sim=minsp*1.0/maxsp
    while sim.shape[0]>0 and sim.shape[1]>0 :
        padx=sim.min()
        keypoints_sim.append(sim.max())
        maxids=npy.where(sim==sim.max())
        rowid=maxids[0][0]
        colid=maxids[1][0]
        try:
            sim=npy.delete(sim,rowid,axis=0)
        except:
            print('del end')
        try:
            sim=npy.delete(sim,colid,axis=1)
        except:
            print('del end')
    if padding and minsp!=maxsp:
        keypoints_sim.extend([padx]*(maxsp-minsp))
    return npy.array(keypoints_sim),shape_sim


if __name__=='__main__':
    # 读取 图片， 并做颜色和大小的基本处理
    fpath='E:\\spyder_workfiles\\test0\\image_algorithm'
    fl1='\\sim32.jpg'
    fl0='\\sim30.jpg'
    fl0='\\sim31.jpg'
    fl0='\\sim41.jpg'
    
    img1=cv2.imread(fpath+fl1)
    img0=cv2.imread(fpath+fl0)
        
    img1new=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img0new=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)

    img0new2=cv2.resize(img0new,(img1new.shape[1],img1new.shape[0]))
    
    plt.figure(figsize=(12,8))
    plt.subplot(2,4,1)
    plt.imshow(img0new2)
    plt.subplot(2,4,2)
    plt.imshow(img1new)
    #plt.show()
    
    # 第一步 提取特征矩阵  (使用 sift 或surf )
    ## 若使用 sift
    
    kp1,des1,img11=keypoints_feature(imgx=img1new,method='sift')
    kp0,des0,img00=keypoints_feature(imgx=img0new2,method='sift')
    plt.subplot(2,4,3)
    plt.imshow(img00)
    plt.subplot(2,4,4)
    plt.imshow(img11)
    plt.suptitle('sift')
    #plt.show()
    '''
    ## 若使用 surf
    
    k=4000
    kp1,des1,img11=keypoints_feature(imgx=img1new,method='surf',ht=k)
    kp0,des0,img00=keypoints_feature(imgx=img0new2,method='surf',ht=k)
    plt.subplot(1,2,1)
    plt.imshow(img00)
    plt.subplot(1,2,2)
    plt.imshow(img11)
    plt.suptitle('surf \n(setHessianThreshold={0})'.format(k))
    plt.show()
    '''
    
    # 第二步 计算特征矩阵相似度 
    sim_mtx=npy.apply_along_axis(sim,1,des0)
    print('提取的特征点数量分别为：',sim_mtx.shape)
    padx=False
    keypoints_sim,shape_sim=sim_total(sim_mtx,padding=padx) 
    sim_coef=keypoints_sim.mean()
    print('特征点相似度 及 特征点数量比例 分别为：',sim_coef,shape_sim)
    plt.subplot(2,4,5)
    plt.axis('off')
    plt.subplot(2,4,(6,7))
    plt.hist(keypoints_sim)
    plt.title('padding={0} \n特征点相似度 分布直方图'.format(padx))
    plt.tight_layout()
    plt.show()
