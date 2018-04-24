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


fpath='E:\\spyder_workfiles\\test0\\image_algorithm'
fl0='\\sim30.jpg'
fl1='\\sim32.jpg'

img0=cv2.imread(fpath+fl0)
img1=cv2.imread(fpath+fl1)

img0new=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
img1new=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img0new2=cv2.resize(img0new,(img1new.shape[1],img1new.shape[0]))

plt.imshow(img0new2)
plt.show()
plt.imshow(img1new)
plt.show()

# 使用 sift 
sift = cv2.xfeatures2d.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0new2,None)  # find keypoints and des is a array data
kp1, des1 = sift.detectAndCompute(img1new,None)  # find keypoints and des is a array data

img00=cv2.drawKeypoints(img0new2,kp0,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img11=cv2.drawKeypoints(img1new,kp1,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

plt.subplot(1,2,1)
plt.imshow(img00)
plt.subplot(1,2,2)
plt.imshow(img11)
plt.suptitle('sift')
plt.show()

# 使用 surf 
surf = cv2.xfeatures2d.SURF_create()

k=10000 # 参数 hessian边界值
surf.setHessianThreshold(k)
kp0, des0 = surf.detectAndCompute(img0new2,None)  # find keypoints and des is a array data
kp1, des1 = surf.detectAndCompute(img1new,None)  # find keypoints and des is a array data

img00=cv2.drawKeypoints(img0new2,kp0,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img11=cv2.drawKeypoints(img1new,kp1,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

plt.subplot(1,2,1)
plt.imshow(img00)
plt.subplot(1,2,2)
plt.imshow(img11)
plt.suptitle('surf \n(setHessianThreshold={0})'.format(k))
plt.show()

k=5000 # 参数 hessian边界值
surf.setHessianThreshold(k)
kp0, des0 = surf.detectAndCompute(img0new2,None)  # find keypoints and des is a array data
kp1, des1 = surf.detectAndCompute(img1new,None)  # find keypoints and des is a array data

img00=cv2.drawKeypoints(img0new2,kp0,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img11=cv2.drawKeypoints(img1new,kp1,None,(255,0,0),4)  # ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

plt.subplot(1,2,1)
plt.imshow(img00)
plt.subplot(1,2,2)
plt.imshow(img11)
plt.suptitle('surf \n(setHessianThreshold={0})'.format(k))
plt.show()
