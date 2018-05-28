# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:04:32 2018

@author: zhangyaxu

canny edge detector canny边缘检测算法
"""

from matplotlib import pyplot as plt
import cv2

img1=cv2.imread('roi_football.jpg')
img11=cv2.Canny(img1,50,150)
img12=cv2.Canny(img1,100,200)
img13=cv2.Canny(img1,150,250)
img14=cv2.Canny(img1,150,500)
img15=cv2.Canny(img1,300,500)

plt.figure(figsize=(10,8))
plt.subplot(3,2,1)
plt.imshow(img1)
plt.title('original piture')
plt.subplot(3,2,2)
plt.imshow(img11,cmap='gray')
plt.title('threthold: 50,150')
plt.subplot(3,2,3)
plt.imshow(img12,cmap='gray')
plt.title('threthold: 100,200')
plt.subplot(3,2,4)
plt.imshow(img13,cmap='gray')
plt.title('threthold: 150,250')
plt.subplot(3,2,5)
plt.imshow(img14,cmap='gray')
plt.title('threthold: 150,500')
plt.subplot(3,2,6)
plt.imshow(img15,cmap='gray')
plt.title('threthold: 300,500')
plt.tight_layout()
plt.show()

