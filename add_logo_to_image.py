# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:10:42 2018

@author: zhangyaxu
"""
from matplotlib import pyplot as plt
import cv2

img1=cv2.imread('roi_football.jpg')
img2=cv2.imread('opencv_logo.jpg')

#plt.figure(figsize=(10,3))
#plt.subplot(1,2,1)
#plt.imshow(img1)
#plt.subplot(1,2,2)
#plt.imshow(img2)
#plt.suptitle('原图')
#plt.show()


# create an img1's roi  that have the same shape with img2
rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]


# Now create a mask of logo and create its inverse mask also
img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
mask_inv=cv2.bitwise_not(mask)  # use in black-out the area in roi #用来涂黑roi中与logo相同的区域

#plt.figure(figsize=(15,3))
#plt.subplot(1,4,1)
#plt.imshow(mask)
#plt.title('logo mask ')
#plt.subplot(1,4,2)
#plt.imshow(mask_inv)
#plt.title('inverse logo mask')

# Now black-out the area of logo in ROI  涂黑roi中与logo相同的区域
img1_bg=cv2.bitwise_and(roi,roi,mask=mask_inv) # 涂黑roi中与logo相同的区域
img2_fg=cv2.bitwise_and(img2,img2,mask=mask) # 保留img2中与logo相同的区域
#plt.subplot(1,4,3)
#plt.imshow(img1_bg)
#plt.title('bitwise_and: roi, inverse logo mask')
#plt.subplot(1,4,4)
#plt.imshow(img2_fg)
#plt.title('bitwise_and: logo, logo mask ')
#plt.show()

dst=cv2.add(img1_bg,img2_fg)  #dst=cv2.add(img1_bg,img2)
img1[0:rows,0:cols]=dst

#plt.figure(figsize=(10,3))
#plt.subplot(1,2,1)
#plt.imshow(dst)
#plt.subplot(1,2,2)
#plt.imshow(img1)
#plt.suptitle('使用bitwise 处理后，image add')
#plt.show()


################ 若不使用 bitwise  ，直接add 
img1=cv2.imread('roi_football.jpg')
img2=cv2.imread('opencv_logo.jpg')

#plt.figure(figsize=(10,3))
#plt.subplot(1,2,1)
#plt.imshow(img1)
#plt.subplot(1,2,2)
#plt.imshow(img2)
#plt.suptitle('原图')
#plt.show()

# create an img1's roi  that have the same shape with img2
rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]

dst=cv2.add(roi,img2)
img1[0:rows,0:cols]=dst

#plt.figure(figsize=(10,3))
#plt.subplot(1,2,1)
#plt.imshow(dst)
#plt.subplot(1,2,2)
#plt.imshow(img1)
#plt.suptitle('直接add ，不使用 bitwise')
#plt.show()
