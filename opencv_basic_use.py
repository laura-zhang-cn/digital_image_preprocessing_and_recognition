# -*- coding:utf-8 -*-

import cv2
from matplotlib import pyplot as plt

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc",size = 14)

######### 1 载入和显示图像
fpath='E:\\spyder_workfiles\\test0\\image_algorithm'
im = cv2.imread(fpath+'\\space.jpg')   # image read on BGR-color-type  

# 颜色空间转换
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  #  convert color from BGR TO GRAY

rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  #  convert color from BGR TO RGB

# 显示cv读取的BGR 图像
fig = plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
#plt.imshow(im)
plt.imshow(im,cmap='gray',interpolation='bicubic')
plt.title('BGR彩色图', fontproperties= font)
plt.axis('off')

# 显示RGB图像
plt.subplot(1,3,2)
#plt.imshow(rgb)
plt.imshow(rgb,cmap='gray',interpolation='bicubic')
plt.title('RGB彩色图', fontproperties= font)
plt.axis('off')

# 显示灰度化图像
plt.subplot(1,3,3)
#plt.imshow(gray)
plt.imshow(gray,cmap='gray',interpolation='bicubic')
plt.title('2gray灰度图', fontproperties= font)
plt.axis('off')

plt.show()


########## 2 在一个img上绘图：如 线  矩形  圆形 半圆 多边形 文本等
import numpy as npy
# Create a black image
img = npy.zeros((512,512,3), npy.uint8)
plt.imshow(img)

# Draw a diagonal red(in RGB, blue in BGR) line with thickness of 5 px
im_line=cv2.line(img,(0,0),(511,511),(255,0,0),5) # 添加到 img 上
plt.imshow(img)  # plt is RGB so (255,0,0) is red ,but cv2 is BGR so (255,0,0) is blue

#Drawing Rectangle on img
im_rectangle=cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)# 添加到 img 上
plt.imshow(img)  

#Drawing Circle on img
im_circle=cv2.circle(img,(447,63), 63, (0,0,255), -1)  # 添加到 img 上
plt.imshow(img) 


#Drawing Ellipse
im_ellipse=cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
plt.imshow(img)

#Drawing Polygon 多边形
pts = npy.array([[10,5],[20,30],[70,20],[50,10]], npy.int32)
pts = pts.reshape((-1,1,2))
im_ploylines=cv2.polylines(img,[pts],True,(0,255,255))
plt.figure(figsize=(8,8))
plt.imshow(img)

#Adding Text to Images:
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
plt.figure(figsize=(8,8))
plt.imshow(img)


############## 3 img Region Of Interest 
img=cv2.imread('E:\\spyder_workfiles\\test0\\image_algorithm\\roi.jpg')
plt.imshow(img)
plt.show()
 # covert to RGB
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# get ROI 
ball=img_rgb[230:260,90:140].copy()
old=img_rgb[240:270,350:400].copy()

# copy the ball region  to another region
img_rgb[240:270,350:400]=ball
plt.imshow(img_rgb)
plt.show()
# regain the covered region
img_rgb[240:270,350:400]=old
plt.imshow(img_rgb)
plt.show()

############## 4 img split RGB AND merge RGB  ,  Index r g b 
img=cv2.imread('E:\\spyder_workfiles\\test0\\image_algorithm\\roi.jpg')
plt.imshow(img)
plt.show()

# split ,merge method
b,g,r=cv2.split(img)
img2=cv2.merge((r,g,b))  # result =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

# Index method
r=img[:,:,2]
g=img[:,:,1]
b=img[:,:,0]

############## 5 img  copyMakeBorder  , making borders for image (padding)

img=cv2.imread('E:\\spyder_workfiles\\test0\\image_algorithm\\space3.jpg')
plt.imshow(img)
plt.show()

colr=[255,0,255]

img1=cv2.copyMakeBorder(img,5,10,5,10,cv2.BORDER_CONSTANT,value=colr) # 固定颜色padding
img2=cv2.copyMakeBorder(img,10,10,20,20,cv2.BORDER_REFLECT) # 边缘对称，重复最外边一次
img3=cv2.copyMakeBorder(img,12,34,12,32,cv2.BORDER_REFLECT_101)# 边缘对称，不重复最外边
img4=cv2.copyMakeBorder(img,15,15,10,10,cv2.BORDER_REPLICATE)# 边缘拉伸
shp_lr=img.shape[1] ;shp_ud=img.shape[0]
img5=cv2.copyMakeBorder(img,shp_ud,shp_ud,shp_lr,shp_lr,cv2.BORDER_WRAP) # 图片平铺（只留所需宽度的像素）

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.imshow(img);plt.title('original')
plt.subplot(2,3,2)
plt.imshow(img1);plt.title('constant')
plt.subplot(2,3,3)
plt.imshow(img2);plt.title('reflect')
plt.subplot(2,3,4)
plt.imshow(img3);plt.title('reflect 101')
plt.subplot(2,3,5)
plt.imshow(img4);plt.title('replicate')
plt.subplot(2,3,6)
plt.imshow(img5);plt.title('wrap')
plt.show()

