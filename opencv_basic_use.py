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


