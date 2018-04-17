# -*- coding: utf-8 -*-

"""
Created on Tue Apr  3 11:26:11 2018

@author: zhangyaxu
"""

import cv2
from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc",size = 14)

import numpy as np

# 膨胀算法 Kernel
_DILATE_KERNEL = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0]], dtype=np.uint8)


class WatermarkRemover(object):
    """"
    去除图片中的水印(Remove Watermark)
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.watermark_template_gray_img = None
        self.watermark_template_mask_img = None
        self.watermark_template_h = 0
        self.watermark_template_w = 0

    def load_watermark_template(self, watermark_template_filename):
        """
        加载水印模板，以便后面批量处理去除水印
        :param watermark_template_filename:
        :return:
        """
        self.generate_template_gray_and_mask(watermark_template_filename)

    def dilate(self, img):
        """
        对图片进行膨胀计算
        :param img:
        :return:
        """
        dilated = cv2.dilate(img, _DILATE_KERNEL)
        return dilated

    def generate_template_gray_and_mask(self, watermark_template_filename):
        """
        处理水印模板，生成对应的检索位图和掩码位图
        检索位图
            即处理后的灰度图，去除了非文字部分
        :param watermark_template_filename: 水印模板图片文件名称
        :return: x1, y1, x2, y2
        """

        # 水印模板原图
        img = cv2.imread(watermark_template_filename)

        # 灰度图、掩码图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        mask = self.dilate(mask)  # 使得掩码膨胀一圈，以免留下边缘没有被修复
        
        # 水印模板原图去除非文字部分
        img = cv2.bitwise_and(img, img, mask=mask)
        plt.imshow(img)
        plt.show()

        # 后面修图时需要用到三个通道
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self.watermark_template_gray_img = gray
        self.watermark_template_mask_img = mask

        self.watermark_template_h = img.shape[0]
        self.watermark_template_w = img.shape[1]

        return gray, mask


    def find_watermark_from_gray(self, gray_img, watermark_template_gray_img):
        """
        从原图的灰度图中寻找水印位置
        :param gray_img: 原图的灰度图
        :param watermark_template_gray_img: 水印模板的灰度图
        :return: x1, y1, x2, y2
        """
        # Load the images in gray scale

        method = cv2.TM_CCOEFF
        # Apply template Matching
        res = cv2.matchTemplate(gray_img, watermark_template_gray_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc

        return x, y, x + self.watermark_template_w, y + self.watermark_template_h

    def remove_watermark_raw(self, img, watermark_template_gray_img, watermark_template_mask_img):
        """
        去除图片中的水印
        :param img: 待去除水印图片位图
        :param watermark_template_gray_img: 水印模板的灰度图片位图，用于确定水印位置
        :param watermark_template_mask_img: 水印模板的掩码图片位图，用于修复原始图片
        :return: 去除水印后的图片位图
        """
        # 寻找水印位置
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(img_gray)
        plt.show()
        x1, y1, x2, y2 = self.find_watermark_from_gray(img_gray, watermark_template_gray_img)
        print('warter mark location (x1,x2),(y1,y2) =({0},{2}),({1},{3}): ',(x1,y1,x2,y2))
        
        # 制作原图的水印位置遮板
        mask = np.zeros(img.shape, np.uint8)
        mask[y1:y2, x1:x2] = watermark_template_mask_img
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        plt.imshow(mask)
        plt.show()

        # 用遮板进行图片修复，使用 TELEA 算法
        dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        # cv2.imwrite('dst.jpg', dst)

        return dst

    def remove_watermark(self, filename, output_filename=None):
        """
        去除图片中的水印
        :param filename: 待去除水印图片文件名称
        :param output_filename: 去除水印图片后的输出文件名称
        :return: 去除水印后的图片位图
        """

        # 读取原图
        img = cv2.imread(filename)
        dst = self.remove_watermark_raw(img,
                                        self.watermark_template_gray_img,
                                        self.watermark_template_mask_img
                                        )

        if output_filename is not None:
            cv2.imwrite(output_filename, dst)
        return dst


path = 'E:\\spyder_workfiles\\test0\\image_algorithm'

## 测试房天下
watermark_template_filename = path + '\\wm4_fangtianxia-watermark-template.jpg'
remover = WatermarkRemover()
remover.load_watermark_template(watermark_template_filename)

rst0=remover.remove_watermark(path + '\\wm0_watermark.jpg')
rst1=remover.remove_watermark(path + '\\wm1_watermark.jpg')
rst2=remover.remove_watermark(path + '\\wm2_watermark.jpg')
rst3=remover.remove_watermark(path + '\\wm3_watermark.jpg')
rst4=remover.remove_watermark(path + '\\wm4_watermark.jpg')

rst=[rst0,rst1,rst2,rst3,rst4]
plt.figure(figsize=(20,15))
for ii in range(5):
    plt.subplot(2,3,ii+1)
    plt.imshow(rst[ii])
    plt.title('{}'.format(ii))
plt.show()

# 测试安居客
#watermark_template_filename = path + '\\wm6_anjuke-watermark-template.jpg'
#remover = WatermarkRemover()
#remover.load_watermark_template(watermark_template_filename)
#
#rst0=remover.remove_watermark(path + '\\wm5_ajk.jpg')
#rst1=remover.remove_watermark(path + '\\wm7_ajk.jpg')
#
#rst=[rst0,rst1]
#plt.figure(figsize=(20,15))
#for ii in range(2):
#    plt.subplot(1,2,ii+1)
#    plt.imshow(rst[ii])
#    plt.title('{}'.format(ii))
#plt.show()
