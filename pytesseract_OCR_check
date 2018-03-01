# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:43:14 2018

@author: zhangyaxu
"""

from PIL import Image  
import pytesseract    

def image_text(image_file,langx='chi_sim+eng'):    
	# 利用tesseract进行图片文字识别     
	imagex=Image.open(image_file)      
	image_text=pytesseract.image_to_string(imagex,lang=langx)      
	return image_text
	
if __name__=='__main__':      
	# 图片地址和图片名称列表需要根据本机实际情况修改      
	filex='/Users/zhangyaxu/Pictures/'      
	imagelist=['jpeg1.jpg','jpeg2.jpeg','jpeg3.jpeg','jpeg4.jpeg','jpeg5.jpeg','jpeg6.jpg']     
	
	# 存放每个图片识别的结果      
	text=[]       
	
	# 循环读取路径下的图片并识别，识别结果放入text列表中     
	for ii in range(len(imagelist)):         
		image_file=filex+imagelist[ii]          
		text.append(image_text(image_file))  # 调用识别函数       
	
	print(text)
    
  
