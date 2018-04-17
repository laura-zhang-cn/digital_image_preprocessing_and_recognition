# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:14:42 2018

@author: zhangyaxu

从“颜色”这个图像特征出发

制作颜色直方图，然后比较颜色直方图的相似度来度量图片相似度

其它图像特征还有  纹理 、 形状、空间 ，但颜色这个特征是最方便计算的，所以先从颜色入手


颜色直方图 本身的是划分颜色的分布， 如果一个人换了件衣服，其它完全相同，都会导致相似度明显降低

"""

from PIL import Image as img

from matplotlib import pyplot as plt
import pandas as pds

from sklearn.metrics import pairwise_distances
#from scipy.spatial.distance import cosine

def split_img(imgx,split_nums=4):
    '''
    把图片 垂直和水平方向都分成  split_nums 个，即总小图片有 split_nums平方个, 默认4*4=16个
    '''
    x,y=imgx.size
    vstep=int(x/split_nums*1.0)
    hstep=int(y/split_nums*1.0)
    imgs=[imgx.crop((i*vstep,j*hstep,(i+1)*vstep,(j+1)*hstep)).copy() for j in range(split_nums) for i in range(split_nums)]
    
    k=1; 
    for imgi in imgs:
        plt.subplot(split_nums,split_nums,k)
        plt.imshow(imgi)
        k=k+1
    plt.show()
    
    return imgs

def img_histogram(imgx,rsize=(256,256)):
    '''
    读取图片，并计算颜色直方图
    '''
    imgx=imgx.resize(rsize).convert('RGB')
    hx=imgx.histogram()
    return hx

def hist_similar_total(hx):
    '''
    比较全局的颜色直方图分布来衡量相似度 ，使用 差距比例值
    hx ： 存储了=2个的图片颜色直方图的列表 ,形如 [[hx1],[hx2]]
    '''
    assert len(hx)==2
    hz=pds.DataFrame(hx).T # m*n维  m图片数量 ，n 颜色直方图的样点数， pil为RGB模式计算的样点数都是n=768个
    hz['each_sim']=1.0-hz.apply(lambda x: abs(x[0]-x[1])*1.0/max(x[0],x[1],1.0),axis=1)
    return hz.each_sim.mean()

def hist_similar_total_cosine(hx):
    '''
    比较全局的颜色直方图分布来衡量相似度 ,这里用余弦相似度
    hx ： 存储了≥2个的图片颜色直方图的列表 ,形如 [[hx1],[hx2],..]
    hx 长度不宜过长，否则 计算速度慢， 或 内存溢出
    '''
    assert len(hx)>1
    hz=pds.DataFrame(hx) # m*n维  m图片数量 ，n 颜色直方图的样点数， pil为RGB模式计算的样点数都是n=768个
    hz=hz.apply(lambda x: x*1.0/max(x),axis=1)
    sim_mat=1.0-pairwise_distances(hz.values,metric='cosine')
    return sim_mat

def sub_hist_sim_avg(sub_ls):
    imgx_ls,imgy_ls=sub_ls
    return  sum([hist_similar_total([img_histogram(imgx_ls[ix]),img_histogram(imgy_ls[ix])]) for ix in range(len(imgx_ls))])/len(imgx_ls)

def sub_hist_sim_max(sub_ls):
    imgx_ls,imgy_ls=sub_ls
    sub_hist_sim_tem=pds.DataFrame([
            [ix,iy,hist_similar_total([img_histogram(imgx_ls[ix]),img_histogram(imgy_ls[iy])])] for ix in range(len(imgx_ls)) for iy in range(len(imgy_ls))
            ],columns=['x','y','each_sim']).sort_values(by='each_sim',ascending=False).reset_index(drop=True) # 16*16次循环，慢
    
    sub_imx=[] # 已获取最大相似度的子图集，下次不再重复获取
    sub_imy=[]
    each_sim=[]
    for tem in range(len(imgx_ls)):
        flush_sub=sub_hist_sim_tem.loc[(sub_hist_sim_tem['x'].isin(sub_imx)==False) & (sub_hist_sim_tem['y'].isin(sub_imy)==False),:]
        each_sim.append(flush_sub.iloc[0,2])
        sub_imx.append(flush_sub.iloc[0,0]) # 把新的两个关联子图 添加到已获取最大相似度的子图集中，下次不再重复获取
        sub_imy.append(flush_sub.iloc[0,1])
    sub_hist_sim_max=sum(each_sim)/len(imgx_ls)
    return sub_hist_sim_max

if __name__=='__main__':
    # 图片读取
    pathx='E:\\spyder_workfiles\\test0\\image_algorithm'
    fx='\\sim61.png'
    fy='\\sim64.png'
    imgx=img.open(pathx+fx)
    imgy=img.open(pathx+fy)
    plt.subplot(1,2,1)
    plt.imshow(imgx)
    plt.subplot(1,2,2)
    plt.imshow(imgy)
    plt.show()
    
    ## 1 整图
    # 直接 两个计算 整图相似度
    hx=img_histogram(imgx,rsize=(256,256))
    hy=img_histogram(imgy,rsize=(256,256))
    total_hist_sim=hist_similar_total([hx,hy])
    print('\n total_hist_sim :',total_hist_sim)
    plt.plot(hx)
    plt.plot(hy)
    plt.show()
    # 可用于更多图片的 ，计算 整图相似度
    total_hist_sim_cosine=hist_similar_total_cosine([hx,hy])
    print('\n total_hist_sim_cosine : \n',total_hist_sim_cosine)
    
    ## 2 分割图，子图 
    
    imgx_ls=split_img(imgx,split_nums=4)
    imgy_ls=split_img(imgy,split_nums=4)
    
    # 通过计算对应位置子图 计算整图相似度
    sub_hist_sim_avg=sub_hist_sim_avg(sub_ls=[imgx_ls,imgy_ls])
    print('\n sub_hist_sim_avg :',sub_hist_sim_avg)
    
    #  通过迭代计算子图相似度，每个子图保留与之相似度最高同时不是另一张图相似度最高的图的相似度 （子图 a1与b1相似度最高为0.8，a2与b1相似度最高为0.7,则使用a1与b1的相似度，a2再另寻与其它子图相似度最高的）， 计算整图相似度
    sub_hist_sim_max=sub_hist_sim_max(sub_ls=[imgx_ls,imgy_ls])
    print('\n sub_hist_sim_max :',sub_hist_sim_max)
