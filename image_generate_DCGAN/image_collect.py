# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:03:51 2021

@author: yxzha

数据收集： 男idol的头像
使用urllib去百度搜收集到图片的url,之后爬取图片并保存

"""

import urllib
import re
import random
import time


# 存储函数
def data_write(strx):
    #数据 追加式写入
    write_path1='E:\\google_download\\image_baidu_urllib\\caixukun\\cxk_url.txt'
    f1=open(write_path1,'a+')
    f1.write(strx.encode('gbk', 'ignore').decode('gbk', 'ignore'))
    f1.close()

# 网页读取

def urlopen_headers(urlx):
    user_agents=['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
         'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
         'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
         'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
         'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16'
         ]

    headersx={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Host': 'image.baidu.com',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': random.choice(user_agents)
    }
    url_req=urllib.request.Request(urlx,headers=headersx)
    # open url,最多尝试5次
    continue_time=1
    while continue_time in range(1,6):
        try:
            elements_read=urllib.request.urlopen(url_req,timeout=15).read() 
        except Exception as e:
            print('\t 第{0}次异常：\t'.format(continue_time),repr(e))
            continue_time=continue_time+1
        else:
            continue_time=0
    elements=elements_read.decode('utf-8')
    return elements,continue_time
#页面解析
def url_parse(elements):
    elements=re.sub('(\n|\r|\t)','',elements)
    patx=re.compile(r'thumbURL\W{1,10}([^"]*)')
    rst=re.findall(patx,elements)
    return rst


if __name__=='__main__':
    #1 根据百度搜素结果，收集idol的头像图片urls
    keywords=[u'蔡徐坤头像',u'王一博头像',u'李现头像',u'李易峰头像',u'吴亦凡头像',
              u'杨洋头像',u'鹿晗头像',u'白敬亭头像',u'吴磊头像',u'李钟硕头像',
              u'张艺兴头像',u'肖战头像',u'易烊千玺头像',u'王俊凯头像',u'黄子韬头像',
              u'周杰伦头像',u'胡歌头像',u'刘德华头像',u'黄晓明头像',u'朱一龙头像']
    for keywordx in keywords[10:]:
        print(keywordx)
        keywordx=urllib.parse.quote(keywordx)
        pagen=list(range(0,100,20))
        for pgx in pagen:
            str_pn=str(pgx)
            str_gsm='00'
            urly='http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keywordx + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'    
            elements,continue_time=urlopen_headers(urly)
            imgurls=url_parse(elements)
            print(len(imgurls))
            if len(imgurls)>0:
                rst='\n'.join(imgurls)+'\n'
                data_write(strx=rst)
            else:
                break
            time.sleep(random.randint(2,5))
    #2 读取收集的image-urls
    write_path1='E:\\google_download\\image_baidu_urllib\\caixukun\\cxk_url.txt'
    f1=open(write_path1,'r').read()
    imgurls=f1.split('\n')
    #3 根据image-urls,爬取图片并保存
    error_urls=[]
    flp='E:\\google_download\\image_wang\\'
    for ix,imgurlx in enumerate(imgurls):
        if re.search('ss[0-9]',imgurlx) and ix>=660:
            try:
                imgdt=urllib.request.urlopen(imgurlx).read()
                with open(flp+str(ix)+'.jpg','wb') as f:
                    f.write(imgdt)
            except:
                error_urls.append((ix,imgurlx))
        if ix%10==0 and ix>=660:
            time.sleep(random.randint(3,10))
        
    
    
    

